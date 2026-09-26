// WGSL for the hybrid Qwen 3.5/3.8 engine: Gated-DeltaNet recurrence (single and
// multi-column, with speculative snapshot slots), gated attention glue, fused pre-pass,
// and the logits argmax. Base kernels come from ./base.js; GEMVs from ./coop.js.
// Register-resident single-token dn_delta (decode): the same transform as dn_delta_mc below, for
// the one-column bindings. Same operations in the same order as the kernel it replaces, so the
// state and output are bit-identical. Requires dState = 128.
function dnDelta1RegsWGSL() {
  const rows = Array.from({ length: 128 }, (_, i) => i);
  const load = rows.map((i) => `s[${i}u] = dl_s[Sb + ${i * 128}u + j];`).join(" ");
  const store = rows.map((i) => `dl_s[Sb + ${i * 128}u + j] = s[${i}u];`).join(" ");
  const loop1 = rows.map((i) => `{ let sd = s[${i}u] * decay; s[${i}u] = sd; vh += sd * dl1_k[${i}u]; sq += sd * dl1_q[${i}u]; kq += dl1_k[${i}u] * dl1_q[${i}u]; }`).join("\n  ");
  const loop2 = rows.map((i) => `s[${i}u] += dl1_k[${i}u] * d;`).join(" ");
  return `
var<workgroup> dl1_k: array<f32, 128>;
var<workgroup> dl1_q: array<f32, 128>;
@compute @workgroup_size(128)
fn dn_delta(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x; let j = lid.x;
  let kh = h % dl_dn.nKH;
  let kOff = kh * 128u; let vOff = h * 128u; let Sb = h * 16384u;
  let decay = dl_decay[h];
  let scale = inverseSqrt(f32(dl_dn.dState));
  dl1_k[j] = dl_k[kOff + j]; dl1_q[j] = dl_q[kOff + j];
  var s: array<f32, 128>;
  ${load}
  workgroupBarrier();
  var vh: f32 = 0.0; var sq: f32 = 0.0; var kq: f32 = 0.0;
  ${loop1}
  let d = (dl_v[vOff + j] - vh) * dl_beta[h];
  ${loop2}
  dl_o[vOff + j] = (sq + d * kq) * scale;
  ${store}
}`;
}

// Register-resident dn_delta_mc (docs/deltanet-prefill-spec.md, RG=1): thread j keeps column j
// of the head's state S in 128 registers for the whole pass (loaded once, stored once) instead of
// two read-modify-write sweeps of global memory per column; q/k of each column are staged in
// workgroup memory. Same operations in the same order as the kernel it replaces, so S, the
// outputs and the snapshots are bit-identical (measured on the GB10 in isolation: 1.24x at 1
// column, 2.03x at 16). Every private-array index is a literal (codegen-unrolled): a loop over a
// constant bound does not unroll on every backend and spills the array. Requires dState = 128.
function dnDeltaRegsWGSL() {
  const rows = Array.from({ length: 128 }, (_, i) => i);
  const load = rows.map((i) => `s[${i}u] = dlm_s[Sb + ${i * 128}u + j];`).join(" ");
  const store = rows.map((i) => `dlm_s[Sb + ${i * 128}u + j] = s[${i}u];`).join(" ");
  const shadow = rows.map((i) => `dlm_shadow[so + ${i * 128}u] = s[${i}u];`).join(" ");
  const loop1 = rows.map((i) => `{ let sd = s[${i}u] * decay; s[${i}u] = sd; vh += sd * dlr_k[${i}u]; sq += sd * dlr_q[${i}u]; kq += dlr_k[${i}u] * dlr_q[${i}u]; }`).join("\n      ");
  const loop2 = rows.map((i) => `s[${i}u] += dlr_k[${i}u] * d;`).join(" ");
  return `
var<workgroup> dlr_k: array<f32, 128>;
var<workgroup> dlr_q: array<f32, 128>;
@compute @workgroup_size(128)
fn dn_delta_mc(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x; let j = lid.x;
  let kh = h % dlm_dn.nKH;
  let kOff = kh * 128u; let vOff = h * 128u; let Sb = h * 16384u;
  let scale = inverseSqrt(f32(dlm_dn.dState));   // a uniform read, as before (a literal changes rounding)
  let nCols = max(frame.nCols, 1u);
  let sSize = dlm_dn.nVH * 16384u;
  var s: array<f32, 128>;
  ${load}
  for (var col: u32 = 0u; col < nCols; col++) {
    let qo = col * dlm_mc.s0 + kOff;
    let ko = col * dlm_mc.s0 + dlm_dn.keyDim + kOff;
    let vo = col * dlm_mc.s0 + 2u * dlm_dn.keyDim + vOff;
    workgroupBarrier();                          // the previous column is done reading k/q
    dlr_k[j] = dlm_c[ko + j]; dlr_q[j] = dlm_c[qo + j];
    workgroupBarrier();
    let decay = dlm_decay[col * dlm_mc.s1 + h];
    var vh: f32 = 0.0; var sq: f32 = 0.0; var kq: f32 = 0.0;
      ${loop1}
    let d = (dlm_c[vo + j] - vh) * dlm_beta[col * dlm_mc.s1 + h];
    ${loop2}
    dlm_o[col * dlm_mc.s2 + vOff + j] = (sq + d * kq) * scale;
    let dlSB = frame.snap & 0xffu;               // snapshot slot base + 1 (0 = off); bit 31: replay rollback, no state snapshots
    if (dlSB != 0u && (frame.snap & 0x80000000u) == 0u && dlSB + col < ((frame.snap >> 8u) & 0xffu)) {
      let so = (dlSB - 1u + col) * sSize + Sb + j;
      ${shadow}
    }
  }
  ${store}
}`;
}

export const WGSL2 = /* wgsl */ `
struct DN {
  convDim: u32, dState: u32, nKH: u32, nVH: u32,
  keyDim: u32, nRot: u32, hd: u32, dInner: u32,
  ropeTheta: f32, eps2: f32, pad0: u32, pad1: u32,
};

// --- causal conv (K=4) + silu; updates rolling state ---
@group(1) @binding(0) var<storage, read> cv_x: array<f32>;      // qkv_mixed [convDim]
@group(1) @binding(1) var<storage, read> cv_w: array<f32>;      // [convDim, 4]
@group(1) @binding(2) var<storage, read_write> cv_st: array<f32>; // [convDim, 3]
@group(1) @binding(3) var<storage, read_write> cv_y: array<f32>;  // [convDim]
@group(1) @binding(4) var<uniform> cv_dn: DN;
@compute @workgroup_size(64)
fn dn_conv(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= cv_dn.convDim) { return; }
  let x = cv_x[c];
  var acc = cv_w[c * 4u + 3u] * x;
  acc += cv_w[c * 4u + 0u] * cv_st[c * 3u + 0u];
  acc += cv_w[c * 4u + 1u] * cv_st[c * 3u + 1u];
  acc += cv_w[c * 4u + 2u] * cv_st[c * 3u + 2u];
  cv_y[c] = acc / (1.0 + exp(-acc));
  cv_st[c * 3u + 0u] = cv_st[c * 3u + 1u];
  cv_st[c * 3u + 1u] = cv_st[c * 3u + 2u];
  cv_st[c * 3u + 2u] = x;
}

// --- per-head gates: beta = sigmoid(betaRaw); decay = exp(softplus(alpha+dt)*A) ---
@group(1) @binding(0) var<storage, read> gt_alpha: array<f32>;
@group(1) @binding(1) var<storage, read> gt_beta: array<f32>;
@group(1) @binding(2) var<storage, read> gt_dt: array<f32>;
@group(1) @binding(3) var<storage, read> gt_a: array<f32>;
@group(1) @binding(4) var<storage, read_write> gt_bout: array<f32>;
@group(1) @binding(5) var<storage, read_write> gt_dout: array<f32>;
@group(1) @binding(6) var<uniform> gt_dn: DN;
@compute @workgroup_size(64)
fn dn_gates(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  if (h >= gt_dn.nVH) { return; }
  gt_bout[h] = 1.0 / (1.0 + exp(-gt_beta[h]));
  let av = gt_alpha[h] + gt_dt[h];
  var sp: f32;
  if (av > 20.0) { sp = av; } else { sp = log(1.0 + exp(av)); }
  gt_dout[h] = exp(sp * gt_a[h]);
}

// --- per-head L2 norm (q/k slices bound with offset) ---
@group(1) @binding(0) var<storage, read_write> l2_v: array<f32>;
@group(1) @binding(1) var<uniform> l2_heads: u32;
@group(1) @binding(2) var<uniform> l2_dn: DN;
@compute @workgroup_size(32)
fn dn_l2(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  if (h >= l2_heads) { return; }
  let off = h * l2_dn.dState;
  var ss: f32 = 0.0;
  for (var i: u32 = 0u; i < l2_dn.dState; i++) { let v = l2_v[off + i]; ss += v * v; }
  let inv = 1.0 / max(sqrt(ss), l2_dn.eps2);
  for (var i: u32 = 0u; i < l2_dn.dState; i++) { l2_v[off + i] *= inv; }
}

// --- gated delta rule: one workgroup per v-head, thread j = state column ---
@group(1) @binding(0) var<storage, read> dl_q: array<f32>;      // [keyDim] (l2-normed)
@group(1) @binding(1) var<storage, read> dl_k: array<f32>;      // [keyDim]
@group(1) @binding(2) var<storage, read> dl_v: array<f32>;      // [dInner]
@group(1) @binding(3) var<storage, read> dl_beta: array<f32>;   // [nVH]
@group(1) @binding(4) var<storage, read> dl_decay: array<f32>;  // [nVH]
@group(1) @binding(5) var<storage, read_write> dl_s: array<f32>; // [nVH, dState, dState]
@group(1) @binding(6) var<storage, read_write> dl_o: array<f32>; // [dInner]
@group(1) @binding(7) var<uniform> dl_dn: DN;
${dnDelta1RegsWGSL()}

// --- gated norm: rmsnorm per head (w[dState]) * silu(z) ---
@group(1) @binding(0) var<storage, read> gn_x: array<f32>;   // [dInner]
@group(1) @binding(1) var<storage, read> gn_z: array<f32>;   // [dInner]
@group(1) @binding(2) var<storage, read> gn_w: array<f32>;   // [dState]
@group(1) @binding(3) var<storage, read_write> gn_y: array<f32>;
@group(1) @binding(4) var<uniform> gn_dn: DN;
var<workgroup> gn_partial: array<f32, 128>;
@compute @workgroup_size(128)
fn dn_gatenorm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x;
  let j = lid.x;
  let off = h * gn_dn.dState;
  let v = gn_x[off + j];
  gn_partial[j] = v * v;
  workgroupBarrier();
  var stride: u32 = 64u;
  while (stride > 0u) {
    if (j < stride) { gn_partial[j] += gn_partial[j + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let inv = inverseSqrt(gn_partial[0] / f32(gn_dn.dState) + cfg.eps);
  let z = gn_z[off + j];
  gn_y[off + j] = gn_x[off + j] * inv * gn_w[j] * (z / (1.0 + exp(-z)));
}

// --- split interleaved [q|gate] per head from q_full ---
@group(1) @binding(0) var<storage, read> qs_full: array<f32>;  // [nH*2*hd]
@group(1) @binding(1) var<storage, read_write> qs_q: array<f32>;
@group(1) @binding(2) var<storage, read_write> qs_g: array<f32>;
@group(1) @binding(3) var<uniform> qs_dn: DN;
@compute @workgroup_size(64)
fn qsplit(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = cfg.nH * qs_dn.hd;
  if (idx >= total) { return; }
  let h = idx / qs_dn.hd;
  let i = idx % qs_dn.hd;
  qs_q[idx] = qs_full[h * 2u * qs_dn.hd + i];
  qs_g[idx] = qs_full[h * 2u * qs_dn.hd + qs_dn.hd + i];
}

// --- partial neox rope: rotate first nRot dims of each head ---
@group(1) @binding(0) var<storage, read_write> rp2_v: array<f32>;
@group(1) @binding(1) var<uniform> rp2_heads: u32;
@group(1) @binding(2) var<uniform> rp2_dn: DN;
@compute @workgroup_size(64)
fn rope_part(@builtin(global_invocation_id) gid: vec3<u32>) {
  let half = rp2_dn.nRot / 2u;
  let total = rp2_heads * half;
  let idx = gid.x;
  if (idx >= total) { return; }
  let h = idx / half;
  let i = idx % half;
  let off = h * rp2_dn.hd;
  let ang = f32(frame.pos) * pow(rp2_dn.ropeTheta, -f32(2u * i) / f32(rp2_dn.nRot));
  let c = cos(ang); let s = sin(ang);
  let a = rp2_v[off + i]; let b = rp2_v[off + i + half];
  rp2_v[off + i] = a * c - b * s;
  rp2_v[off + i + half] = b * c + a * s;
}

// --- a *= sigmoid(g) ---
@group(1) @binding(0) var<storage, read_write> sm_a: array<f32>;
@group(1) @binding(1) var<storage, read> sm_g: array<f32>;
@group(1) @binding(2) var<uniform> sm_n: u32;
@compute @workgroup_size(64)
fn sigmoid_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= sm_n) { return; }
  let g = sm_g[i];
  sm_a[i] = sm_a[i] * (1.0 / (1.0 + exp(-g)));
}
// ================= multi-column variants (batched prefill / verify) =================
// One dispatch covers every live column: y = column for the parallel ops; the
// recurrent ops (conv, delta rule) loop over columns inside the kernel and
// write rollback snapshots when frame.snap is set. Math is identical to the
// single-column kernels above (same operation order).
struct MC { n: u32, s0: u32, s1: u32, s2: u32 };

@group(1) @binding(0) var<storage, read> rnm_x: array<f32>;
@group(1) @binding(1) var<storage, read> rnm_w: array<f32>;
@group(1) @binding(2) var<storage, read_write> rnm_y: array<f32>;
@group(1) @binding(3) var<uniform> rnm_mc: MC;          // n, x stride, y stride
var<workgroup> rnm_partial: array<f32, 256>;
@compute @workgroup_size(256)
fn rmsnorm_mc(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x; let n = rnm_mc.n;
  let xo = wg.y * rnm_mc.s0; let yo = wg.y * rnm_mc.s1;
  var ss: f32 = 0.0;
  for (var i: u32 = t; i < n; i += 256u) { let v = rnm_x[xo + i]; ss += v * v; }
  rnm_partial[t] = ss;
  workgroupBarrier();
  var stride: u32 = 128u;
  while (stride > 0u) {
    if (t < stride) { rnm_partial[t] += rnm_partial[t + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let inv = inverseSqrt(rnm_partial[0] / f32(n) + cfg.eps);
  for (var i: u32 = t; i < n; i += 256u) { rnm_y[yo + i] = rnm_x[xo + i] * inv * rnm_w[i]; }
}

@group(1) @binding(0) var<storage, read_write> adm_a: array<f32>;
@group(1) @binding(1) var<storage, read> adm_b: array<f32>;
@group(1) @binding(2) var<uniform> adm_mc: MC;          // n, a stride, b stride
@compute @workgroup_size(64)
fn add_res_mc(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= adm_mc.n) { return; }
  adm_a[gid.y * adm_mc.s0 + i] += adm_b[gid.y * adm_mc.s1 + i];
}

@group(1) @binding(0) var<storage, read> gtm_alpha: array<f32>;
@group(1) @binding(1) var<storage, read> gtm_beta: array<f32>;
@group(1) @binding(2) var<storage, read> gtm_dt: array<f32>;
@group(1) @binding(3) var<storage, read> gtm_a: array<f32>;
@group(1) @binding(4) var<storage, read_write> gtm_bout: array<f32>;
@group(1) @binding(5) var<storage, read_write> gtm_dout: array<f32>;
@group(1) @binding(6) var<uniform> gtm_mc: MC;          // n = heads, s0 = column stride (all four)
@compute @workgroup_size(64)
fn dn_gates_mc(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  if (h >= gtm_mc.n) { return; }
  let o = gid.y * gtm_mc.s0;
  gtm_bout[o + h] = 1.0 / (1.0 + exp(-gtm_beta[o + h]));
  let av = gtm_alpha[o + h] + gtm_dt[h];
  var sp: f32;
  if (av > 20.0) { sp = av; } else { sp = log(1.0 + exp(av)); }
  gtm_dout[o + h] = exp(sp * gtm_a[h]);
}

@group(1) @binding(0) var<storage, read> cvm_x: array<f32>;
@group(1) @binding(1) var<storage, read> cvm_w: array<f32>;
@group(1) @binding(2) var<storage, read_write> cvm_st: array<f32>;
@group(1) @binding(3) var<storage, read_write> cvm_y: array<f32>;
@group(1) @binding(4) var<uniform> cvm_mc: MC;          // n = convDim, x stride, y stride
@group(1) @binding(5) var<storage, read_write> cvm_shadow: array<f32>;   // [7][convDim*3]
@compute @workgroup_size(64)
fn dn_conv_mc(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x; let n = cvm_mc.n;
  if (c >= n) { return; }
  let w0 = cvm_w[c * 4u]; let w1 = cvm_w[c * 4u + 1u]; let w2 = cvm_w[c * 4u + 2u]; let w3 = cvm_w[c * 4u + 3u];
  var s0 = cvm_st[c * 3u]; var s1 = cvm_st[c * 3u + 1u]; var s2 = cvm_st[c * 3u + 2u];
  let nCols = max(frame.nCols, 1u);
  for (var col: u32 = 0u; col < nCols; col++) {
    let x = cvm_x[col * cvm_mc.s0 + c];
    var acc = w3 * x;
    acc += w0 * s0;
    acc += w1 * s1;
    acc += w2 * s2;
    cvm_y[col * cvm_mc.s1 + c] = acc / (1.0 + exp(-acc));
    s0 = s1; s1 = s2; s2 = x;
    let cvSB = frame.snap & 0xffu;       // snapshot slot base + 1 (0 = off)
    if (cvSB != 0u && cvSB + col < ((frame.snap >> 8u) & 0xffu)) {
      let so = (cvSB - 1u + col) * n * 3u + c * 3u;
      cvm_shadow[so] = s0; cvm_shadow[so + 1u] = s1; cvm_shadow[so + 2u] = s2;
    }
  }
  cvm_st[c * 3u] = s0; cvm_st[c * 3u + 1u] = s1; cvm_st[c * 3u + 2u] = s2;
}

@group(1) @binding(0) var<storage, read_write> l2m_v: array<f32>;
@group(1) @binding(1) var<uniform> l2m_mc: MC;          // n = heads, s0 = column stride, s1 = part offset (z = 1 -> k)
@group(1) @binding(2) var<uniform> l2m_dn: DN;
@compute @workgroup_size(32)
fn dn_l2_mc(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  if (h >= l2m_mc.n) { return; }
  let off = gid.y * l2m_mc.s0 + gid.z * l2m_mc.s1 + h * l2m_dn.dState;
  var ss: f32 = 0.0;
  for (var i: u32 = 0u; i < l2m_dn.dState; i++) { let v = l2m_v[off + i]; ss += v * v; }
  let inv = 1.0 / max(sqrt(ss), l2m_dn.eps2);
  for (var i: u32 = 0u; i < l2m_dn.dState; i++) { l2m_v[off + i] *= inv; }
}

@group(1) @binding(0) var<storage, read> dlm_c: array<f32>;      // conv output columns: [q | k | v]
@group(1) @binding(1) var<storage, read> dlm_beta: array<f32>;
@group(1) @binding(2) var<storage, read> dlm_decay: array<f32>;
@group(1) @binding(3) var<storage, read_write> dlm_s: array<f32>;
@group(1) @binding(4) var<storage, read_write> dlm_o: array<f32>;
@group(1) @binding(5) var<uniform> dlm_mc: MC;          // s0 conv stride, s1 gate stride, s2 out stride
@group(1) @binding(6) var<uniform> dlm_dn: DN;
@group(1) @binding(7) var<storage, read_write> dlm_shadow: array<f32>;   // [7][nVH*dState*dState]
${dnDeltaRegsWGSL()}

// --- draft chain: the embedding row of the token the last argmax picked, dequantized on the GPU
// exactly as the host's _embedRowF32 does (f16 scale x small integer: exact in f32), so K drafts
// can run in one submit instead of K round trips through the CPU ---
@group(1) @binding(0) var<storage, read> eg_qs: array<u32>;
@group(1) @binding(1) var<storage, read> eg_sc: array<u32>;
@group(1) @binding(2) var<storage, read> eg_id: array<u32>;
@group(1) @binding(3) var<storage, read_write> eg_out: array<f32>;
@group(1) @binding(4) var<uniform> eg_u: vec4<u32>;       // dim, kind (0 q4, 1 q8), rows
@compute @workgroup_size(64)
fn emb_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; let dim = eg_u.x;
  if (i >= dim) { return; }
  let id = min(eg_id[0], eg_u.z - 1u);
  let nb = dim / 32u; let b = i / 32u; let e = i % 32u;
  let si = id * nb + b;
  let s = unpack2x16float(eg_sc[si >> 1u])[si & 1u];
  var q: f32;
  if (eg_u.y == 0u) {
    let bi = si * 16u + (e % 16u);
    let by = (eg_qs[bi >> 2u] >> ((bi & 3u) * 8u)) & 0xFFu;
    q = f32(i32(select(by & 0xFu, by >> 4u, e >= 16u)) - 8);
  } else {
    let bi = si * 32u + e;
    let by = (eg_qs[bi >> 2u] >> ((bi & 3u) * 8u)) & 0xFFu;
    q = f32(bitcast<i32>(by << 24u) >> 24u);
  }
  eg_out[i] = s * q;
}

// --- argmax over n floats (single workgroup): out = [index, bitcast(value)] ---
@group(1) @binding(0) var<storage, read> am_x: array<f32>;
@group(1) @binding(1) var<storage, read_write> am_out: array<u32>;
@group(1) @binding(2) var<uniform> am_n: vec4<u32>;
var<workgroup> am_v: array<f32, 256>;
var<workgroup> am_i: array<u32, 256>;
@compute @workgroup_size(256)
fn argmax(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x; let n = am_n.x;
  var bv: f32 = -3.402823e38; var bi: u32 = 0xffffffffu;
  for (var i: u32 = t; i < n; i += 256u) {
    let v = am_x[i];
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
  }
  am_v[t] = bv; am_i[t] = bi;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (t < s) {
      let ov = am_v[t + s]; let oi = am_i[t + s];
      if (ov > am_v[t] || (ov == am_v[t] && oi < am_i[t])) { am_v[t] = ov; am_i[t] = oi; }
    }
    workgroupBarrier();
  }
  if (t == 0u) { am_out[0] = am_i[0]; am_out[1] = bitcast<u32>(am_v[0]); }
}


// --- fused DeltaNet pre-pass (after conv): gates (sigmoid beta, decay) on
// threads [2*nKH, 2*nKH+nVH), per-head L2 norm of the q and k heads on threads
// [0, 2*nKH). One dispatch instead of three. ---
@group(1) @binding(0) var<storage, read> pp_alpha: array<f32>;
@group(1) @binding(1) var<storage, read> pp_beta: array<f32>;
@group(1) @binding(2) var<storage, read> pp_dt: array<f32>;
@group(1) @binding(3) var<storage, read> pp_a: array<f32>;
@group(1) @binding(4) var<storage, read_write> pp_bout: array<f32>;
@group(1) @binding(5) var<storage, read_write> pp_dout: array<f32>;
@group(1) @binding(6) var<storage, read_write> pp_v: array<f32>;   // conv output [q heads | k heads | v]
@group(1) @binding(7) var<uniform> pp_dn: DN;
fn pp_l2(off: u32) {
  var ss: f32 = 0.0;
  for (var i: u32 = 0u; i < pp_dn.dState; i++) { let v = pp_v[off + i]; ss += v * v; }
  let inv = 1.0 / max(sqrt(ss), pp_dn.eps2);
  for (var i: u32 = 0u; i < pp_dn.dState; i++) { pp_v[off + i] *= inv; }
}
@compute @workgroup_size(128)
fn dn_pre(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x; let nL2 = 2u * pp_dn.nKH;
  if (t < nL2) { pp_l2(t * pp_dn.dState); }
  else if (t < nL2 + pp_dn.nVH) {
    let h = t - nL2;
    pp_bout[h] = 1.0 / (1.0 + exp(-pp_beta[h]));
    let av = pp_alpha[h] + pp_dt[h];
    var sp: f32;
    if (av > 20.0) { sp = av; } else { sp = log(1.0 + exp(av)); }
    pp_dout[h] = exp(sp * pp_a[h]);
  }
}

@group(1) @binding(0) var<storage, read> ppm_alpha: array<f32>;
@group(1) @binding(1) var<storage, read> ppm_beta: array<f32>;
@group(1) @binding(2) var<storage, read> ppm_dt: array<f32>;
@group(1) @binding(3) var<storage, read> ppm_a: array<f32>;
@group(1) @binding(4) var<storage, read_write> ppm_bout: array<f32>;
@group(1) @binding(5) var<storage, read_write> ppm_dout: array<f32>;
@group(1) @binding(6) var<storage, read_write> ppm_v: array<f32>;
@group(1) @binding(7) var<uniform> ppm_mc: MC;          // s0 = gate column stride, s1 = conv-out column stride
@group(1) @binding(8) var<uniform> ppm_dn: DN;
fn ppm_l2(off: u32) {
  var ss: f32 = 0.0;
  for (var i: u32 = 0u; i < ppm_dn.dState; i++) { let v = ppm_v[off + i]; ss += v * v; }
  let inv = 1.0 / max(sqrt(ss), ppm_dn.eps2);
  for (var i: u32 = 0u; i < ppm_dn.dState; i++) { ppm_v[off + i] *= inv; }
}
@compute @workgroup_size(128)
fn dn_pre_mc(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x; let col = wg.y; let nL2 = 2u * ppm_dn.nKH;
  if (t < nL2) { ppm_l2(col * ppm_mc.s1 + t * ppm_dn.dState); }
  else if (t < nL2 + ppm_dn.nVH) {
    let h = t - nL2; let o = col * ppm_mc.s0;
    ppm_bout[o + h] = 1.0 / (1.0 + exp(-ppm_beta[o + h]));
    let av = ppm_alpha[o + h] + ppm_dt[h];
    var sp: f32;
    if (av > 20.0) { sp = av; } else { sp = log(1.0 + exp(av)); }
    ppm_dout[o + h] = exp(sp * ppm_a[h]);
  }
}

@group(1) @binding(0) var<storage, read> gnm_x: array<f32>;
@group(1) @binding(1) var<storage, read> gnm_z: array<f32>;
@group(1) @binding(2) var<storage, read> gnm_w: array<f32>;
@group(1) @binding(3) var<storage, read_write> gnm_y: array<f32>;
@group(1) @binding(4) var<uniform> gnm_mc: MC;          // s0 x stride, s1 z stride, s2 y stride
@group(1) @binding(5) var<uniform> gnm_dn: DN;
var<workgroup> gnm_partial: array<f32, 128>;
@compute @workgroup_size(128)
fn dn_gatenorm_mc(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x; let j = lid.x;
  let off = h * gnm_dn.dState;
  let xo = wg.y * gnm_mc.s0 + off; let zo = wg.y * gnm_mc.s1 + off; let yo = wg.y * gnm_mc.s2 + off;
  let v = gnm_x[xo + j];
  gnm_partial[j] = v * v;
  workgroupBarrier();
  var stride: u32 = 64u;
  while (stride > 0u) {
    if (j < stride) { gnm_partial[j] += gnm_partial[j + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let inv = inverseSqrt(gnm_partial[0] / f32(gnm_dn.dState) + cfg.eps);
  let z = gnm_z[zo + j];
  gnm_y[yo + j] = gnm_x[xo + j] * inv * gnm_w[j] * (z / (1.0 + exp(-z)));
}

@group(1) @binding(0) var<storage, read> qsm_full: array<f32>;
@group(1) @binding(1) var<storage, read_write> qsm_q: array<f32>;
@group(1) @binding(2) var<storage, read_write> qsm_g: array<f32>;
@group(1) @binding(3) var<uniform> qsm_mc: MC;          // s0 full stride, s1 q stride, s2 g stride
@group(1) @binding(4) var<uniform> qsm_dn: DN;
@compute @workgroup_size(64)
fn qsplit_mc(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = cfg.nH * qsm_dn.hd;
  if (idx >= total) { return; }
  let h = idx / qsm_dn.hd;
  let i = idx % qsm_dn.hd;
  let fo = gid.y * qsm_mc.s0 + h * 2u * qsm_dn.hd;
  qsm_q[gid.y * qsm_mc.s1 + idx] = qsm_full[fo + i];
  qsm_g[gid.y * qsm_mc.s2 + idx] = qsm_full[fo + qsm_dn.hd + i];
}

@group(1) @binding(0) var<storage, read_write> hnm_v: array<f32>;
@group(1) @binding(1) var<storage, read> hnm_w: array<f32>;
@group(1) @binding(2) var<uniform> hnm_mc: MC;          // n = heads, s0 = column stride
@compute @workgroup_size(32)
fn head_norm_mc(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  if (h >= hnm_mc.n) { return; }
  let off = gid.y * hnm_mc.s0 + h * cfg.headDim;
  var ss: f32 = 0.0;
  for (var i: u32 = 0u; i < cfg.headDim; i++) { let v = hnm_v[off + i]; ss += v * v; }
  let inv = inverseSqrt(ss / f32(cfg.headDim) + cfg.eps);
  for (var i: u32 = 0u; i < cfg.headDim; i++) { hnm_v[off + i] *= inv * hnm_w[i]; }
}

@group(1) @binding(0) var<storage, read_write> rpm_v: array<f32>;
@group(1) @binding(1) var<uniform> rpm_mc: MC;          // n = heads, s0 = column stride
@group(1) @binding(2) var<uniform> rpm_dn: DN;
@compute @workgroup_size(64)
fn rope_part_mc(@builtin(global_invocation_id) gid: vec3<u32>) {
  let half = rpm_dn.nRot / 2u;
  let total = rpm_mc.n * half;
  let idx = gid.x;
  if (idx >= total) { return; }
  let h = idx / half;
  let i = idx % half;
  let off = gid.y * rpm_mc.s0 + h * rpm_dn.hd;
  let ang = f32(frame.pos + gid.y) * pow(rpm_dn.ropeTheta, -f32(2u * i) / f32(rpm_dn.nRot));
  let c = cos(ang); let s = sin(ang);
  let a = rpm_v[off + i]; let b = rpm_v[off + i + half];
  rpm_v[off + i] = a * c - b * s;
  rpm_v[off + i + half] = b * c + a * s;
}

@group(1) @binding(0) var<storage, read_write> smm_a: array<f32>;
@group(1) @binding(1) var<storage, read> smm_g: array<f32>;
@group(1) @binding(2) var<uniform> smm_mc: MC;          // n, a stride, g stride
@compute @workgroup_size(64)
fn sigmoid_mul_mc(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= smm_mc.n) { return; }
  let g = smm_g[gid.y * smm_mc.s1 + i];
  let ai = gid.y * smm_mc.s0 + i;
  smm_a[ai] = smm_a[ai] * (1.0 / (1.0 + exp(-g)));
}

// --- fused attention glue: qsplit + q/k head_norm + partial rope in one dispatch ---
// One workgroup per (head, column); heads [0, nH) are q heads (split out of q_full, gate
// written to g), heads [nH, nH+nKV) are k heads (in place). Same arithmetic in the same order
// as the five separate kernels: the sum of squares runs serially on thread 0, so the result
// is bit-identical to qsplit -> head_norm -> rope_part. Needs headDim <= 256.
@group(1) @binding(0) var<storage, read> ag_full: array<f32>;
@group(1) @binding(1) var<storage, read_write> ag_q: array<f32>;
@group(1) @binding(2) var<storage, read_write> ag_g: array<f32>;
@group(1) @binding(3) var<storage, read_write> ag_k: array<f32>;
@group(1) @binding(4) var<storage, read> ag_qw: array<f32>;
@group(1) @binding(5) var<storage, read> ag_kw: array<f32>;
@group(1) @binding(6) var<uniform> ag_mc: MC;          // n = k stride, s0 full stride, s1 q stride, s2 g stride
@group(1) @binding(7) var<uniform> ag_dn: DN;
var<workgroup> ag_v: array<f32, 256>;
var<workgroup> ag_inv: f32;
@compute @workgroup_size(64)
fn attn_glue(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let hd = ag_dn.hd;
  let isQ = wg.x < cfg.nH;
  let h = select(wg.x - cfg.nH, wg.x, isQ);
  let col = wg.y;
  let fo = col * ag_mc.s0 + h * 2u * hd;
  let qo = col * ag_mc.s1 + h * hd;
  let go = col * ag_mc.s2 + h * hd;
  let ko = col * ag_mc.n + h * hd;
  for (var i: u32 = lid.x; i < hd; i += 64u) {
    if (isQ) { ag_v[i] = ag_full[fo + i]; ag_g[go + i] = ag_full[fo + hd + i]; }
    else { ag_v[i] = ag_k[ko + i]; }
  }
  workgroupBarrier();
  if (lid.x == 0u) {
    var ss: f32 = 0.0;
    for (var i: u32 = 0u; i < cfg.headDim; i++) { let v = ag_v[i]; ss += v * v; }
    ag_inv = inverseSqrt(ss / f32(cfg.headDim) + cfg.eps);
  }
  workgroupBarrier();
  let inv = ag_inv;
  for (var i: u32 = lid.x; i < hd; i += 64u) {
    let w = select(ag_kw[i], ag_qw[i], isQ);
    ag_v[i] = ag_v[i] * (inv * w);
  }
  workgroupBarrier();
  let half = ag_dn.nRot / 2u;
  for (var i: u32 = lid.x; i < half; i += 64u) {
    let ang = f32(frame.pos + col) * pow(ag_dn.ropeTheta, -f32(2u * i) / f32(ag_dn.nRot));
    let c = cos(ang); let s = sin(ang);
    let a = ag_v[i]; let b = ag_v[i + half];
    ag_v[i] = a * c - b * s;
    ag_v[i + half] = b * c + a * s;
  }
  workgroupBarrier();
  for (var i: u32 = lid.x; i < hd; i += 64u) {
    if (isQ) { ag_q[qo + i] = ag_v[i]; } else { ag_k[ko + i] = ag_v[i]; }
  }
}
`;
