// bello Qwen3.5/3.8 engine — hybrid Gated-DeltaNet + gated-attention WebGPU
// inference, layer-shardable like BelloEngine. Golden reference: ref_q38.mjs
// (validated line-by-line against llama.cpp eval-callback dumps).
import { WGSL, coopWGSL, probeUnpack } from "./engine.js";
import { f16ToF32 } from "./gguf.js";

const WGSL2 = /* wgsl */ `
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
@compute @workgroup_size(128)
fn dn_delta(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x;
  let j = lid.x;
  if (h >= dl_dn.nVH || j >= dl_dn.dState) { return; }
  let kh = h % dl_dn.nKH;
  let kOff = kh * dl_dn.dState;
  let vOff = h * dl_dn.dState;
  let Sb = h * dl_dn.dState * dl_dn.dState;
  let decay = dl_decay[h];
  let scale = inverseSqrt(f32(dl_dn.dState));
  var vhat: f32 = 0.0;
  var sq: f32 = 0.0;
  var kq: f32 = 0.0;
  for (var i: u32 = 0u; i < dl_dn.dState; i++) {
    let idx = Sb + i * dl_dn.dState + j;
    let sdec = dl_s[idx] * decay;
    dl_s[idx] = sdec;
    let ki = dl_k[kOff + i];
    let qi = dl_q[kOff + i];
    vhat += sdec * ki;
    sq += sdec * qi;
    kq += ki * qi;
  }
  let d = (dl_v[vOff + j] - vhat) * dl_beta[h];
  for (var i: u32 = 0u; i < dl_dn.dState; i++) {
    let idx = Sb + i * dl_dn.dState + j;
    dl_s[idx] += dl_k[kOff + i] * d;
  }
  dl_o[vOff + j] = (sq + d * kq) * scale;
}

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
    if (cvSB != 0u && cvSB + col < (frame.snap >> 8u)) {
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
@compute @workgroup_size(128)
fn dn_delta_mc(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x; let j = lid.x; let dS = dlm_dn.dState;
  if (h >= dlm_dn.nVH || j >= dS) { return; }
  let kh = h % dlm_dn.nKH;
  let kOff = kh * dS; let vOff = h * dS; let Sb = h * dS * dS;
  let scale = inverseSqrt(f32(dS));
  let nCols = max(frame.nCols, 1u);
  let sSize = dlm_dn.nVH * dS * dS;
  for (var col: u32 = 0u; col < nCols; col++) {
    let qo = col * dlm_mc.s0 + kOff;
    let ko = col * dlm_mc.s0 + dlm_dn.keyDim + kOff;
    let vo = col * dlm_mc.s0 + 2u * dlm_dn.keyDim + vOff;
    let decay = dlm_decay[col * dlm_mc.s1 + h];
    var vhat: f32 = 0.0;
    var sq: f32 = 0.0;
    var kq: f32 = 0.0;
    for (var i: u32 = 0u; i < dS; i++) {
      let idx = Sb + i * dS + j;
      let sdec = dlm_s[idx] * decay;
      dlm_s[idx] = sdec;
      let ki = dlm_c[ko + i];
      let qi = dlm_c[qo + i];
      vhat += sdec * ki;
      sq += sdec * qi;
      kq += ki * qi;
    }
    let d = (dlm_c[vo + j] - vhat) * dlm_beta[col * dlm_mc.s1 + h];
    for (var i: u32 = 0u; i < dS; i++) {
      let idx = Sb + i * dS + j;
      dlm_s[idx] += dlm_c[ko + i] * d;
    }
    dlm_o[col * dlm_mc.s2 + vOff + j] = (sq + d * kq) * scale;
    let dlSB = frame.snap & 0xffu;     // snapshot slot base + 1 (0 = off)
    if (dlSB != 0u && dlSB + col < (frame.snap >> 8u)) {
      let slot = dlSB - 1u + col;
      for (var i: u32 = 0u; i < dS; i++) { dlm_shadow[slot * sSize + Sb + i * dS + j] = dlm_s[Sb + i * dS + j]; }
    }
  }
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
`;

export class Qwen35Engine {
  static async create(opts) {
    const e = new Qwen35Engine();
    await e._init(opts);
    return e;
  }

  // Fresh context: forget the conversation so far. Recurrent (DeltaNet) states and
  // conv windows are zeroed; the KV caches are simply overwritten from position 0.
  reset() {
    const enc = this.device.createCommandEncoder();
    for (const R of this.layers) {
      if (R.convState) enc.clearBuffer(R.convState);
      if (R.S) enc.clearBuffer(R.S);
    }
    this.device.queue.submit([enc.finish()]);
    this.pos = 0;
  }

  // opts: { device, meta (gguf meta), weights, layerRange, hasEmbed, hasHead, maxSeq }
  async _init({ device, meta, weights, layerRange, hasEmbed = true, hasHead = true, maxSeq = 512, vocab: vocabOpt, matvecVariant = "coop", coopWG = 256, coopRows = 4, batchCols = 4, coopRowsB = coopRows }) {
    this.device = device;
    this.mvVariant = matvecVariant;
    this.coopWG = coopWG; this.coopRows = coopRows;
    this.NC = batchCols; this.coopRowsB = coopRowsB;   // batched (prefill/verify) column count, rows per WG
    const M = meta;
    const dim = M["qwen35.embedding_length"];
    const nH = M["qwen35.attention.head_count"];
    const nKV = M["qwen35.attention.head_count_kv"];
    const hd = M["qwen35.attention.key_length"];
    const nRot = M["qwen35.rope.dimension_count"];
    const ropeTheta = M["qwen35.rope.freq_base"];
    const eps = M["qwen35.attention.layer_norm_rms_epsilon"];
    const inter = M["qwen35.feed_forward_length"];
    const dState = M["qwen35.ssm.state_size"];
    const nKH = M["qwen35.ssm.group_count"];
    const nVH = M["qwen35.ssm.time_step_rank"];
    const dInner = M["qwen35.ssm.inner_size"];
    const keyDim = dState * nKH;
    const convDim = keyDim * 2 + dInner;
    // workers parse the header without the vocab; the embedding's row count says the same thing
    const vocab = M["tokenizer.ggml.tokens"]?.length ?? vocabOpt ?? M["qwen35.vocab_size"] ?? 248320;
    const qDim = nH * hd, kvDim = nKV * hd;
    this.dims = { dim, nH, nKV, hd, nRot, inter, dState, nKH, nVH, dInner, keyDim, convDim, vocab, qDim, kvDim };
    this.maxSeq = maxSeq;
    const [lo, hi] = layerRange;
    this.lo = lo; this.hi = hi;
    this.hasEmbed = hasEmbed; this.hasHead = hasHead;
    this.pos = 0;

    // ---- pipelines with explicit layouts ----
    const mod = device.createShaderModule({ code: WGSL + coopWGSL(coopWG, coopRows, 64, batchCols, coopRowsB, await probeUnpack(device)) + WGSL2 });
    const C = GPUShaderStage.COMPUTE;
    const layout0 = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: C, buffer: { type: "uniform" } },
        { binding: 1, visibility: C, buffer: { type: "uniform" } },
      ],
    });
    const G1 = {
      matvec: ["ro", "ro", "rw", "u"], matvec_q8: ["ro", "ro", "ro", "rw", "u"],
      matvec_q4: ["ro", "ro", "ro", "rw", "u"],
      matvec_coop: ["ro", "ro", "rw", "u"], matvec_q8_coop: ["ro", "ro", "ro", "rw", "u"],
      matvec_q4_coop: ["ro", "ro", "ro", "rw", "u"],
      matvec_coop_acc: ["ro", "ro", "rw", "u"], matvec_q8_coop_acc: ["ro", "ro", "ro", "rw", "u"], matvec_q4_coop_acc: ["ro", "ro", "ro", "rw", "u"],
      matvec_coop_b_acc: ["ro", "ro", "rw", "u"], matvec_q8_coop_b_acc: ["ro", "ro", "ro", "rw", "u"], matvec_q4_coop_b_acc: ["ro", "ro", "ro", "rw", "u"],
      matvec_coop_b: ["ro", "ro", "rw", "u"], matvec_q8_coop_b: ["ro", "ro", "ro", "rw", "u"],
      matvec_q4_coop_b: ["ro", "ro", "ro", "rw", "u"],
      matvec_gu: ["ro", "ro", "ro", "rw", "u"], matvec_gu_b: ["ro", "ro", "ro", "rw", "u"],
      matvec_q8_gu: ["ro", "ro", "ro", "ro", "ro", "rw", "u"], matvec_q8_gu_b: ["ro", "ro", "ro", "ro", "ro", "rw", "u"],
      matvec_q4_gu: ["ro", "ro", "ro", "ro", "ro", "rw", "u"], matvec_q4_gu_b: ["ro", "ro", "ro", "ro", "ro", "rw", "u"],
      rmsnorm: ["ro", "ro", "rw", "u"], head_norm: ["rw", "ro", "u"],
      attn_scores: ["ro", "ro", "rw"], attn_softmax: ["rw"],
      attn_out: ["ro", "ro", "rw"], silu_mul: ["rw", "ro"], add_res: ["rw", "ro"],
      dn_conv: ["ro", "ro", "rw", "rw", "u"],
      dn_gates: ["ro", "ro", "ro", "ro", "rw", "rw", "u"],
      dn_l2: ["rw", "u", "u"],
      dn_delta: ["ro", "ro", "ro", "ro", "ro", "rw", "rw", "u"],
      dn_gatenorm: ["ro", "ro", "ro", "rw", "u"],
      qsplit: ["ro", "rw", "rw", "u"],
      rope_part: ["rw", "u", "u"],
      sigmoid_mul: ["rw", "ro", "u"],
      rmsnorm_mc: ["ro", "ro", "rw", "u"], add_res_mc: ["rw", "ro", "u"],
      dn_gates_mc: ["ro", "ro", "ro", "ro", "rw", "rw", "u"], dn_conv_mc: ["ro", "ro", "rw", "rw", "u", "rw"],
      dn_pre: ["ro", "ro", "ro", "ro", "rw", "rw", "rw", "u"], dn_pre_mc: ["ro", "ro", "ro", "ro", "rw", "rw", "rw", "u", "u"],
      dn_l2_mc: ["rw", "u", "u"], dn_delta_mc: ["ro", "ro", "ro", "rw", "rw", "u", "u", "rw"],
      dn_gatenorm_mc: ["ro", "ro", "ro", "rw", "u", "u"], qsplit_mc: ["ro", "rw", "rw", "u", "u"],
      head_norm_mc: ["rw", "ro", "u"], rope_part_mc: ["rw", "u", "u"], sigmoid_mul_mc: ["rw", "ro", "u"],
      argmax: ["ro", "rw", "u"],
    };
    if (batchCols > 4) Object.assign(G1, {   // 4-column set for verifies with <= 4 live columns
      matvec_coop_b4: G1.matvec_coop_b, matvec_q8_coop_b4: G1.matvec_q8_coop_b, matvec_q4_coop_b4: G1.matvec_q4_coop_b,
      matvec_coop_b4_acc: G1.matvec_coop_b, matvec_q8_coop_b4_acc: G1.matvec_q8_coop_b, matvec_q4_coop_b4_acc: G1.matvec_q4_coop_b,
      matvec_gu_b4: G1.matvec_gu_b, matvec_q8_gu_b4: G1.matvec_q8_gu_b, matvec_q4_gu_b4: G1.matvec_q4_gu_b,
    });
    const bufType = { u: "uniform", ro: "read-only-storage", rw: "storage" };
    this.pipes = {};
    // compile every pipeline in parallel (async): overlaps shader compilation
    // with the rest of setup instead of serializing 20+ compiles
    await Promise.all(Object.entries(G1).map(async ([name, spec]) => {
      const layout1 = device.createBindGroupLayout({
        entries: spec.map((t, i) => ({ binding: i, visibility: C, buffer: { type: bufType[t] } })),
      });
      this.pipes[name] = await device.createComputePipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout0, layout1] }),
        compute: { module: mod, entryPoint: name },
      });
    }));

    // ---- uniforms ----
    const cfgData = new ArrayBuffer(48);
    const cu = new Uint32Array(cfgData), cf = new Float32Array(cfgData);
    cu.set([dim, kvDim, nH, nKV, hd, inter, vocab, maxSeq], 0);
    cf[8] = eps; cf[9] = ropeTheta; cu[10] = qDim;
    this.cfgBuf = this._buf(cfgData, GPUBufferUsage.UNIFORM);
    this.frameBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const dnData = new ArrayBuffer(48);
    const du = new Uint32Array(dnData), df = new Float32Array(dnData);
    du.set([convDim, dState, nKH, nVH, keyDim, nRot, hd, dInner], 0);
    df[8] = ropeTheta; df[9] = eps;
    this.dnBuf = this._buf(dnData, GPUBufferUsage.UNIFORM);
    this._shapes = {};
    this.uNH = this._buf(new Uint32Array([nH]), GPUBufferUsage.UNIFORM);
    this.uNKV = this._buf(new Uint32Array([nKV]), GPUBufferUsage.UNIFORM);
    this.uNKH = this._buf(new Uint32Array([nKH]), GPUBufferUsage.UNIFORM);
    this.uDim = this._buf(new Uint32Array([dim]), GPUBufferUsage.UNIFORM);
    this.uQDim = this._buf(new Uint32Array([qDim]), GPUBufferUsage.UNIFORM);
    this.uHd = this._buf(new Uint32Array([hd]), GPUBufferUsage.UNIFORM);

    // ---- working buffers ----
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.x = device.createBuffer({ size: dim * 4, usage: S });
    this.xn = device.createBuffer({ size: dim * 4, usage: S });
    this.tmpDim = device.createBuffer({ size: dim * 4, usage: S });
    this.g = device.createBuffer({ size: inter * 4, usage: S });
    this.u = device.createBuffer({ size: inter * 4, usage: S });
    // delta-net
    this.qkv = device.createBuffer({ size: convDim * 4, usage: S });
    this.convOut = device.createBuffer({ size: convDim * 4, usage: S });
    this.z = device.createBuffer({ size: dInner * 4, usage: S });
    this.alpha = device.createBuffer({ size: 64 * 4, usage: S });
    this.betaRaw = device.createBuffer({ size: 64 * 4, usage: S });
    this.beta = device.createBuffer({ size: 64 * 4, usage: S });
    this.decay = device.createBuffer({ size: 64 * 4, usage: S });
    this.dOut = device.createBuffer({ size: dInner * 4, usage: S });
    this.gated = device.createBuffer({ size: dInner * 4, usage: S });
    // full attention
    this.qFull = device.createBuffer({ size: nH * hd * 2 * 4, usage: S });
    this.q = device.createBuffer({ size: qDim * 4, usage: S });
    this.gAttn = device.createBuffer({ size: qDim * 4, usage: S });
    this.k = device.createBuffer({ size: kvDim * 4, usage: S });
    this.v = device.createBuffer({ size: kvDim * 4, usage: S });
    this.attnOut = device.createBuffer({ size: qDim * 4, usage: S });
    this.scores = device.createBuffer({ size: nH * maxSeq * 4, usage: S });
    this.stageX = device.createBuffer({ size: dim * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    // ---- weights + per-layer resources ----
    const up = (e2) => {
      if (!e2) return null;
      if (e2.gpu) return e2.gpu;          // already streamed onto the GPU during download
      let r;
      if (e2.kind === "q8" || e2.kind === "q4")
        r = { kind: e2.kind, qs: this._buf(e2.qs, GPUBufferUsage.STORAGE), sc: this._buf(e2.scales, GPUBufferUsage.STORAGE) };
      else r = { kind: "f32", buf: this._buf(e2.data, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) };
      e2.qs = e2.scales = e2.data = null; // release CPU copy once it lives on the GPU
      return r;
    };
    const coop = this.mvVariant === "coop";
    // acc: y[row] += W x (coop only) -> the residual add needs no dispatch
    const mv = (w, x, y, dOut, dIn, acc = false) => {
      const base = w.kind === "q8" ? "matvec_q8" : w.kind === "q4" ? "matvec_q4" : "matvec";
      acc = acc && coop;
      const pipe = coop ? base + "_coop" + (acc ? "_acc" : "") : base;
      const bufs = w.kind === "f32" ? [w.buf, x, y, this._shape(dOut, dIn)] : [w.qs, w.sc, x, y, this._shape(dOut, dIn)];
      return { pipe, acc, wgs: coop ? Math.ceil(dOut / this.coopRows) : Math.ceil(dOut / 64), bg: this._bg(this.pipes[pipe], 1, bufs) };
    };
    this._mv = mv;
    const guOp = (wg2, wu2, x, y, dOut, dIn, xB, yB) => {
      if (!coop || !wg2 || !wu2 || wg2.kind !== wu2.kind) return null;
      const base = wg2.kind === "q8" ? "matvec_q8_gu" : wg2.kind === "q4" ? "matvec_q4_gu" : "matvec_gu";
      const pipe = xB ? base + "_b" : base;
      const shp = xB ? this._shapeB(dOut, dIn, xB.stride / 16, yB.stride / 4) : this._shapeB(dOut, dIn, 0, 0);
      const bufs = wg2.kind === "f32" ? [wg2.buf, wu2.buf, x, y, shp] : [wg2.qs, wg2.sc, wu2.qs, wu2.sc, x, y, shp];
      const op = { pipe, wgs: Math.ceil(dOut / (xB ? this.coopRowsB : this.coopRows)), bg: this._bg(this.pipes[pipe], 1, bufs) };
      if (xB && this.NC > 4) { op.pipe4 = base + "_b4"; op.bg4 = this._bg(this.pipes[op.pipe4], 1, bufs); }
      return op;
    };
    this._guOp = guOp;
    const bgNorm = (x, w, y) => this._bg(this.pipes.rmsnorm, 1, [x, w.buf, y, this.uDim]);

    this.layers = [];
    const buildLayer = (L) => {
      const R = { isFull: L.isFull };
      R.attnNorm = up(L.attnNorm); R.postNorm = up(L.postNorm);
      R.ffnGate = up(L.ffnGate); R.ffnUp = up(L.ffnUp); R.ffnDown = up(L.ffnDown);
      R.bgNorm1 = bgNorm(this.x, R.attnNorm, this.xn);
      R.bgNorm2 = bgNorm(this.x, R.postNorm, this.xn);
      R.mvGate = mv(R.ffnGate, this.xn, this.g, inter, dim);
      R.mvUp = mv(R.ffnUp, this.xn, this.u, inter, dim);
      R.gu = guOp(R.ffnGate, R.ffnUp, this.xn, this.g, inter, dim);
      R.mvDown = coop ? mv(R.ffnDown, this.g, this.x, dim, inter, true) : mv(R.ffnDown, this.g, this.tmpDim, dim, inter);
      if (L.isFull) {
        R.wq = up(L.wq); R.wk = up(L.wk); R.wv = up(L.wv); R.wo = up(L.wo);
        R.qNorm = up(L.qNorm); R.kNorm = up(L.kNorm);
        R.kCache = device.createBuffer({ size: maxSeq * kvDim * 4, usage: S });
        R.vCache = device.createBuffer({ size: maxSeq * kvDim * 4, usage: S });
        R.mvQ = mv(R.wq, this.xn, this.qFull, nH * hd * 2, dim);
        R.mvK = mv(R.wk, this.xn, this.k, kvDim, dim);
        R.mvV = mv(R.wv, this.xn, this.v, kvDim, dim);
        R.mvO = coop ? mv(R.wo, this.attnOut, this.x, dim, qDim, true) : mv(R.wo, this.attnOut, this.tmpDim, dim, qDim);
        R.bgQsplit = this._bg(this.pipes.qsplit, 1, [this.qFull, this.q, this.gAttn, this.dnBuf]);
        R.bgQNorm = this._bg(this.pipes.head_norm, 1, [this.q, R.qNorm.buf, this.uNH]);
        R.bgKNorm = this._bg(this.pipes.head_norm, 1, [this.k, R.kNorm.buf, this.uNKV]);
        R.bgRopeQ = this._bg(this.pipes.rope_part, 1, [this.q, this.uNH, this.dnBuf]);
        R.bgRopeK = this._bg(this.pipes.rope_part, 1, [this.k, this.uNKV, this.dnBuf]);
        R.bgScores = this._bg(this.pipes.attn_scores, 1, [this.q, R.kCache, this.scores]);
        R.bgSoftmax = this._bg(this.pipes.attn_softmax, 1, [this.scores]);
        R.bgAttnOut = this._bg(this.pipes.attn_out, 1, [this.scores, R.vCache, this.attnOut]);
        R.bgSigMul = this._bg(this.pipes.sigmoid_mul, 1, [this.attnOut, this.gAttn, this.uQDim]);
      } else {
        R.wqkv = up(L.wqkv); R.wz = up(L.wz);
        R.wBeta = up(L.wBeta); R.wAlpha = up(L.wAlpha);
        R.wOut = up(L.wOut);
        R.dtBias = this._buf(L.dtBias.data, GPUBufferUsage.STORAGE);
        R.ssmA = this._buf(L.ssmA.data, GPUBufferUsage.STORAGE);
        R.convW = this._buf(L.conv.data, GPUBufferUsage.STORAGE);
        R.ssmNorm = this._buf(L.ssmNorm.data, GPUBufferUsage.STORAGE);
        R.convState = device.createBuffer({ size: convDim * 3 * 4, usage: S });
        R.S = device.createBuffer({ size: nVH * dState * dState * 4, usage: S });
        R.mvQKV = mv(R.wqkv, this.xn, this.qkv, convDim, dim);
        R.mvZ = mv(R.wz, this.xn, this.z, dInner, dim);
        R.mvBeta = mv(R.wBeta, this.xn, this.betaRaw, nVH, dim);
        R.mvAlpha = mv(R.wAlpha, this.xn, this.alpha, nVH, dim);
        R.mvOut = coop ? mv(R.wOut, this.gated, this.x, dim, dInner, true) : mv(R.wOut, this.gated, this.tmpDim, dim, dInner);
        R.bgConv = this._bg(this.pipes.dn_conv, 1, [this.qkv, R.convW, R.convState, this.convOut, this.dnBuf]);
        R.bgGates = this._bg(this.pipes.dn_gates, 1, [this.alpha, this.betaRaw, R.dtBias, R.ssmA, this.beta, this.decay, this.dnBuf]);
        R.bgPre = this._bg(this.pipes.dn_pre, 1, [this.alpha, this.betaRaw, R.dtBias, R.ssmA, this.beta, this.decay, this.convOut, this.dnBuf]);
        R.bgL2Q = this._bg2(this.pipes.dn_l2, [
          { buffer: this.convOut, offset: 0, size: keyDim * 4 },
          { buffer: this.uNKH }, { buffer: this.dnBuf }]);
        R.bgL2K = this._bg2(this.pipes.dn_l2, [
          { buffer: this.convOut, offset: keyDim * 4, size: keyDim * 4 },
          { buffer: this.uNKH }, { buffer: this.dnBuf }]);
        R.bgDelta = this._bg2(this.pipes.dn_delta, [
          { buffer: this.convOut, offset: 0, size: keyDim * 4 },
          { buffer: this.convOut, offset: keyDim * 4, size: keyDim * 4 },
          { buffer: this.convOut, offset: keyDim * 2 * 4, size: dInner * 4 },
          { buffer: this.beta }, { buffer: this.decay },
          { buffer: R.S }, { buffer: this.dOut }, { buffer: this.dnBuf }]);
        R.bgGateNorm = this._bg(this.pipes.dn_gatenorm, 1, [this.dOut, this.z, R.ssmNorm, this.gated, this.dnBuf]);
      }
      return R;
    };
    for (const L of weights.layers) this.layers.push(buildLayer(L));

    if (hasEmbed || hasHead) this.cpuEmbed = weights.embed;
    if (hasHead) {
      this.finalNorm = up(weights.finalNorm);
      if (weights.head) this.headEntry = up(weights.head);
      else { // tied: upload embed for the head but keep CPU copy for row lookups
        const e = weights.embed;
        this.headEntry = { kind: e.kind, qs: this._buf(e.qs, GPUBufferUsage.STORAGE), sc: this._buf(e.scales, GPUBufferUsage.STORAGE) };
      }
      this.logits = device.createBuffer({ size: vocab * 4, usage: S });
      this.stageLogits = device.createBuffer({ size: vocab * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      // on-GPU argmax of the logits (draft chain): 8-byte readback instead of 1 MB
      this.argBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      this.stageArg = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      this.bgArgmax = this._bg(this.pipes.argmax, 1, [this.logits, this.argBuf, this._buf(new Uint32Array([vocab, 0, 0, 0]), GPUBufferUsage.UNIFORM)]);
      this.bgFinalNorm = bgNorm(this.x, this.finalNorm, this.xn);
      this.headOp = mv(this.headEntry, this.xn, this.logits, vocab, dim);
    }

    this.bgCommonFor = {};
    for (const [k2, p] of Object.entries(this.pipes))
      this.bgCommonFor[k2] = this._bg(p, 0, [this.cfgBuf, this.frameBuf]);
    this.bgSilu = this._bg(this.pipes.silu_mul, 1, [this.g, this.u]);
    this.bgAddTmp = this._bg(this.pipes.add_res, 1, [this.x, this.tmpDim]);

    // ---- multi-token prediction (draft) head ----
    if (weights.mtp && hasHead) {
      const W = weights.mtp;
      this.mtpLayer = buildLayer(W.layer);
      this.mtp = {
        ehProj: up(W.ehProj), enorm: up(W.enorm), hnorm: up(W.hnorm), headNorm: up(W.sharedHeadNorm),
        emb: device.createBuffer({ size: dim * 4, usage: S }),
        ehIn: device.createBuffer({ size: 2 * dim * 4, usage: S }),   // [enorm(e) | hnorm(h)]
        stats: { drafts: 0, accepted: 0 },
      };
      const M2 = this.mtp;
      M2.bgENorm = this._bg2res(this.pipes.rmsnorm, [{ buffer: M2.emb }, { buffer: M2.enorm.buf },
        { buffer: M2.ehIn, offset: 0, size: dim * 4 }, { buffer: this.uDim }]);
      M2.bgHNormX = this._bg2res(this.pipes.rmsnorm, [{ buffer: this.x }, { buffer: M2.hnorm.buf },
        { buffer: M2.ehIn, offset: dim * 4, size: dim * 4 }, { buffer: this.uDim }]);
      M2.proj = mv(M2.ehProj, M2.ehIn, this.x, dim, 2 * dim);           // eh_proj -> MTP residual (in x)
      M2.bgHeadNorm = bgNorm(this.x, M2.headNorm, this.xn);               // shared_head_norm -> xn
    }
  }

  _shape(dOut, dIn) {
    const key = dOut + "," + dIn;
    if (!this._shapes[key])
      this._shapes[key] = this._buf(new Uint32Array([dOut, dIn, 0, 0]), GPUBufferUsage.UNIFORM);
    return this._shapes[key];
  }
  _shapeB(dOut, dIn, xs4, ys) {
    const key = "b" + dOut + "," + dIn + "," + xs4 + "," + ys;
    if (!this._shapes[key])
      this._shapes[key] = this._buf(new Uint32Array([dOut, dIn, xs4, ys]), GPUBufferUsage.UNIFORM);
    return this._shapes[key];
  }
  _buf(data, usage) {
    const src = ArrayBuffer.isView(data) ? data : new Uint8Array(data);
    const size = Math.ceil(src.byteLength / 4) * 4;
    const buf = this.device.createBuffer({ size, usage, mappedAtCreation: true });
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(src.buffer, src.byteOffset, src.byteLength));
    buf.unmap();
    return buf;
  }
  _bg(pipe, group, buffers) {
    return this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(group),
      entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
  }
  _bg2(pipe, resources) {
    return this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(1),
      entries: resources.map((r, i) => ({ binding: i, resource: r })),
    });
  }
  _d(pass, name, bg, threads, wg = 64) {
    if (this.skip && this.skip.has(name)) return;   // profiling aid (bench_breakdown)
    pass.setPipeline(this.pipes[name]);
    pass.setBindGroup(0, this.bgCommonFor[name]);
    pass.setBindGroup(1, bg);
    pass.dispatchWorkgroups(Math.ceil(threads / wg));
  }
  _dop(pass, op, nCols = 0) {
    if (this.skip && this.skip.has(op.pipe)) return;
    const four = this.b4 !== false && nCols > 0 && nCols <= 4 && op.pipe4;   // <= 4 live columns: 4-column kernel
    const pipe = four ? op.pipe4 : op.pipe;
    pass.setPipeline(this.pipes[pipe]);
    pass.setBindGroup(0, this.bgCommonFor[pipe]);
    pass.setBindGroup(1, four ? op.bg4 : op.bg);
    if (op.wgs > 32768) pass.dispatchWorkgroups(32768, Math.ceil(op.wgs / 32768)); else pass.dispatchWorkgroups(op.wgs);   // > 65535 per dimension is silently dropped
  }
  _setFrame(pos, seqLen) {
    this.device.queue.writeBuffer(this.frameBuf, 0, new Uint32Array([pos, seqLen]));
  }

  _encodeLayer(enc, i) { this._encodeLayerR(enc, this.layers[i], this.pos); }
  _encodeLayerR(enc, L, pos) {
    const D = this.dims;
    const seqLen = pos + 1;
    if (L.isFull) {
      {
        const p = enc.beginComputePass();
        this._d(p, "rmsnorm", L.bgNorm1, 256, 256);
        this._dop(p, L.mvQ);
        this._dop(p, L.mvK);
        this._dop(p, L.mvV);
        this._d(p, "qsplit", L.bgQsplit, D.nH * D.hd);
        this._d(p, "head_norm", L.bgQNorm, D.nH, 32);
        this._d(p, "head_norm", L.bgKNorm, D.nKV, 32);
        this._d(p, "rope_part", L.bgRopeQ, D.nH * D.nRot / 2);
        this._d(p, "rope_part", L.bgRopeK, D.nKV * D.nRot / 2);
        p.end();
      }
      enc.copyBufferToBuffer(this.k, 0, L.kCache, pos * D.kvDim * 4, D.kvDim * 4);
      enc.copyBufferToBuffer(this.v, 0, L.vCache, pos * D.kvDim * 4, D.kvDim * 4);
      {
        const p = enc.beginComputePass();
        this._d(p, "attn_scores", L.bgScores, D.nH * seqLen);
        this._d(p, "attn_softmax", L.bgSoftmax, D.nH, 1);
        this._d(p, "attn_out", L.bgAttnOut, D.qDim);
        this._d(p, "sigmoid_mul", L.bgSigMul, D.qDim);
        this._dop(p, L.mvO);
        if (!L.mvO.acc) this._d(p, "add_res", this.bgAddTmp, D.dim);
        p.end();
      }
    } else {
      const p = enc.beginComputePass();
      this._d(p, "rmsnorm", L.bgNorm1, 256, 256);
      this._dop(p, L.mvQKV);
      this._dop(p, L.mvZ);
      this._dop(p, L.mvBeta);
      this._dop(p, L.mvAlpha);
      this._d(p, "dn_conv", L.bgConv, D.convDim);
      this._d(p, "dn_pre", L.bgPre, 128, 128);      // gates + L2(q,k) fused
      this._d(p, "dn_delta", L.bgDelta, D.nVH * 128, 128);
      this._d(p, "dn_gatenorm", L.bgGateNorm, D.nVH * 128, 128);
      this._dop(p, L.mvOut);
      if (!L.mvOut.acc) this._d(p, "add_res", this.bgAddTmp, D.dim);
      p.end();
    }
    // ffn
    {
      const p = enc.beginComputePass();
      this._d(p, "rmsnorm", L.bgNorm2, 256, 256);
      if (L.gu) this._dop(p, L.gu);
      else {
        this._dop(p, L.mvGate);
        this._dop(p, L.mvUp);
        this._d(p, "silu_mul", this.bgSilu, D.inter);
      }
      this._dop(p, L.mvDown);
      if (!L.mvDown.acc) this._d(p, "add_res", this.bgAddTmp, D.dim);
      p.end();
    }
  }

  _embedRowF32(id) {
    const { dim } = this.dims;
    const e = this.cpuEmbed;
    if (e.kind === "f32") return e.data.subarray(id * dim, (id + 1) * dim);
    const nb = dim / 32;
    const out = new Float32Array(dim);
    const rowB = id * nb;
    for (let b = 0; b < nb; b++) {
      const si = rowB + b;
      const s = f16ToF32((e.scales[si >> 1] >>> ((si & 1) * 16)) & 0xFFFF);
      if (e.kind === "q4") {
        const qBase = (rowB + b) * 16;
        for (let j = 0; j < 16; j++) {
          const q = e.qs[qBase + j];
          out[b * 32 + j] = s * ((q & 0xF) - 8);
          out[b * 32 + j + 16] = s * ((q >> 4) - 8);
        }
      } else {
        const qBase = (rowB + b) * 32;
        for (let i = 0; i < 32; i++) {
          const q = e.qs[qBase + i];
          out[b * 32 + i] = s * (q > 127 ? q - 256 : q);
        }
      }
    }
    return out;
  }

  async _readback(srcBuf, stageBuf, n) {
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(srcBuf, 0, stageBuf, 0, n * 4);
    this.device.queue.submit([enc.finish()]);
    await stageBuf.mapAsync(GPUMapMode.READ);
    const out = Float32Array.from(new Float32Array(stageBuf.getMappedRange(), 0, n));
    stageBuf.unmap();
    return out;
  }

  // ---- batched prefill (4 prompt tokens per pass) ----
  _bg2g0(pipe, resources) {
    return this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: resources.map((r, i) => ({ binding: i, resource: r })),
    });
  }
  _bg2res(pipe, resources) {
    return this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(1),
      entries: resources.map((r, i) => ({ binding: i, resource: r })),
    });
  }
  _dMC(pass, name, bg, threads, wg, ny, nz = 1) {
    if (this.skip && this.skip.has(name)) return;   // profiling aid (bench_breakdown)
    pass.setPipeline(this.pipes[name]);
    pass.setBindGroup(0, this.bgCommonB[0][name]);
    pass.setBindGroup(1, bg);
    pass.dispatchWorkgroups(Math.ceil(threads / wg), ny, nz);
  }
  _dCol(pass, name, col, bg, threads, wg = 64) {
    if (this.skip && this.skip.has(name)) return;
    pass.setPipeline(this.pipes[name]);
    pass.setBindGroup(0, this.bgCommonB[col][name] || this.bgCommonFor[name]);
    pass.setBindGroup(1, bg);
    pass.dispatchWorkgroups(Math.ceil(threads / wg));
  }

  _initBatch() {
    const D = this.dims;
    const dev = this.device;
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const NC = this.NC, cix = Array.from({ length: NC }, (_, c) => c);
    const al = (n) => Math.ceil(n * 4 / 256) * 256;
    const mkB = (n) => ({ buf: dev.createBuffer({ size: NC * al(n), usage: S }), stride: al(n), n });
    const B = this.B = {
      x: mkB(D.dim), xn: mkB(D.dim), tmpDim: mkB(D.dim), g: mkB(D.inter), u: mkB(D.inter),
      qkv: mkB(D.convDim), convOut: mkB(D.convDim), z: mkB(D.dInner),
      alpha: mkB(D.nVH), betaRaw: mkB(D.nVH), beta: mkB(D.nVH), decay: mkB(D.nVH),
      dOut: mkB(D.dInner), gated: mkB(D.dInner),
      qFull: mkB(D.nH * D.hd * 2), q: mkB(D.qDim), gAttn: mkB(D.qDim),
      k: mkB(D.kvDim), v: mkB(D.kvDim), attnOut: mkB(D.qDim),
    };
    this.stageXB = dev.createBuffer({ size: NC * D.dim * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    // rollback shadows for the recurrent layers (speculative decoding)
    for (const L of this.layers) if (!L.isFull && !L.S_shadow) {
      L.S_shadow = dev.createBuffer({ size: 7 * L.S.size, usage: S });
      L.conv_shadow = dev.createBuffer({ size: 7 * L.convState.size, usage: S });
    }
    this.stageLogitsN = dev.createBuffer({ size: NC * D.vocab * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const slice = (b, c) => ({ buffer: b.buf, offset: c * b.stride, size: b.n * 4 });
    const part = (b, c, off, size) => ({ buffer: b.buf, offset: c * b.stride + off, size });
    this.frameBufsB = cix.map(() => dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    const colPipes = ["rmsnorm", "head_norm", "attn_scores", "attn_softmax", "attn_out",
      "silu_mul", "add_res", "rope_part", "qsplit", "sigmoid_mul",
      "dn_gates", "dn_conv", "dn_l2", "dn_delta", "dn_gatenorm",
      "rmsnorm_mc", "add_res_mc", "dn_gates_mc", "dn_conv_mc", "dn_l2_mc", "dn_pre_mc", "dn_delta_mc", "dn_gatenorm_mc",
      "qsplit_mc", "head_norm_mc", "rope_part_mc", "sigmoid_mul_mc"];
    this.bgCommonB = cix.map((c) => {
      const m = {};
      for (const name of colPipes)
        m[name] = this._bg2g0(this.pipes[name], [{ buffer: this.cfgBuf }, { buffer: this.frameBufsB[c] }]);
      return m;
    });
    const mvB = (w, xB, yB, dOut, dIn, acc = false) => {
      const base = w.kind === "q8" ? "matvec_q8" : w.kind === "q4" ? "matvec_q4" : "matvec";
      const pipe = base + "_coop_b" + (acc ? "_acc" : "");
      const shp = this._shapeB(dOut, dIn, xB.stride / 16, yB.stride / 4);
      const bufs = w.kind === "f32" ? [w.buf, xB.buf, yB.buf, shp] : [w.qs, w.sc, xB.buf, yB.buf, shp];
      const op = { pipe, acc, wgs: Math.ceil(dOut / this.coopRowsB), bg: this._bg(this.pipes[pipe], 1, bufs) };
      if (this.NC > 4) { op.pipe4 = base + "_coop_b4" + (acc ? "_acc" : ""); op.bg4 = this._bg(this.pipes[op.pipe4], 1, bufs); }
      return op;
    };
    if (this.hasHead) {
      B.logits = mkB(D.vocab);
      this.headB = mvB(this.headEntry, B.xn, B.logits, D.vocab, D.dim);
      this.bgFinalNormB = cix.map((c) => this._bg2res(this.pipes.rmsnorm,
        [slice(B.x, c), { buffer: this.finalNorm.buf }, { buffer: this.xn }, { buffer: this.uDim }]));
      if (this.mtp) this.mtp.bgHNormB = cix.map((c) => this._bg2res(this.pipes.rmsnorm,
        [slice(B.x, c), { buffer: this.mtp.hnorm.buf }, { buffer: this.mtp.ehIn, offset: D.dim * 4, size: D.dim * 4 }, { buffer: this.uDim }]));
    }
    this._mcU = this._mcU || {};
    const mcU = (n, s0 = 0, s1 = 0, s2 = 0) => {
      const k = n + "," + s0 + "," + s1 + "," + s2;
      return this._mcU[k] || (this._mcU[k] = { buffer: this._buf(new Uint32Array([n, s0, s1, s2]), GPUBufferUsage.UNIFORM) });
    };
    const st = (b) => b.stride / 4;   // column stride in floats
    const whole = (b) => ({ buffer: b.buf });
    const dn = { buffer: this.dnBuf };
    if (this.hasHead) this.bgFinalNormMC = this._bg2res(this.pipes.rmsnorm_mc,
      [whole(B.x), { buffer: this.finalNorm.buf }, whole(B.xn), mcU(D.dim, st(B.x), st(B.xn))]);
    this.layerB = this.layers.map((L) => {
      const bgNormC = (w, c) => this._bg2res(this.pipes.rmsnorm,
        [slice(B.x, c), { buffer: w.buf }, slice(B.xn, c), { buffer: this.uDim }]);
      const mc = {
        norm1: this._bg2res(this.pipes.rmsnorm_mc, [whole(B.x), { buffer: L.attnNorm.buf }, whole(B.xn), mcU(D.dim, st(B.x), st(B.xn))]),
        norm2: this._bg2res(this.pipes.rmsnorm_mc, [whole(B.x), { buffer: L.postNorm.buf }, whole(B.xn), mcU(D.dim, st(B.x), st(B.xn))]),
        addTmp: this._bg2res(this.pipes.add_res_mc, [whole(B.x), whole(B.tmpDim), mcU(D.dim, st(B.x), st(B.tmpDim))]),
      };
      if (L.isFull) Object.assign(mc, {
        qsplit: this._bg2res(this.pipes.qsplit_mc, [whole(B.qFull), whole(B.q), whole(B.gAttn), mcU(0, st(B.qFull), st(B.q), st(B.gAttn)), dn]),
        qNorm: this._bg2res(this.pipes.head_norm_mc, [whole(B.q), { buffer: L.qNorm.buf }, mcU(D.nH, st(B.q))]),
        kNorm: this._bg2res(this.pipes.head_norm_mc, [whole(B.k), { buffer: L.kNorm.buf }, mcU(D.nKV, st(B.k))]),
        ropeQ: this._bg2res(this.pipes.rope_part_mc, [whole(B.q), mcU(D.nH, st(B.q)), dn]),
        ropeK: this._bg2res(this.pipes.rope_part_mc, [whole(B.k), mcU(D.nKV, st(B.k)), dn]),
        sigMul: this._bg2res(this.pipes.sigmoid_mul_mc, [whole(B.attnOut), whole(B.gAttn), mcU(D.qDim, st(B.attnOut), st(B.gAttn))]),
      });
      else Object.assign(mc, {
        gates: this._bg2res(this.pipes.dn_gates_mc, [whole(B.alpha), whole(B.betaRaw), { buffer: L.dtBias }, { buffer: L.ssmA },
          whole(B.beta), whole(B.decay), mcU(D.nVH, st(B.alpha))]),
        conv: this._bg2res(this.pipes.dn_conv_mc, [whole(B.qkv), { buffer: L.convW }, { buffer: L.convState }, whole(B.convOut),
          mcU(D.convDim, st(B.qkv), st(B.convOut)), { buffer: L.conv_shadow }]),
        l2: this._bg2res(this.pipes.dn_l2_mc, [whole(B.convOut), mcU(D.nKH, st(B.convOut), D.keyDim), dn]),
        pre: this._bg2res(this.pipes.dn_pre_mc, [whole(B.alpha), whole(B.betaRaw), { buffer: L.dtBias }, { buffer: L.ssmA },
          whole(B.beta), whole(B.decay), whole(B.convOut), mcU(0, st(B.alpha), st(B.convOut)), dn]),
        delta: this._bg2res(this.pipes.dn_delta_mc, [whole(B.convOut), whole(B.beta), whole(B.decay), { buffer: L.S }, whole(B.dOut),
          mcU(0, st(B.convOut), st(B.beta), st(B.dOut)), dn, { buffer: L.S_shadow }]),
        gatenorm: this._bg2res(this.pipes.dn_gatenorm_mc, [whole(B.dOut), whole(B.z), { buffer: L.ssmNorm }, whole(B.gated),
          mcU(0, st(B.dOut), st(B.z), st(B.gated)), dn]),
      });
      const R = {
        mc,
        gateUp: [mvB(L.ffnGate, B.xn, B.g, D.inter, D.dim), mvB(L.ffnUp, B.xn, B.u, D.inter, D.dim)],
        gu: this._guOp(L.ffnGate, L.ffnUp, B.xn.buf, B.g.buf, D.inter, D.dim, B.xn, B.g),
        down: mvB(L.ffnDown, B.g, B.x, D.dim, D.inter, true),
        cols: cix.map((c) => ({
          norm1: bgNormC(L.attnNorm, c),
          norm2: bgNormC(L.postNorm, c),
          addTmp: this._bg2res(this.pipes.add_res, [slice(B.x, c), slice(B.tmpDim, c)]),
          silu: this._bg2res(this.pipes.silu_mul, [slice(B.g, c), slice(B.u, c)]),
        })),
      };
      if (L.isFull) {
        R.qkvOps = [mvB(L.wq, B.xn, B.qFull, D.nH * D.hd * 2, D.dim),
          mvB(L.wk, B.xn, B.k, D.kvDim, D.dim), mvB(L.wv, B.xn, B.v, D.kvDim, D.dim)];
        R.o = mvB(L.wo, B.attnOut, B.x, D.dim, D.qDim, true);
        for (let c = 0; c < NC; c++) Object.assign(R.cols[c], {
          qsplit: this._bg2res(this.pipes.qsplit, [slice(B.qFull, c), slice(B.q, c), slice(B.gAttn, c), { buffer: this.dnBuf }]),
          qNorm: this._bg2res(this.pipes.head_norm, [slice(B.q, c), { buffer: L.qNorm.buf }, { buffer: this.uNH }]),
          kNorm: this._bg2res(this.pipes.head_norm, [slice(B.k, c), { buffer: L.kNorm.buf }, { buffer: this.uNKV }]),
          ropeQ: this._bg2res(this.pipes.rope_part, [slice(B.q, c), { buffer: this.uNH }, { buffer: this.dnBuf }]),
          ropeK: this._bg2res(this.pipes.rope_part, [slice(B.k, c), { buffer: this.uNKV }, { buffer: this.dnBuf }]),
          scores: this._bg2res(this.pipes.attn_scores, [slice(B.q, c), { buffer: L.kCache }, { buffer: this.scores }]),
          softmax: this._bg2res(this.pipes.attn_softmax, [{ buffer: this.scores }]),
          attnOut: this._bg2res(this.pipes.attn_out, [{ buffer: this.scores }, { buffer: L.vCache }, slice(B.attnOut, c)]),
          sigMul: this._bg2res(this.pipes.sigmoid_mul, [slice(B.attnOut, c), slice(B.gAttn, c), { buffer: this.uQDim }]),
        });
      } else {
        R.dnOps = [mvB(L.wqkv, B.xn, B.qkv, D.convDim, D.dim), mvB(L.wz, B.xn, B.z, D.dInner, D.dim),
          mvB(L.wBeta, B.xn, B.betaRaw, D.nVH, D.dim), mvB(L.wAlpha, B.xn, B.alpha, D.nVH, D.dim)];
        R.out = mvB(L.wOut, B.gated, B.x, D.dim, D.dInner, true);
        for (let c = 0; c < NC; c++) Object.assign(R.cols[c], {
          gates: this._bg2res(this.pipes.dn_gates, [slice(B.alpha, c), slice(B.betaRaw, c),
            { buffer: L.dtBias }, { buffer: L.ssmA }, slice(B.beta, c), slice(B.decay, c), { buffer: this.dnBuf }]),
          conv: this._bg2res(this.pipes.dn_conv, [slice(B.qkv, c), { buffer: L.convW },
            { buffer: L.convState }, slice(B.convOut, c), { buffer: this.dnBuf }]),
          l2q: this._bg2res(this.pipes.dn_l2, [part(B.convOut, c, 0, D.keyDim * 4),
            { buffer: this.uNKH }, { buffer: this.dnBuf }]),
          l2k: this._bg2res(this.pipes.dn_l2, [part(B.convOut, c, D.keyDim * 4, D.keyDim * 4),
            { buffer: this.uNKH }, { buffer: this.dnBuf }]),
          delta: this._bg2res(this.pipes.dn_delta, [part(B.convOut, c, 0, D.keyDim * 4),
            part(B.convOut, c, D.keyDim * 4, D.keyDim * 4),
            part(B.convOut, c, D.keyDim * 2 * 4, D.dInner * 4),
            slice(B.beta, c), slice(B.decay, c), { buffer: L.S }, slice(B.dOut, c), { buffer: this.dnBuf }]),
          gatenorm: this._bg2res(this.pipes.dn_gatenorm, [slice(B.dOut, c), slice(B.z, c),
            { buffer: L.ssmNorm }, slice(B.gated, c), { buffer: this.dnBuf }]),
        });
      }
      return R;
    });
  }

  // nCols < 4: batched matvecs still compute 4 columns (extra columns are
  // garbage into scratch); every per-column op runs as ONE multi-column
  // dispatch over the live columns. snapshotDN (set through frame.snap) makes
  // the recurrent kernels save their state after each non-final column so a
  // rejected speculative suffix can be rolled back.
  _encodeLayerBatch(enc, i, basePos, nCols = this.NC, snapshotDN = false) {
    const D = this.dims, L = this.layers[i], LB = this.layerB[i], B = this.B, M = LB.mc;
    if (L.isFull) {
      {
        const p = enc.beginComputePass();
        this._dMC(p, "rmsnorm_mc", M.norm1, 256, 256, nCols);
        for (const op of LB.qkvOps) this._dop(p, op, nCols);
        this._dMC(p, "qsplit_mc", M.qsplit, D.nH * D.hd, 64, nCols);
        this._dMC(p, "head_norm_mc", M.qNorm, D.nH, 32, nCols);
        this._dMC(p, "head_norm_mc", M.kNorm, D.nKV, 32, nCols);
        this._dMC(p, "rope_part_mc", M.ropeQ, D.nH * D.nRot / 2, 64, nCols);
        this._dMC(p, "rope_part_mc", M.ropeK, D.nKV * D.nRot / 2, 64, nCols);
        p.end();
      }
      for (let c = 0; c < nCols; c++) {
        enc.copyBufferToBuffer(B.k.buf, c * B.k.stride, L.kCache, (basePos + c) * D.kvDim * 4, D.kvDim * 4);
        enc.copyBufferToBuffer(B.v.buf, c * B.v.stride, L.vCache, (basePos + c) * D.kvDim * 4, D.kvDim * 4);
      }
      {
        const p = enc.beginComputePass();
        for (let c = 0; c < nCols; c++) {   // shared score scratch: columns in turn
          const C = LB.cols[c];
          this._dCol(p, "attn_scores", c, C.scores, D.nH * (basePos + c + 1));
          this._dCol(p, "attn_softmax", c, C.softmax, D.nH, 1);
          this._dCol(p, "attn_out", c, C.attnOut, D.qDim);
        }
        this._dMC(p, "sigmoid_mul_mc", M.sigMul, D.qDim, 64, nCols);
        this._dop(p, LB.o, nCols);
        if (!LB.o.acc) this._dMC(p, "add_res_mc", M.addTmp, D.dim, 64, nCols);
        p.end();
      }
    } else {
      const p = enc.beginComputePass();
      this._dMC(p, "rmsnorm_mc", M.norm1, 256, 256, nCols);
      for (const op of LB.dnOps) this._dop(p, op, nCols);
      this._dMC(p, "dn_conv_mc", M.conv, D.convDim, 64, 1);           // loops over columns
      this._dMC(p, "dn_pre_mc", M.pre, 128, 128, nCols);              // gates + L2(q,k) fused, one WG per column
      this._dMC(p, "dn_delta_mc", M.delta, D.nVH * 128, 128, 1);     // loops over columns
      this._dMC(p, "dn_gatenorm_mc", M.gatenorm, D.nVH * 128, 128, nCols);
      this._dop(p, LB.out, nCols);
      if (!LB.out.acc) this._dMC(p, "add_res_mc", M.addTmp, D.dim, 64, nCols);
      p.end();
    }
    {
      const p = enc.beginComputePass();
      this._dMC(p, "rmsnorm_mc", M.norm2, 256, 256, nCols);
      if (LB.gu) this._dop(p, LB.gu, nCols);
      else {
        for (const op of LB.gateUp) this._dop(p, op, nCols);
        for (let c = 0; c < nCols; c++) this._dCol(p, "silu_mul", c, LB.cols[c].silu, D.inter);
      }
      this._dop(p, LB.down, nCols);
      if (!LB.down.acc) this._dMC(p, "add_res_mc", M.addTmp, D.dim, 64, nCols);
      p.end();
    }
  }

  async _runBatchAndRead(basePos, n = this.NC) {
    const { dim } = this.dims;
    const enc = this.device.createCommandEncoder();
    for (let l = 0; l < this.layers.length; l++) this._encodeLayerBatch(enc, l, basePos, n);
    for (let c = 0; c < n; c++) enc.copyBufferToBuffer(this.B.x.buf, c * this.B.x.stride, this.stageXB, c * dim * 4, dim * 4);
    this.device.queue.submit([enc.finish()]);
    await this.stageXB.mapAsync(GPUMapMode.READ, 0, n * dim * 4);
    const out = Float32Array.from(new Float32Array(this.stageXB.getMappedRange(0, n * dim * 4), 0, n * dim));
    this.stageXB.unmap();
    this.pos = basePos + n;
    return out;
  }
  // ids.length columns (1..4). snapshot: save recurrent state after every
  // non-final column so restoreDN(k) can undo a rejected speculative suffix.
  async embedRunBatch(ids, basePos, snapshot = false) {
    if (!this.B) this._initBatch();
    const n = ids.length;
    this.pos = basePos;
    const sp = !snapshot ? 0 : typeof snapshot === "object"
      ? ((snapshot.total << 8) | (snapshot.base + 1)) : ((n << 8) | 1);
    for (let c = 0; c < n; c++) {
      this.device.queue.writeBuffer(this.frameBufsB[c], 0, new Uint32Array([basePos + c, basePos + c + 1, n, sp]));
      this.device.queue.writeBuffer(this.B.x.buf, c * this.B.x.stride, this._embedRowF32(ids[c]));
    }
    return this._runBatchAndRead(basePos, n);
  }
  async runHiddenBatch(xs, basePos, snapshot = false) {
    if (!this.B) this._initBatch();
    const { dim } = this.dims;
    const n = xs.length / dim;
    this.pos = basePos;
    const sp = !snapshot ? 0 : typeof snapshot === "object"
      ? ((snapshot.total << 8) | (snapshot.base + 1)) : ((n << 8) | 1);
    for (let c = 0; c < n; c++) {
      this.device.queue.writeBuffer(this.frameBufsB[c], 0, new Uint32Array([basePos + c, basePos + c + 1, n, sp]));
      this.device.queue.writeBuffer(this.B.x.buf, c * this.B.x.stride, xs.subarray(c * dim, (c + 1) * dim));
    }
    return this._runBatchAndRead(basePos, n);
  }
  restoreDN(k) { this._restoreDN(k); }
  setHidden(h) { this.device.queue.writeBuffer(this.x, 0, h); }   // final trunk hidden (chain host) for the draft head

  // final norm + LM head for n hidden states (n*dim floats, or null to use the
  // columns already in B.x) -> array of n logits vectors. Leaves the hiddens
  // in B.x so the draft head can read them by column.
  async headBatch(hs, n = hs ? hs.length / this.dims.dim : this.NC) {
    if (!this.B) this._initBatch();
    const { dim, vocab } = this.dims;
    if (hs) for (let c = 0; c < n; c++) this.device.queue.writeBuffer(this.B.x.buf, c * this.B.x.stride, hs.subarray(c * dim, (c + 1) * dim));
    this.device.queue.writeBuffer(this.frameBufsB[0], 0, new Uint32Array([this.pos, this.pos + 1, n, 0]));
    const enc = this.device.createCommandEncoder();
    const p = enc.beginComputePass();
    this._dMC(p, "rmsnorm_mc", this.bgFinalNormMC, 256, 256, n);
    this._dop(p, this.headB, n);
    p.end();
    for (let c = 0; c < n; c++) enc.copyBufferToBuffer(this.B.logits.buf, c * this.B.logits.stride, this.stageLogitsN, c * vocab * 4, vocab * 4);
    this.device.queue.submit([enc.finish()]);
    await this.stageLogitsN.mapAsync(GPUMapMode.READ, 0, n * vocab * 4);
    const all = new Float32Array(this.stageLogitsN.getMappedRange(0, n * vocab * 4)).slice();
    this.stageLogitsN.unmap();
    const out = [];
    for (let c = 0; c < n; c++) out.push(all.subarray(c * vocab, (c + 1) * vocab));
    return out;
  }

  // ---- speculative decoding with the MTP head ----
  // Run the draft block for the token `tNext` (at position `pos`) given the
  // trunk hidden of the previous position: srcCol === null reads this.x,
  // otherwise batch column srcCol. Appends to the MTP layer's own KV cache.
  // wantLogits -> returns draft logits (argmax = drafted token).
  async mtpRun(srcCol, tNext, pos, wantLogits) {
    const M2 = this.mtp, { dim, vocab } = this.dims;
    this.device.queue.writeBuffer(M2.emb, 0, this._embedRowF32(tNext));
    this._setFrame(pos, pos + 1);
    const enc = this.device.createCommandEncoder();
    {
      const p = enc.beginComputePass();
      this._d(p, "rmsnorm", M2.bgENorm, 256, 256);
      this._d(p, "rmsnorm", srcCol === null ? M2.bgHNormX : M2.bgHNormB[srcCol], 256, 256);
      this._dop(p, M2.proj);
      p.end();
    }
    this._encodeLayerR(enc, this.mtpLayer, pos);
    if (wantLogits) {
      const p = enc.beginComputePass();
      this._d(p, "rmsnorm", M2.bgHeadNorm, 256, 256);
      this._dop(p, this.headOp);
      if (wantLogits === "argmax") this._d(p, "argmax", this.bgArgmax, 256, 256);
      p.end();
    }
    if (wantLogits === "argmax") enc.copyBufferToBuffer(this.argBuf, 0, this.stageArg, 0, 16);
    this.device.queue.submit([enc.finish()]);
    if (!wantLogits) return null;
    if (wantLogits === "argmax") {
      await this.stageArg.mapAsync(GPUMapMode.READ);
      const id = new Uint32Array(this.stageArg.getMappedRange())[0];
      this.stageArg.unmap();
      return id;
    }
    return await this._readback(this.logits, this.stageLogits, vocab);
  }

  // verify tokens[k] at positions pos+k (2..4 tokens): trunk (local batched
  // pass with DeltaNet snapshots, or a caller-supplied runTrunk for a device
  // chain) then one batched head pass -> logits per column.
  async verifyN(tokens, pos, runTrunk = null) {
    const n = tokens.length, { dim } = this.dims;
    let hs;
    if (runTrunk) hs = await runTrunk(tokens, pos);
    else if (n <= this.NC) hs = await this.embedRunBatch(tokens, pos, true);
    else {
      hs = new Float32Array(n * dim);
      for (let c0 = 0; c0 < n; c0 += this.NC) {
        const m = Math.min(this.NC, n - c0);
        hs.set(await this.embedRunBatch(tokens.slice(c0, c0 + m), pos + c0, { base: c0, total: n }), c0 * dim);
      }
    }
    const lgs = [];
    for (let c0 = 0; c0 < n; c0 += this.NC)
      lgs.push(...await this.headBatch(hs.subarray(c0 * dim, Math.min(n, c0 + this.NC) * dim), Math.min(this.NC, n - c0)));
    return { lgs, hs };
  }
  _restoreDN(k) {   // recurrent state as it was after verify column k
    const enc = this.device.createCommandEncoder();
    for (const L of this.layers) if (!L.isFull) {
      enc.copyBufferToBuffer(L.S_shadow, k * L.S.size, L.S, 0, L.S.size);
      enc.copyBufferToBuffer(L.conv_shadow, k * L.convState.size, L.convState, 0, L.convState.size);
    }
    this.device.queue.submit([enc.finish()]);
  }
  _adoptHidden(col) {   // batch column -> this.x (the hidden the next draft reads)
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.B.x.buf, col * this.B.x.stride, this.x, 0, this.dims.dim * 4);
    this.device.queue.submit([enc.finish()]);
  }

  // One speculative step with K chained drafts (K <= 3). Precondition: this.x
  // holds the trunk hidden of the previous position and `tNext` is the
  // already-sampled token for this.pos. Returns 1..K+1 new tokens.
  // runTrunk(tokens, pos) -> hidden states for all columns (chain mode);
  // onReject(k) tells the other devices to roll their recurrent state back
  // to what it was after column k.
  async specStep(tNext, sample, K = 3, { runTrunk = null, onReject = null } = {}) {
    const pos = this.pos, M2 = this.mtp, { dim } = this.dims;
    K = Math.max(1, Math.min(7, K));
    const drafts = [];
    for (let k = 0; k < K; k++) {
      // after the first call this.x holds the MTP block's own output hidden,
      // which is what chained drafting feeds back in
      drafts.push(await this.mtpRun(null, k === 0 ? tNext : drafts[k - 1], pos + k, "argmax"));
    }
    const { lgs, hs } = await this.verifyN([tNext, ...drafts], pos, runTrunk);
    const out = [];
    let a = 0;   // accepted drafts
    for (let k = 0; k <= K; k++) {
      const t = sample(lgs[k]);
      out.push(t);
      if (k < K && t === drafts[k]) a++; else break;
    }
    M2.stats.drafts += K; M2.stats.accepted += a;
    if (a < K) { this._restoreDN(a); if (onReject) await onReject(a); }
    // re-fill the draft cache for the accepted positions with exact trunk hiddens
    for (let j = 1; j <= a; j++) {
      this.setHidden(hs.subarray((j - 1) * dim, j * dim));
      await this.mtpRun(null, out[j - 1], pos + j, false);
    }
    this.setHidden(hs.subarray(a * dim, (a + 1) * dim));
    this.pos = pos + a + 1;
    return out;
  }

  async prefillTokens(ids) {
    if (!this.B) this._initBatch();
    let i = 0, sinceSync = 0;
    const NC = this.NC;
    while (ids.length - i >= NC) {
      const basePos = this.pos;
      for (let c = 0; c < NC; c++) {
        this.device.queue.writeBuffer(this.frameBufsB[c], 0, new Uint32Array([basePos + c, basePos + c + 1, NC, 0]));
        this.device.queue.writeBuffer(this.B.x.buf, c * this.B.x.stride, this._embedRowF32(ids[i + c]));
      }
      const enc = this.device.createCommandEncoder();
      for (let l = 0; l < this.layers.length; l++) this._encodeLayerBatch(enc, l, basePos, NC);
      enc.copyBufferToBuffer(this.B.x.buf, (NC - 1) * this.B.x.stride, this.x, 0, this.dims.dim * 4);
      this.device.queue.submit([enc.finish()]);
      if (this.mtp && this.mtpFill !== false)
        for (let c = 0; c < NC; c++) if (i + c + 1 < ids.length) await this.mtpRun(c, ids[i + c + 1], basePos + c + 1, false);
      this.pos += NC;
      i += NC;
      if (++sinceSync >= 4) { await this.device.queue.onSubmittedWorkDone(); sinceSync = 0; }
    }
    for (; i < ids.length; i++) {
      await this.prefillToken(ids[i]);
      if (this.mtp && this.mtpFill !== false && i + 1 < ids.length) await this.mtpRun(null, ids[i + 1], i + 1, false);
      if (i % 8 === 7) await this.device.queue.onSubmittedWorkDone();
    }
    await this.device.queue.onSubmittedWorkDone();
  }

  // prefill fast path: layers only, no head, no readback
  async prefillToken(tokenId) {
    this._setFrame(this.pos, this.pos + 1);
    this.device.queue.writeBuffer(this.x, 0, this._embedRowF32(tokenId));
    const enc = this.device.createCommandEncoder();
    for (let i = 0; i < this.layers.length; i++) this._encodeLayer(enc, i);
    this.device.queue.submit([enc.finish()]);
    this.pos++;
    // fire-and-forget; callers batch backpressure via onSubmittedWorkDone()
  }

  async embedRun(tokenId, pos) {
    const { dim } = this.dims;
    this.pos = pos;
    this._setFrame(pos, pos + 1);
    this.device.queue.writeBuffer(this.x, 0, this._embedRowF32(tokenId));
    const enc = this.device.createCommandEncoder();
    for (let i = 0; i < this.layers.length; i++) this._encodeLayer(enc, i);
    this.device.queue.submit([enc.finish()]);
    return await this._readback(this.x, this.stageX, dim);
  }

  async runHidden(xIn, pos) {
    const { dim } = this.dims;
    this.pos = pos;
    this._setFrame(pos, pos + 1);
    this.device.queue.writeBuffer(this.x, 0, xIn);
    const enc = this.device.createCommandEncoder();
    for (let i = 0; i < this.layers.length; i++) this._encodeLayer(enc, i);
    this.device.queue.submit([enc.finish()]);
    return await this._readback(this.x, this.stageX, dim);
  }

  async headFromHidden(xIn) {
    const { vocab } = this.dims;
    this.device.queue.writeBuffer(this.x, 0, xIn);
    const enc = this.device.createCommandEncoder();
    {
      const p = enc.beginComputePass();
      this._d(p, "rmsnorm", this.bgFinalNorm, 256, 256);
      this._dop(p, this.headOp);
      p.end();
    }
    this.device.queue.submit([enc.finish()]);
    return await this._readback(this.logits, this.stageLogits, vocab);
  }

  // Whole token in one encoder + one submit; no hidden-state readback between
  // the last layer and the head (that round trip cost a full pipeline drain).
  async forwardToken(tokenId) {
    const { vocab } = this.dims;
    this._setFrame(this.pos, this.pos + 1);
    this.device.queue.writeBuffer(this.x, 0, this._embedRowF32(tokenId));
    const enc = this.device.createCommandEncoder();
    for (let i = 0; i < this.layers.length; i++) this._encodeLayer(enc, i);
    {
      const p = enc.beginComputePass();
      this._d(p, "rmsnorm", this.bgFinalNorm, 256, 256);
      this._dop(p, this.headOp);
      p.end();
    }
    this.device.queue.submit([enc.finish()]);
    const logits = await this._readback(this.logits, this.stageLogits, vocab);
    this.pos++;
    return logits;
  }
}
