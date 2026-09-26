// Mixture-of-experts FFN kernels (Qwen3.5 / 3.6 MoE: top-k routed experts + one shared expert).
//
// Experts are stored as the GGUF stacks them, [nExp][dOut][dIn], and uploaded exactly like any
// Q4_0 / Q8_0 matrix with nExp * dOut rows, so expert e's row r is row e * dOut + r: the kernels
// only add an offset, read from the routing result on the GPU (no readback to pick experts).
//
// One launch covers every (column, slot) pair: workgroup y = col * K + slot. Each pair's math is
// the same whether a pass has 1 column (decode) or many (verify / prefill), so the batched path
// gives the same bits as the one-token path, which keeps speculative decoding exact.
//
//   moe_router      logits [col][nExp] -> sel [col][K] (expert ids), selw [col][K] (weights)
//   moe_gu_{q4,q8}  h[col,slot] = silu(Wgate[e] x[col]) * (Wup[e] x[col])
//   moe_dn_{q4,q8}  y[col,slot] = Wdown[e] h[col,slot]
//   moe_combine     x[col] += sum_k selw[col,k] * y[col,k] + sigmoid(sg[col]) * shared[col]

const ROWS = 4, WG = 64;

// dot of one 32-weight block (row word base wb, block scale sc) with x[xb .. xb+32)
function blockDot(fmt, arr, wb, sc, xarr, xb) {
  if (fmt === "q4") return `{
      var s: f32 = 0.0;
      for (var qt: u32 = 0u; qt < 4u; qt++) {
        let w = ${arr}[${wb} + qt];
        for (var i: u32 = 0u; i < 4u; i++) {
          let lo = f32((w >> (8u * i)) & 0xFu) - 8.0;
          let hi = f32((w >> (8u * i + 4u)) & 0xFu) - 8.0;
          s += lo * ${xarr}[${xb} + 4u * qt + i] + hi * ${xarr}[${xb} + 16u + 4u * qt + i];
        }
      }
      acc += ${sc} * s;
    }`;
  return `{
      var s: f32 = 0.0;
      for (var qt: u32 = 0u; qt < 8u; qt++) {
        let w = bitcast<i32>(${arr}[${wb} + qt]);
        for (var i: u32 = 0u; i < 4u; i++) {
          s += f32((w << (24u - 8u * i)) >> 24u) * ${xarr}[${xb} + 4u * qt + i];
        }
      }
      acc += ${sc} * s;
    }`;
}
const wordsPerBlock = (fmt) => (fmt === "q4" ? 4 : 8);

function guKernel(fmt) {
  const P = `mg${fmt}`, W = wordsPerBlock(fmt);
  return `
@group(1) @binding(0) var<storage, read> ${P}_gq: array<u32>;
@group(1) @binding(1) var<storage, read> ${P}_gs: array<u32>;
@group(1) @binding(2) var<storage, read> ${P}_uq: array<u32>;
@group(1) @binding(3) var<storage, read> ${P}_us: array<u32>;
@group(1) @binding(4) var<storage, read> ${P}_x: array<f32>;
@group(1) @binding(5) var<storage, read_write> ${P}_h: array<f32>;
@group(1) @binding(6) var<storage, read> ${P}_sel: array<u32>;
@group(1) @binding(7) var<uniform> ${P}_s: MOE;
fn ${P}_sc(a: u32, i: u32, gate: bool) -> f32 {
  if (gate) { return unpack2x16float(${P}_gs[i >> 1u])[i & 1u]; }
  return unpack2x16float(${P}_us[i >> 1u])[i & 1u];
}
@compute @workgroup_size(${WG})
fn moe_gu_${fmt}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let S = ${P}_s; let t = lid.x; let cs = wg.y;
  let col = cs / S.K;
  let e = ${P}_sel[cs];
  let nb = S.dIn / 32u;
  let row0 = wg.x * ${ROWS}u;
  let xb0 = col * S.xs;
  for (var r: u32 = 0u; r < ${ROWS}u; r++) {
    let row = row0 + r;
    var g: f32 = 0.0; var u: f32 = 0.0;
    if (row < S.dOut) {
      let er = e * S.dOut + row;
      for (var b: u32 = t; b < nb; b += ${WG}u) {
        var acc: f32 = 0.0;
        ${blockDot(fmt, `${P}_gq`, `(er * nb + b) * ${W}u`, `${P}_sc(0u, er * nb + b, true)`, `${P}_x`, "xb0 + b * 32u")}
        g += acc;
        acc = 0.0;
        ${blockDot(fmt, `${P}_uq`, `(er * nb + b) * ${W}u`, `${P}_sc(0u, er * nb + b, false)`, `${P}_x`, "xb0 + b * 32u")}
        u += acc;
      }
    }
    moe_red[t] = g; moe_red[${WG}u + t] = u;
    workgroupBarrier();
    for (var st: u32 = ${WG / 2}u; st > 0u; st >>= 1u) {
      if (t < st) { moe_red[t] += moe_red[t + st]; moe_red[${WG}u + t] += moe_red[${WG}u + t + st]; }
      workgroupBarrier();
    }
    if (t == 0u && row < S.dOut) {
      let gg = moe_red[0]; let uu = moe_red[${WG}u];
      ${P}_h[cs * S.ys + row] = gg / (1.0 + exp(-gg)) * uu;
    }
    workgroupBarrier();
  }
}`;
}

function dnKernel(fmt) {
  const P = `md${fmt}`, W = wordsPerBlock(fmt);
  return `
@group(1) @binding(0) var<storage, read> ${P}_q: array<u32>;
@group(1) @binding(1) var<storage, read> ${P}_sc: array<u32>;
@group(1) @binding(2) var<storage, read> ${P}_x: array<f32>;
@group(1) @binding(3) var<storage, read_write> ${P}_y: array<f32>;
@group(1) @binding(4) var<storage, read> ${P}_sel: array<u32>;
@group(1) @binding(5) var<uniform> ${P}_s: MOE;
@compute @workgroup_size(${WG})
fn moe_dn_${fmt}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let S = ${P}_s; let t = lid.x; let cs = wg.y;
  let e = ${P}_sel[cs];
  let nb = S.dIn / 32u;
  let row0 = wg.x * ${ROWS}u;
  let xb0 = cs * S.xs;   // each (column, slot) has its own input h
  for (var r: u32 = 0u; r < ${ROWS}u; r++) {
    let row = row0 + r;
    var y: f32 = 0.0;
    if (row < S.dOut) {
      let er = e * S.dOut + row;
      for (var b: u32 = t; b < nb; b += ${WG}u) {
        var acc: f32 = 0.0;
        let i = er * nb + b;
        ${blockDot(fmt, `${P}_q`, `i * ${W}u`, `unpack2x16float(${P}_sc[i >> 1u])[i & 1u]`, `${P}_x`, "xb0 + b * 32u")}
        y += acc;
      }
    }
    moe_red[t] = y;
    workgroupBarrier();
    for (var st: u32 = ${WG / 2}u; st > 0u; st >>= 1u) {
      if (t < st) { moe_red[t] += moe_red[t + st]; }
      workgroupBarrier();
    }
    if (t == 0u && row < S.dOut) { ${P}_y[cs * S.ys + row] = moe_red[0]; }
    workgroupBarrier();
  }
}`;
}

export function moeWGSL() {
  return /* wgsl */ `
// ---------------- mixture of experts (engine/wgsl/moe.js) ----------------
struct MOE { dOut: u32, dIn: u32, K: u32, nExp: u32, xs: u32, ys: u32, norm: u32, pad: u32 };
var<workgroup> moe_red: array<f32, ${2 * WG}>;

// Router: softmax over all experts, the K largest (ties: lower index), their probabilities as
// weights, renormalised to sum 1 when norm = 1 (norm_topk_prob). One workgroup per column; the
// sums run serially on thread 0 so the order is fixed.
@group(1) @binding(0) var<storage, read> mr_l: array<f32>;
@group(1) @binding(1) var<storage, read_write> mr_sel: array<u32>;
@group(1) @binding(2) var<storage, read_write> mr_w: array<f32>;
@group(1) @binding(3) var<uniform> mr_s: MOE;
var<workgroup> mr_p: array<f32, 1024>;
var<workgroup> mr_m: array<f32, 256>;
@compute @workgroup_size(256)
fn moe_router(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let col = wg.x; let t = lid.x; let n = mr_s.nExp; let K = mr_s.K;
  let lb = col * mr_s.xs;
  var m: f32 = -3.0e38;
  for (var i: u32 = t; i < n; i += 256u) { m = max(m, mr_l[lb + i]); }
  mr_m[t] = m;
  workgroupBarrier();
  for (var st: u32 = 128u; st > 0u; st >>= 1u) {
    if (t < st) { mr_m[t] = max(mr_m[t], mr_m[t + st]); }
    workgroupBarrier();
  }
  let mx = mr_m[0];
  for (var i: u32 = t; i < n; i += 256u) { mr_p[i] = exp(mr_l[lb + i] - mx); }
  workgroupBarrier();
  if (t == 0u) {
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < n; i++) { sum += mr_p[i]; }
    for (var i: u32 = 0u; i < n; i++) { mr_p[i] = mr_p[i] / sum; }
    var tot: f32 = 0.0;
    for (var k: u32 = 0u; k < K; k++) {
      var best: u32 = 0u; var bv: f32 = -1.0;
      for (var i: u32 = 0u; i < n; i++) { if (mr_p[i] > bv) { bv = mr_p[i]; best = i; } }
      mr_sel[col * K + k] = best; mr_w[col * K + k] = bv; tot += bv;
      mr_p[best] = -2.0;   // taken
    }
    if (mr_s.norm == 1u) { for (var k: u32 = 0u; k < K; k++) { mr_w[col * K + k] = mr_w[col * K + k] / tot; } }
  }
}
${guKernel("q4")}
${guKernel("q8")}
${dnKernel("q4")}
${dnKernel("q8")}

// Combine: x[col] += sum_k w_k y[col,k] (k in order) + sigmoid(sg[col]) * shared[col].
// dOut = dim, xs = x column stride, ys = y (col,slot) stride, nExp (reused) = shared column stride,
// norm (reused) = 1 when there is a shared expert, pad (reused) = shared-gate logit column stride.
@group(1) @binding(0) var<storage, read_write> mc_x: array<f32>;
@group(1) @binding(1) var<storage, read> mc_y: array<f32>;
@group(1) @binding(2) var<storage, read> mc_w: array<f32>;
@group(1) @binding(3) var<storage, read> mc_sh: array<f32>;
@group(1) @binding(4) var<storage, read> mc_sg: array<f32>;
@group(1) @binding(5) var<uniform> mc_s: MOE;
@compute @workgroup_size(64)
fn moe_combine(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; let col = gid.y; let S = mc_s;
  if (i >= S.dOut) { return; }
  var o: f32 = 0.0;
  for (var k: u32 = 0u; k < S.K; k++) { o += mc_w[col * S.K + k] * mc_y[(col * S.K + k) * S.ys + i]; }
  if (S.norm == 1u) {
    let g = mc_sg[col * S.pad];
    o += (1.0 / (1.0 + exp(-g))) * mc_sh[col * S.nExp + i];
  }
  mc_x[col * S.xs + i] += o;
}
`;
}
