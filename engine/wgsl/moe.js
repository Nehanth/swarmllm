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

const ROWS = 4;

// Expert GEMV kernels, laid out like the dense cooperative GEMV (coop.js): four threads share a 32-weight
// block (one Q4 word or two Q8 words each), x is read once per block as two vec4s and reused for every row
// in the workgroup, and dequant is unpack4xU8 / unpack4xI8 into vec4 dots. Workgroup y = (column, slot):
// every (column, slot) pair does the same math whether a pass has 1 column or many, so the batched path
// stays bit-identical to the one-token path.
const q4lo = (w) => `vec4<f32>(unpack4xU8(${w} & 0x0F0F0F0Fu)) - vec4<f32>(8.0)`;
const q4hi = (w) => `vec4<f32>(unpack4xU8((${w} >> 4u) & 0x0F0F0F0Fu)) - vec4<f32>(8.0)`;
const i8x4 = (w) => `vec4<f32>(unpack4xI8(${w}))`;
// one thread's share of block b of expert row er: x from vec4 array X at vec4 offset xc
function term(fmt, Q, SC, X, er, xc) {
  const sc = `unpack2x16float(${SC}[(${er} * nb + b) >> 1u])[(${er} * nb + b) & 1u]`;
  if (fmt === "q4") return `${sc} * (dot(${q4lo(`${Q}[(${er} * nb + b) * 4u + qt]`)}, ${X}[${xc} + b * 8u + qt]) + dot(${q4hi(`${Q}[(${er} * nb + b) * 4u + qt]`)}, ${X}[${xc} + b * 8u + qt + 4u]))`;
  return `${sc} * (dot(${i8x4(`${Q}[(${er} * nb + b) * 8u + qt * 2u]`)}, ${X}[${xc} + b * 8u + qt * 2u]) + dot(${i8x4(`${Q}[(${er} * nb + b) * 8u + qt * 2u + 1u]`)}, ${X}[${xc} + b * 8u + qt * 2u + 1u]))`;
}
const tree = (WG, n, red) => `
  workgroupBarrier();
  for (var st: u32 = ${WG / 2}u; st > 0u; st >>= 1u) {
    if (t < st) {
${Array.from({ length: n }, (_, r) => `      ${red}[${r * WG}u + t] += ${red}[${r * WG}u + t + st];`).join("\n")}
    }
    workgroupBarrier();
  }`;

function guKernel(fmt, WG) {
  const P = `mg${fmt}`, LANES = WG / 4;
  const rows = (f) => Array.from({ length: ROWS }, (_, r) => f(r)).join("\n");
  return `
@group(1) @binding(0) var<storage, read> ${P}_gq: array<u32>;
@group(1) @binding(1) var<storage, read> ${P}_gs: array<u32>;
@group(1) @binding(2) var<storage, read> ${P}_uq: array<u32>;
@group(1) @binding(3) var<storage, read> ${P}_us: array<u32>;
@group(1) @binding(4) var<storage, read> ${P}_x: array<vec4<f32>>;
@group(1) @binding(5) var<storage, read_write> ${P}_h: array<f32>;
@group(1) @binding(6) var<storage, read> ${P}_sel: array<u32>;
@group(1) @binding(7) var<uniform> ${P}_s: MOE;
var<workgroup> ${P}_red: array<f32, ${2 * ROWS * WG}>;
@compute @workgroup_size(${WG})
fn moe_gu_${fmt}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let S = ${P}_s; let t = lid.x; let cs = wg.y; let qt = t & 3u; let bl = t >> 2u;
  let e = ${P}_sel[cs]; let nb = S.dIn / 32u; let row0 = wg.x * ${ROWS}u; let xc = (cs / S.K) * (S.xs / 4u);
${rows((r) => `  var g${r}: f32 = 0.0; var u${r}: f32 = 0.0; let er${r} = e * S.dOut + min(row0 + ${r}u, S.dOut - 1u);`)}
  for (var b: u32 = bl; b < nb; b += ${LANES}u) {
${rows((r) => `    g${r} += ${term(fmt, `${P}_gq`, `${P}_gs`, `${P}_x`, `er${r}`, "xc")};\n    u${r} += ${term(fmt, `${P}_uq`, `${P}_us`, `${P}_x`, `er${r}`, "xc")};`)}
  }
${rows((r) => `  ${P}_red[${r * WG}u + t] = g${r}; ${P}_red[${(ROWS + r) * WG}u + t] = u${r};`)}
${tree(WG, 2 * ROWS, `${P}_red`)}
  if (t < ${ROWS}u) {
    let row = row0 + t;
    if (row < S.dOut) { let gg = ${P}_red[t * ${WG}u]; ${P}_h[cs * S.ys + row] = gg / (1.0 + exp(-gg)) * ${P}_red[(${ROWS}u + t) * ${WG}u]; }
  }
}`;
}

function dnKernel(fmt, WG) {
  const P = `md${fmt}`, LANES = WG / 4;
  const rows = (f) => Array.from({ length: ROWS }, (_, r) => f(r)).join("\n");
  return `
@group(1) @binding(0) var<storage, read> ${P}_q: array<u32>;
@group(1) @binding(1) var<storage, read> ${P}_sc: array<u32>;
@group(1) @binding(2) var<storage, read> ${P}_x: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> ${P}_y: array<f32>;
@group(1) @binding(4) var<storage, read> ${P}_sel: array<u32>;
@group(1) @binding(5) var<uniform> ${P}_s: MOE;
var<workgroup> ${P}_red: array<f32, ${ROWS * WG}>;
@compute @workgroup_size(${WG})
fn moe_dn_${fmt}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let S = ${P}_s; let t = lid.x; let cs = wg.y; let qt = t & 3u; let bl = t >> 2u;
  let e = ${P}_sel[cs]; let nb = S.dIn / 32u; let row0 = wg.x * ${ROWS}u; let xc = cs * (S.xs / 4u);   // each (column, slot) has its own input h
${rows((r) => `  var y${r}: f32 = 0.0; let er${r} = e * S.dOut + min(row0 + ${r}u, S.dOut - 1u);`)}
  for (var b: u32 = bl; b < nb; b += ${LANES}u) {
${rows((r) => `    y${r} += ${term(fmt, `${P}_q`, `${P}_sc`, `${P}_x`, `er${r}`, "xc")};`)}
  }
${rows((r) => `  ${P}_red[${r * WG}u + t] = y${r};`)}
${tree(WG, ROWS, `${P}_red`)}
  if (t < ${ROWS}u) { let row = row0 + t; if (row < S.dOut) { ${P}_y[cs * S.ys + row] = ${P}_red[t * ${WG}u]; } }
}`;
}

export function moeWGSL() {
  return /* wgsl */ `
// ---------------- mixture of experts (engine/wgsl/moe.js) ----------------
struct MOE { dOut: u32, dIn: u32, K: u32, nExp: u32, xs: u32, ys: u32, norm: u32, pad: u32 };

// Router: softmax over all experts, the K largest probabilities (ties: lower index), their probabilities as
// weights, renormalised to sum 1 when norm = 1 (norm_topk_prob). One workgroup of 256 per column; max and sum
// are tree reductions and each of the K picks is an argmax tree over (probability, index), so every step is
// parallel and the order of every sum is fixed.
@group(1) @binding(0) var<storage, read> mr_l: array<f32>;
@group(1) @binding(1) var<storage, read_write> mr_sel: array<u32>;
@group(1) @binding(2) var<storage, read_write> mr_w: array<f32>;
@group(1) @binding(3) var<uniform> mr_s: MOE;
var<workgroup> mr_p: array<f32, 1024>;
var<workgroup> mr_v: array<f32, 256>;
var<workgroup> mr_i: array<u32, 256>;
var<workgroup> mr_k: array<f32, 16>;
@compute @workgroup_size(256)
fn moe_router(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let col = wg.x; let t = lid.x; let n = mr_s.nExp; let K = mr_s.K;
  let lb = col * mr_s.xs;
  var m: f32 = -3.0e38;
  for (var i: u32 = t; i < n; i += 256u) { m = max(m, mr_l[lb + i]); }
  mr_v[t] = m;
  workgroupBarrier();
  for (var st: u32 = 128u; st > 0u; st >>= 1u) { if (t < st) { mr_v[t] = max(mr_v[t], mr_v[t + st]); } workgroupBarrier(); }
  let mx = mr_v[0];
  workgroupBarrier();
  var s: f32 = 0.0;
  for (var i: u32 = t; i < n; i += 256u) { let p = exp(mr_l[lb + i] - mx); mr_p[i] = p; s += p; }
  mr_v[t] = s;
  workgroupBarrier();
  for (var st: u32 = 128u; st > 0u; st >>= 1u) { if (t < st) { mr_v[t] += mr_v[t + st]; } workgroupBarrier(); }
  let inv = 1.0 / mr_v[0];
  for (var i: u32 = t; i < n; i += 256u) { mr_p[i] = mr_p[i] * inv; }
  workgroupBarrier();
  for (var k: u32 = 0u; k < K; k++) {
    var bv: f32 = -1.0; var bi: u32 = 0u;
    for (var i: u32 = t; i < n; i += 256u) { if (mr_p[i] > bv) { bv = mr_p[i]; bi = i; } }
    mr_v[t] = bv; mr_i[t] = bi;
    workgroupBarrier();
    for (var st: u32 = 128u; st > 0u; st >>= 1u) {
      if (t < st) { let ov = mr_v[t + st]; let oi = mr_i[t + st]; if (ov > mr_v[t] || (ov == mr_v[t] && oi < mr_i[t])) { mr_v[t] = ov; mr_i[t] = oi; } }
      workgroupBarrier();
    }
    if (t == 0u) { mr_sel[col * K + k] = mr_i[0]; mr_k[k] = mr_v[0]; mr_p[mr_i[0]] = -2.0; }   // taken
    workgroupBarrier();
  }
  if (t == 0u) {
    var tot: f32 = 0.0;
    for (var k: u32 = 0u; k < K; k++) { tot += mr_k[k]; }
    for (var k: u32 = 0u; k < K; k++) { mr_w[col * K + k] = select(mr_k[k], mr_k[k] / tot, mr_s.norm == 1u); }
  }
}
${guKernel("q4", 256)}
${guKernel("q8", 256)}
${dnKernel("q4", 64)}
${dnKernel("q8", 64)}

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
