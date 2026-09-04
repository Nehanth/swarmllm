// Row-stationary Q4_0 GEMM for wide prefill passes (16 token columns).
//
// The batched GEMV (matvec_q4_coop_b) is column-stationary: it reloads the
// activation tile per row-group and runs at ~26 GB/s on 16 columns. This kernel
// makes each thread own R weight rows and ALL N columns, keeps the weight tile
// in shared memory as packed nibbles (vec4<u32>, never dequantized to shared),
// and broadcast-reads activations, giving 8 vec4 FMAs per shared load instead
// of ~5. Measured on a GB10: 0.757 ms vs 1.919 ms for the 4-column GEMV on
// 17408x5120x16, relDiff 6.3e-7. See docs/research/prefill-gemm-v2.md.
//
// Split-K (S workgroups per row tile, each covering 1/S of the reduction) keeps
// tall-tile shapes occupied; partials are summed by gemm_red in a fixed order,
// so results are run-to-run deterministic. S is PINNED per shape, never
// autotuned: every peer in a room must produce identical hidden states.
//
// Bindings reuse the matvec_q4_coop_b layout verbatim (qs, sc, x, y, shape).
// The only new declarations are type aliases of bindings 0 and 2, which is
// legal because no single entry point references two views of one binding.

export const GEMM_TILE = 128;          // rows per workgroup tile (T * R)

// Pinned split-K factor per output shape. Keys are `${dOut}x${dIn}`.
// Chosen by benchmark (docs/research/prefill-gemm-v2.md S5); treat edits as a
// protocol change under GOVERNANCE.md.
export const GEMM_S = {
  "17408x5120": 4, "5120x17408": 2, "5120x6144": 8, "10240x5120": 2,
  "6144x5120": 4, "12288x5120": 2, "1024x5120": 16, "5120x5120": 4,
};

export function gemmWGSL({ N = 16, T = 64, R = 2, KB = 2, splits = [2, 4, 8, 16],
                          dIns = [5120, 6144, 17408], pairs = null, UNPACK = true } = {}) {
  // pairs: [[dIn, S], ...] to emit exactly the kernels a model needs (each is
  // fully unrolled, so the module size and shader compile time scale with it)
  const rng = (n) => Array.from({ length: n }, (_, i) => i);
  const TM = T * R, WV = TM * KB, WPT = WV / T, XV = 32 * KB * (N / 4), XPT = XV / T;
  const RR = rng(R), QN = rng(N / 4);
  if (WV % T || XV % T) throw new Error("gemm: T must divide the stage sizes");
  if (N % 4) throw new Error("gemm: N must be a multiple of 4");
  const dq = (m, s) => UNPACK
    ? `(vec4<f32>(unpack4xU8(${m})) - vec4<f32>(8.0)) * ${s}`
    : `(vec4<f32>(f32(${m} & 0xFFu), f32((${m} >> 8u) & 0xFFu), f32((${m} >> 16u) & 0xFFu), f32(${m} >> 24u)) - vec4<f32>(8.0)) * ${s}`;

  const kernel = (dIn, S) => {
    const nb = dIn / 32, nStages = nb / KB;
    if (nb % 2) throw new Error("gemm: dIn must be a multiple of 64 (f16 scales are paired)");
    if (nStages % S) throw new Error(`gemm: S=${S} must divide ${nStages} stages for dIn=${dIn}`);
    const stPerWG = nStages / S;
    return `
@compute @workgroup_size(${T})
fn gemm_q4_${dIn}_s${S}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let dOut = qb_shape.dOut;
  let wgl = wg.y * 32768u + wg.x;                     // 2-D dispatch (trick 21)
  let tile = wgl / ${S}u; let split = wgl % ${S}u;
  let row0 = tile * ${TM}u; let st0 = split * ${stPerWG}u; let rb = row0 + t * ${R}u;
  ${RR.map((r) => QN.map((q) => `var a${r}_${q} = vec4<f32>(0.0);`).join(" ")).join("\n  ")}
  ${rng(WPT).map((j) => `let li${j} = t + ${j * T}u; let lr${j} = min(row0 + li${j} / ${KB}u, dOut - 1u); let lb${j} = li${j} % ${KB}u;`).join("\n  ")}
  ${rng(WPT).map((j) => `var w${j} = gm_qs4[lr${j} * ${nb}u + st0 * ${KB}u + lb${j}];`).join("\n  ")}
  ${rng(XPT).map((j) => `var xv${j} = gm_xT[st0 * ${XV}u + t + ${j * T}u];`).join("\n  ")}
  for (var s: u32 = 0u; s < ${stPerWG}u; s++) {
    workgroupBarrier();
    ${rng(WPT).map((j) => `gm_W[li${j}] = w${j};`).join(" ")}
    ${rng(XPT).map((j) => `gm_X[t + ${j * T}u] = xv${j};`).join(" ")}
    workgroupBarrier();
    let bs = (st0 + s) * ${KB}u;
    if (s + 1u < ${stPerWG}u) {                       // one-deep register prefetch
      ${rng(WPT).map((j) => `w${j} = gm_qs4[lr${j} * ${nb}u + bs + ${KB}u + lb${j}];`).join(" ")}
      ${rng(XPT).map((j) => `xv${j} = gm_xT[(st0 + s + 1u) * ${XV}u + t + ${j * T}u];`).join(" ")}
    }
    ${RR.map((r) => rng(Math.max(1, KB >> 1)).map((pp) => `let sw${r}_${pp} = q4_sc[((min(rb + ${r}u, dOut - 1u) * ${nb}u + bs) >> 1u) + ${pp}u];`).join(" ")).join(" ")}
    ${rng(KB).map((b) => `
    {
      ${RR.map((r) => `let sv${r} = unpack2x16float(sw${r}_${b >> 1})[${b & 1}u];`).join(" ")}
      ${RR.map((r) => `let wa${r} = gm_W[(t * ${R}u + ${r}u) * ${KB}u + ${b}u];`).join("\n      ")}
      ${rng(4).map((j) => `
      { ${RR.map((r) => `let lo${r} = ${dq(`(wa${r}[${j}] & 0x0F0F0F0Fu)`, `sv${r}`)}; let hi${r} = ${dq(`((wa${r}[${j}] >> 4u) & 0x0F0F0F0Fu)`, `sv${r}`)};`).join(" ")}
        ${rng(4).map((i) => { const kl = 32 * b + 4 * j + i, kh = kl + 16; return `
        { ${QN.map((q) => `let xl${q} = gm_X[${kl * (N / 4) + q}u];`).join(" ")} ${RR.map((r) => QN.map((q) => `a${r}_${q} += lo${r}[${i}] * xl${q};`).join(" ")).join(" ")} }
        { ${QN.map((q) => `let xh${q} = gm_X[${kh * (N / 4) + q}u];`).join(" ")} ${RR.map((r) => QN.map((q) => `a${r}_${q} += hi${r}[${i}] * xh${q};`).join(" ")).join(" ")} }`; }).join("")}
      }`).join("")}
    }`).join("")}
  }
  let pb = split * ${N}u * dOut;                      // partials: p[split][col][dOut]
  ${RR.map((r) => `if (rb + ${r}u < dOut) { ${QN.map((q) => rng(4).map((c) => `q4_y[pb + ${4 * q + c}u * dOut + rb + ${r}u] = a${r}_${q}[${c}];`).join(" ")).join(" ")} }`).join("\n  ")}
}`;
  };

  // Fixed-order split-K reduce. `_acc` folds the residual add in, exactly like
  // matvec_*_coop_b_acc. Writes into the engine's column-strided y layout.
  const reduce = (S, ACC) => `
@compute @workgroup_size(64)
fn gemm_red_s${S}${ACC ? "_acc" : ""}(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; let dOut = qb_shape.dOut; let n = ${N}u * dOut;
  if (i >= n) { return; }
  var acc = 0.0;
  ${rng(S).map((s) => `acc += gm_p[${s}u * n + i];`).join(" ")}
  q4_y[(i / dOut) * qb_shape.ys + (i % dOut)] ${ACC ? "+=" : "="} acc;
}`;

  // Column-major staging of the activations the GEMM broadcast-reads.
  // src is the engine's [col][dIn] layout with a 256-byte-aligned column stride.
  const xpose = `
@compute @workgroup_size(64)
fn gemm_xpose(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; let dIn = qb_shape.dIn;
  if (i >= dIn * ${N}u) { return; }
  q4_y[i] = gm_p[(i % ${N}u) * (qb_shape.xs4 * 4u) + i / ${N}u];
}`;

  const kernels = [];
  const want = pairs ?? dIns.flatMap((dIn) => splits.map((S) => [dIn, S]));
  const seen = new Set();
  for (const [dIn, S] of want) {
    const key = `${dIn}:${S}`; if (seen.has(key)) continue; seen.add(key);
    if (((dIn / 32) / KB) % S === 0) kernels.push(kernel(dIn, S));
  }
  const usedSplits = [...new Set(want.map(([, S]) => S))];
  return /* wgsl */ `
// ---- row-stationary Q4_0 GEMM (N=${N}, T=${T}, R=${R}, KB=${KB}) ----
// Aliases of bindings already declared by the matvec kernels. Legal because no
// entry point below references more than one view of the same binding.
@group(1) @binding(0) var<storage, read> gm_qs4: array<vec4<u32>>;   // packed nibbles
@group(1) @binding(0) var<storage, read> gm_p: array<f32>;           // split-K partials / xpose source
@group(1) @binding(2) var<storage, read> gm_xT: array<vec4<f32>>;    // column-major activations
var<workgroup> gm_W: array<vec4<u32>, ${WV}>;
var<workgroup> gm_X: array<vec4<f32>, ${XV}>;
${kernels.join("\n")}
${usedSplits.map((S) => reduce(S, false) + reduce(S, true)).join("\n")}
${xpose}
`;
}
