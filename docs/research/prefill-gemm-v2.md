# SwarmLLM prefill GEMM: implementation-ready design

**Scope.** A complete WGSL kernel (row-stationary Q4_0 GEMM, 16 token columns) that replaces the batched GEMV on the prefill path, the integration path through the existing `_dop` twin-kernel hook, and a validation plan. Everything measured on the GB10 (DGX Spark, Deno 2.9.5/wgpu Vulkan) with the GPU otherwise idle. Claims not backed by a measurement in this document are marked **UNCERTAIN**.

Source files read: `docs/architecture.md`, `docs/kernels.md`, `docs/protocol.md`, `docs/bench-log.md`, `docs/deltanet-prefill-spec.md`, `docs/kernel-plan-3.md`, `roadmap/02-prefill-gemm.md`, `engine/wgsl/{base,coop,qwen35}.js`, `engine/qwen35.js`, `engine/autotune.js`, `room.js`, `benchmarks/bench_gemm.js`, `benchmarks/bench_gemm_rowstat.js`, `tests/*`.

---

## 0. Headline, up front

| Whole-model matvec inventory for 16 prompt tokens (48 DeltaNet + 16 attention layers, 13.70 GB of Q4_0 weights per pass, distinct 13.7 GB working set, min of 3×3 reps, two independent runs) | ms | GB/s |
|---|---|---|
| `coop_b` 4 columns × 4 dispatches (today's default path) | 506.7 / 506.9 | 27 |
| `coop_b` 8 columns × 2 dispatches (today's room path) | 552.5 / 553.0 | 25 |
| **row-stationary GEMM, 16 columns** (+ transposes + split-K reduces) | **199.5 / 199.7** | **69** |
| control: `coop` 1-column GEMV over the same inventory (= one decode token) | 103.2 / 103.9 | 133 |

**2.54× on the matvec family vs the 4-column path, 2.77× vs the 8-column path**, reproducible to ±0.3%.

Honest end-to-end projection, and the part roadmap 02 will not hit:

- Calibration: the engine measures the identical 1-column inventory at 74.6 ms (82 ms for 15.05 GB incl. the LM head, `docs/kernels.md`), this harness at 103.5 ms → the harness is **1.26–1.39× slower than in-engine** (the range comes from cross-checking against the engine's own `matvec_*_b` figures of 108/170/401 ms at 4/8/16 columns in `docs/deltanet-prefill-spec.md`).
- In-engine matvec for 16 tokens: **~145–160 ms** (from 432 ms at 4 cols × 4, or 340–401 ms at 8/16 cols).
- Non-matvec glue at 16 columns is 58 ms (459 − 401, same doc). Pass ≈ **205–220 ms → 73–78 tok/s at the pass level**, from 37 tok/s (8-col path) / 30 tok/s (4-col path). **~2.0–2.4×.**
- **Roadmap 02's "≥100 tok/s on the GB10" is not reachable from this kernel alone.** The floor is 13.70 GB / 184 GB/s = 74.5 ms of weight streaming plus 58 ms of glue = 132 ms = 121 tok/s. This GEMM lands at ~47% of the streaming roofline, so 100 tok/s needs *both* a faster inner loop (§7 items 5–6) *and* glue work. Say so in the bench-log row rather than moving the goalposts.

---

## 1. What the prototype lacks (five defects, each with evidence)

`benchmarks/bench_gemm.js` is correct (2.5e-7) and 1.21–1.25× over `coop_b`. Re-measured here at 1.582 ms vs `coop_b` 1.932 ms on 17408×5120×16. The row-stationary kernel is 0.740 ms. The gap is five distinct defects, not one:

1. **Wrong operand role (the big one).** The prototype is column-stationary: a 2×4 thread tile over a *dequantized f32* shared weight tile. Its inner step is 6 vec4 shared loads (4 x + 2 w) per 32 FMAs, plus 10 vec4 shared *stores* per thread per 64-k stage to publish the dequantized tile. It is shared-memory-issue-bound, which is why register prefetch, 2-blocks-per-barrier and 4×4 tiles all did nothing and only bank-conflict padding moved it (`docs/kernels.md` trick 25). The fix is to make each thread own R weight rows × **all 16 columns**, keep the weight tile in shared memory as **packed nibbles** (`vec4<u32>`, 4 KB instead of 16 KB, zero dequantized stores), and broadcast-read the activations: **8 vec4 FMAs per shared vec4 load** instead of ~5, and 1 global load per 128 FMAs.
2. **No split-K.** Row-stationary tiles are tall (128 rows/WG), so 5120-row shapes get only 40 tiles. Measured on 5120×17408: S=1 = 2.743 ms, S=2 = 1.282, S=4 = 1.338, S=8 = 1.302, S=16 = 1.307. Split-K is worth **2.1×** there and 1.18× on the FFN shape (0.870 → 0.740).
3. **The 16-column width is nearly free and the prototype never exploits it.** Same process, twice: 16 tokens as one 16-column GEMM = 0.723/0.737 ms; as two 8-column GEMMs = 1.195/1.210 ms; a single 8-column GEMM alone = 0.705/0.709 ms. **Columns 9–16 cost 4%.** The kernel is weight-load/latency-bound, not FMA-bound. 32 columns regresses (1.449 vs 1.297 for 2×16), so **16 is the width**.
4. **Dynamic shared-array indexing.** The prototype indexes `Ws`/`Xs` with runtime `q`, `rq`, `c0`; naga's `index: Restrict` policy clamps every one of those on Vulkan *and* Metal (it is not covered by NVIDIA's `robustBufferAccess2`). The row-stationary kernel indexes `gm_X` with literals only (`trick 2`, generated unrolled).
5. **Tiny shapes routed into the tile kernel.** The 96 `wBeta`/`wAlpha` ops per pass are 48×5120 — one 128-row tile, one workgroup. Measured: including them costs **+42 ms of a 243 ms pass (17%)** for 0.28 MB of weights. They must stay on the GEMV.

Non-defects, measured, so nobody re-litigates them: split-K partial traffic + the transpose dispatches cost **6.3 ms of 243 ms (2.6%)** in the full inventory (no-xT 241.6, no-reduce 237.7, kernels only 236.7). Do **not** spend time fusing them into neighbours in phase 1. `T=32` beat `T=64` by 30% on the down-projection in isolation (0.843 vs 1.176) but made no difference in the full inventory (241.9 vs 243.0) — keep one `T`.

---

## 2. The kernel

New file `engine/wgsl/gemm.js`, spliced into the same shader module as `base.js` + `coop.js` + `wgsl/qwen35.js`. It reuses the existing `matvec_q4_coop_b` bind group layout verbatim (`["ro","ro","ro","rw","u"]` = qs, sc, x, y, shape) and adds exactly **one** new declaration, a `vec4<u32>` alias of the nibble buffer at `@group(1) @binding(0)` (trick 7: legal because no entry point references both views). `q4_sc` (b1), `q4_x4` (b2, bound to the *transposed* activations), `q4_y` (b3, bound to the split-K partials) and `qb_shape` (b4) are reused as-is.

**Verified:** this generator compiles clean inside the real module (`WGSL + coopWGSL(256,4,64,16,1,unpack) + gemmWGSL(...) + WGSL2`, 453 KB, 0 errors) and matches `matvec_q4_coop_b` on every engine shape.

```js
// engine/wgsl/gemm.js — row-stationary Q4_0 GEMM for wide prefill passes.
//
// N: token columns (16). T: threads/WG. R: rows/thread (TM = T*R rows per tile).
// KB: Q4_0 blocks staged per barrier pair. S: split-K factor, PINNED per shape.
// Workgroup memory: gm_W = TM*KB vec4<u32> (4 KB) + gm_X = 32*KB*N/4 vec4<f32>
// (4 KB) = 8 KB, under the 16 KB WebGPU default (device reports 16384).
export function gemmWGSL({ N = 16, T = 64, R = 2, KB = 2,
                           splits = [2, 4, 8, 16], dIns = [5120, 6144, 17408],
                           UNPACK = true }) {
  const rng = (n) => Array.from({ length: n }, (_, i) => i);
  const TM = T * R, WV = TM * KB, WPT = WV / T, XV = 32 * KB * (N / 4), XPT = XV / T;
  const RR = rng(R), QN = rng(N / 4);
  if (WV % T || XV % T) throw new Error("gemm: T must divide the stage sizes");
  const dq = (w, s) => UNPACK
    ? `(vec4<f32>(unpack4xU8(${w})) - vec4<f32>(8.0)) * ${s}`
    : `(vec4<f32>(f32(${w} & 0xFFu), f32((${w} >> 8u) & 0xFFu), f32((${w} >> 16u) & 0xFFu), f32(${w} >> 24u)) - vec4<f32>(8.0)) * ${s}`;

  const body = (dIn, S) => {
    const nb = dIn / 32, nStages = nb / KB, stPerWG = nStages / S;
    if (nb % 2) throw new Error("gemm: dIn must be a multiple of 64 (paired f16 scales)");
    if (nStages % S) throw new Error(`gemm: S=${S} must divide ${nStages} stages`);
    return `
@compute @workgroup_size(${T})
fn gemm_q4_${dIn}_s${S}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let dOut = qb_shape.dOut;
  let wgl = wg.y * 32768u + wg.x;                 // 2-D dispatch (trick 21)
  let tile = wgl / ${S}u; let split = wgl % ${S}u;
  let row0 = tile * ${TM}u; let st0 = split * ${stPerWG}u; let rb = row0 + t * ${R}u;
  ${RR.map((r) => QN.map((q) => `var a${r}_${q} = vec4<f32>(0.0);`).join(" ")).join("\n  ")}
  ${rng(WPT).map((j) => `let li${j} = t + ${j * T}u; let lr${j} = min(row0 + li${j} / ${KB}u, dOut - 1u); let lb${j} = li${j} % ${KB}u;`).join("\n  ")}
  ${rng(WPT).map((j) => `var w${j} = gm_qs4[lr${j} * ${nb}u + st0 * ${KB}u + lb${j}];`).join("\n  ")}
  ${rng(XPT).map((j) => `var xv${j} = q4_x4[st0 * ${XV}u + t + ${j * T}u];`).join("\n  ")}
  for (var s: u32 = 0u; s < ${stPerWG}u; s++) {
    workgroupBarrier();
    ${rng(WPT).map((j) => `gm_W[li${j}] = w${j};`).join(" ")}
    ${rng(XPT).map((j) => `gm_X[t + ${j * T}u] = xv${j};`).join(" ")}
    workgroupBarrier();
    let bs = (st0 + s) * ${KB}u;
    if (s + 1u < ${stPerWG}u) {                   // 1-deep register prefetch
      ${rng(WPT).map((j) => `w${j} = gm_qs4[lr${j} * ${nb}u + bs + ${KB}u + lb${j}];`).join(" ")}
      ${rng(XPT).map((j) => `xv${j} = q4_x4[(st0 + s + 1u) * ${XV}u + t + ${j * T}u];`).join(" ")}
    }
    ${RR.map((r) => rng(KB >> 1).map((pp) => `let sw${r}_${pp} = q4_sc[((min(rb + ${r}u, dOut - 1u) * ${nb}u + bs) >> 1u) + ${pp}u];`).join(" ")).join(" ")}
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
  let pb = split * ${N}u * dOut;                  // partials are packed: p[s][col][dOut]
  ${RR.map((r) => `if (rb + ${r}u < dOut) { ${QN.map((q) => rng(4).map((c) => `q4_y[pb + ${4 * q + c}u * dOut + rb + ${r}u] = a${r}_${q}[${c}];`).join(" ")).join(" ")} }`).join("\n  ")}
}`;
  };

  // deterministic fixed-order split-K reduce (WGSL has no f32 atomics anyway);
  // _acc form folds the residual add in, exactly like matvec_*_coop_b_acc.
  const reduce = (S, ACC) => `
@compute @workgroup_size(64)
fn gemm_red_s${S}${ACC ? "_acc" : ""}(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; let dOut = qb_shape.dOut; let n = ${N}u * dOut;
  if (i >= n) { return; }
  var acc = 0.0;
  ${rng(S).map((s) => `acc += gm_p[${s}u * n + i];`).join(" ")}
  q4_y[(i / dOut) * qb_shape.ys + (i % dOut)] ${ACC ? "+=" : "="} acc;
}`;

  return /* wgsl */ `
// ---- row-stationary Q4_0 GEMM (${N} cols, ${T} threads, ${R} rows/thread, ${KB} blocks/stage) ----
@group(1) @binding(0) var<storage, read> gm_qs4: array<vec4<u32>>;   // vec4 view of q4_qs
@group(1) @binding(0) var<storage, read> gm_p: array<f32>;           // partials / xpose source
var<workgroup> gm_W: array<vec4<u32>, ${WV}>;   // packed nibbles, ${WV * 16} B
var<workgroup> gm_X: array<vec4<f32>, ${XV}>;   // k-major activations, ${XV * 16} B
${dIns.flatMap((d) => splits.filter((S) => (d / 32 / KB) % S === 0).map((S) => body(d, S))).join("")}
${splits.flatMap((S) => [false, true].map((a) => reduce(S, a))).join("")}
// [col][stride] -> k-major [k][${N}] staging for gm_X. dOut = elements per column,
// xs4 = source column stride in floats (the engine's 256-B aligned stride).
@compute @workgroup_size(64)
fn gemm_xpose(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; let n = qb_shape.dOut;
  if (i >= n * ${N}u) { return; }
  q4_y[i] = gm_p[(i % ${N}u) * qb_shape.xs4 + i / ${N}u];
}`;
}
```

### 2.1 Pinned constants and the S table

`T = 64, R = 2, KB = 2, N = 16` for every shape. `S` is **pinned per shape** — it changes the summation order and therefore the bits, so it must never be autotuned per device (see §5).

| op | shape (dOut×dIn) | S | tiles×S = WGs | GEMM ms | `coop_b` 4c×4 ms | ratio |
|---|---|---|---|---|---|---|
| `ffn_gate`, `ffn_up` | 17408×5120 | 2 | 272 | 0.740 | 1.932 | 2.61× |
| `wq` | 12288×5120 | 2 | 192 | 0.586 | 1.417 | 2.42× |
| `wqkv` | 10240×5120 | 2 | 160 | 0.557 | 1.161 | 2.08× |
| `wz` | 6144×5120 | 4 | 192 | 0.369 | 0.673 | 1.82× |
| `wo`, `wOut` (acc) | 5120×6144 | 8 | 320 | 0.406 | 0.727 | 1.79× |
| `ffn_down` (acc) | 5120×17408 | 2 | 80 | 1.176 | 2.129 | 1.81× |
| `wk`, `wv` | 1024×5120 | 16 | 128 | 0.254 | 0.331 | 1.30× |
| `wBeta`, `wAlpha` | 48×5120 | — | — | **GEMV, do not route** | | |

Rows are from separate processes, so **compare within a row only** — the GB10 clock governor moves the in-process streaming probe between 82 and 118 GB/s across runs. Confirm the table with one interleaved run before pinning (`§6.5`). Notable alternates measured but not chosen: `5120×6144` at S=12 is 0.377 (7% better than S=8); `5120×17408` at `T=32,S=4` is 0.843 in isolation but neutral in the full inventory.

### 2.2 Invariants the generator asserts

- `dIn % 64 == 0` — `nb` must be even so the paired-f16-scale index `(row*nb + bs) >> 1` is exact. 5120/6144/17408 all pass.
- `S` divides `nb/KB`. 5120→80, 6144→96, 17408→272.
- **`_acc` ops must use S ≥ 2**, so the residual accumulate lives only in `gemm_red_s*_acc` and no `gemm_*_acc` variant is needed. The table satisfies this (down S=2, wo/wOut S=8).
- `dOut ≥ 2*TM` (=256) to route to the GEMM; below that, GEMV.
- Q4 only. The LM head and `eh_proj` are Q8 but never run in a prefill pass, so no Q8 GEMM is needed. Anything not `kind === "q4"` falls through to the GEMV.

### 2.3 Portability

- Workgroup storage **8 KB**, under the 16 KB WebGPU default → no `requiredLimits` change, works on phones and Safari 26.
- `unpack4xU8` is gated by the existing `probeUnpack`; the shift/mask fallback is generated by `UNPACK: false`. Independently measured neutral (`DQ=magic` = 0.749 vs `unpack` = 0.750 in `bench_gemm_rowstat.js`), so dequant instruction mix is not the limiter — Safari loses nothing.
- No subgroups, f32 accumulate throughout, 2-D dispatch helper retained.
- Module grows from ~330 KB to 453 KB with the pinned S set (5 GEMM + 8 reduce + 1 xpose entry points). Creating 11 pipelines from it took 93 ms. **UNCERTAIN on Metal/Tint**: MSL compile time for 5 fully-unrolled 8 KB-shared kernels is unmeasured; if it hurts load time, generate only the S values the device's model actually needs.

---

## 3. Integration through the twin-kernel hook

The existing hook is `_dop(pass, op, nCols)` at `engine/qwen35.js:352`, which swaps `op.pipe → op.pipe4` when `nCols ≤ 4`. This design turns it into a four-rung ladder. **Nine edits, none of them structural.**

### 3.1 `engine/wgsl/coop.js` — widen the GEMV twin ladder

Two one-line changes, both verified to compile:

```js
// line 371: generate 16 / 8 / 4-column GEMV twins instead of 16 / 4
${[COLS, 8, 4].filter((c, i) => i === 0 || c < COLS).map((C) => ...
// line 401 and line 194: name the twins by width, not by the literal "4"
fn matvec${kind}_coop_b${C === COLS ? "" : C}${ACC ? "_acc" : ""}(...)
fn matvec${kind === "f32" ? "" : "_" + kind}_gu_b${C === COLS ? "" : C}(...)
// line 239: emit the 8-column gu twin too
+ (COLS > 8 ? [...].map((k) => guKernelB(k, 8)).join("\n") : "")
+ (COLS > 4 ? [...].map((k) => guKernelB(k, 4)).join("\n") : "")
```

**Gotcha found by compiling it:** with three widths the current naming produces two functions called `matvec_coop_b4` and the module fails with `redefinition of matvec_coop_b4`. Renaming to `_b8`/`_b4` fixes it (verified, 0 errors, 476 KB module).

The 16-column GEMV is what covers `wBeta`/`wAlpha` (dOut = 48) in a single dispatch. It needs `ROWSB = 1` (16 accumulators/thread), so the engine's `coopRowsB` becomes width-dependent: 1 at 16 columns, 2 at 8, 4 at 4.

### 3.2 `engine/qwen35.js` — five edits

**(a) `_init` (line ~65):** compile the GEMM into the same module and register its layouts.

```js
const mod = device.createShaderModule({ code: WGSL
  + coopWGSL(coopWG, coopRows, 64, batchCols, coopRowsB, unpack)
  + (batchCols >= 16 ? gemmWGSL({ N: batchCols, UNPACK: unpack,
      splits: [...new Set(Object.values(GEMM_S))], dIns: DINS_FOR(meta) }) : "")
  + WGSL2 });
// G1 additions — all three reuse the matvec_q4_coop_b layout verbatim:
//   gemm_q4_<dIn>_s<S>: ["ro","ro","ro","rw","u"]   (qs, sc, xT, partials, shape)
//   gemm_red_s<S>[_acc]: ["ro","ro","ro","rw","u"]  (partials, -, -, y, shape)
//   gemm_xpose:          ["ro","ro","ro","rw","u"]  (src, -, -, dst, shape)
// bindings 1 and 2 are unused by the reduce/xpose entry points; a pipeline
// layout that is a superset of the shader's interface is legal in WebGPU.
```

**(b) `_initBatch` (line 492):** four transposed staging buffers plus two ping-pong partials buffers.

```js
const NT = this.NC;                         // 16
B.xnT   = dev.createBuffer({ size: D.dim    * NT * 4, usage: S });  // 328 KB
B.gT    = dev.createBuffer({ size: D.inter  * NT * 4, usage: S });  // 1.11 MB
B.aoT   = dev.createBuffer({ size: D.dInner * NT * 4, usage: S });  // 393 KB, qDim == dInner == 6144
this.gemmP = [0, 1].map(() => dev.createBuffer({ size: 4 * 1024 * 1024, usage: S }));
// max partials = max(S*N*dOut)*4 = 12*16*5120*4 = 3.9 MB
```

**(c) `mvB` (line 528):** attach a GEMM plan when the shape qualifies.

```js
const mvB = (w, xB, yB, dOut, dIn, acc = false, xT = null) => {
  const op = /* ...unchanged 16/8/4-column GEMV ladder... */;
  const S = GEMM_S[dOut + "x" + dIn];
  if (this.NC >= 16 && w.kind === "q4" && xT && S && dOut >= 256 && dIn % 64 === 0) {
    if (acc && S < 2) throw new Error("gemm: acc ops need S >= 2");
    const shp = this._shapeB(dOut, dIn, xB.stride / 16, yB.stride / 4);
    op.gemm = {
      pipe: `gemm_q4_${dIn}_s${S}`, wgs: Math.ceil(dOut / 128) * S,
      bg:  [0, 1].map((z) => this._bg(this.pipes[`gemm_q4_${dIn}_s${S}`], 1,
             [w.qs, w.sc, xT, this.gemmP[z], shp])),
      red: `gemm_red_s${S}${acc ? "_acc" : ""}`,
      redBg: [0, 1].map((z) => this._bg(this.pipes[...], 1,
               [this.gemmP[z], w.sc, w.sc, yB.buf, shp])),
      redThreads: this.NC * dOut,
    };
  }
  return op;
};
```

**(d) `_dop` (line 352):** the ladder. This is the whole hook.

```js
_dop(pass, op, nCols = 0) {
  if (this.skip && this.skip.has(op.pipe)) return;
  if (op.gemm && nCols === this.NC && this.gemm !== false) {
    const z = this._gz = (this._gz ^ 1);                  // ping-pong the partials
    const g = op.gemm;
    pass.setPipeline(this.pipes[g.pipe]);
    pass.setBindGroup(0, this.bgCommonFor[g.pipe]);
    pass.setBindGroup(1, g.bg[z]);
    if (g.wgs > 32768) pass.dispatchWorkgroups(32768, Math.ceil(g.wgs / 32768));
    else pass.dispatchWorkgroups(g.wgs);
    pass.setPipeline(this.pipes[g.red]);
    pass.setBindGroup(0, this.bgCommonFor[g.red]);
    pass.setBindGroup(1, g.redBg[z]);
    pass.dispatchWorkgroups(Math.ceil(g.redThreads / 64));
    return;
  }
  const w = this.b4 === false ? this.NC : nCols;
  const pipe = w > 0 && w <= 4 && op.pipe4 ? op.pipe4
             : w > 0 && w <= 8 && op.pipe8 ? op.pipe8 : op.pipe;
  /* ...unchanged... */
}
```

`this.gemm = false` is the kill switch required by AGENTS.md ("approximations need a default-off switch") and the A/B lever for the bench-log row.

**(e) `_encodeLayerBatch` (line 641):** four `gemm_xpose` dispatches per layer, emitted only in GEMM mode, each immediately after its producer and inside the same compute pass (WebGPU's implicit intra-pass ordering covers the dependency).

| after | transpose | feeds |
|---|---|---|
| `rmsnorm_mc` (norm1) | `B.xn → B.xnT` | wqkv/wz/wBeta/wAlpha, or wq/wk/wv |
| `dn_gatenorm_mc` / `sigmoid_mul_mc` | `B.gated`/`B.attnOut → B.aoT` | wOut / wo |
| `rmsnorm_mc` (norm2) | `B.xn → B.xnT` | gate/up |
| `gu` (or `silu_mul`) | `B.g → B.gT` | down |

Measured cost of all 256 of them: **1.4 ms per pass (0.6%)**.

**(f) `prefillTokens` (line 869) — required, not optional.** The loop is `while (ids.length - i >= NC)`; at NC=16 a tail of up to 15 tokens would fall into single-token `prefillToken` at ~113 ms each (1.7 s worst case). Add a step-down: run the remainder as one batched pass at 8, then 4, then singles. `_encodeLayerBatch` already takes `nCols` and `_dop` already picks the twin, so this is ~8 lines. (This is also the `deltanet-prefill-research.md` "prompt tail as one batched pass" item.)

### 3.3 `room.js` — one edit, no protocol change

`room.js:919` computes `nChunks = Math.min(16 / NC, ...)`. At NC=16 that is 1 chunk of 16 → **one 160 KB `ai-hidden-b` frame per network round, exactly as `docs/protocol.md` already specifies**. The worker handler (`room.js:1096`) already chunks with `Math.min(NC, nTok - c)`, so partial rounds work unchanged. The one edit is the same tail problem as (f): `while (ids.length - 1 - i >= NC)` leaves up to 15 tokens going through one *network lap each*. Loosen it to allow a final partial batched round.

Room construction (`room.js:664`) becomes `batchCols: 16, coopRowsB: 1`. Memory delta at NC=16: `B.logits` and `stageLogitsN` go 7.9 → 15.9 MB each, the other `B.*` buffers roughly double to ~5 MB total, and per-column bind groups go from 64×8 to 64×16 (~9 k bind groups at load).

---

## 4. Where the time goes after this lands

Engine-equivalent, 16-token pass, GB10 (calibration range from §0):

| | today (8-col × 2) | with the GEMM |
|---|---|---|
| matvec family | 340–432 ms | **145–160 ms** |
| `wBeta`/`wAlpha` (GEMV) | included | ~5 ms |
| glue (`dn_delta_mc`, attention, norms) | 58 ms | 58 ms |
| **pass total** | **432 ms → 37 tok/s** | **~210 ms → ~76 tok/s** |
| `mtpFill` (solo only, 6.4 ms/token measured) | 102 ms (hidden: 24%) | 102 ms (**48%**) |
| **solo end-to-end incl. mtpFill** | ~28 tok/s | **~51 tok/s** |

The MTP prefill fill (`prefillTokens` line 891: one submitted single-column MTP pass per prompt token) goes from a 24% tax to a 48% tax the moment the GEMM lands. It does not affect rooms (the room prefill path never fills the MTP cache at all — a separate known defect). Batching it into a 16-column pass is item 3 in §7.

---

## 5. Numerics and the bit-exactness contract

Measured `relDiff(L2)` of the GEMM against `matvec_q4_coop_b` on the real engine shapes with the engine's 256-B-aligned strides (random Q4 weights, f16 scales in [0.005, 0.035], x ~ U(−1,1)):

| shape | S | relDiff | max abs | ‖y‖rms |
|---|---|---|---|---|
| 17408×5120 | 2 | 8.83e-7 | 3.05e-5 | 4.17 |
| 5120×17408 | 2 | 1.63e-6 | 1.30e-4 | 7.65 |
| 5120×6144 | 8 | 4.95e-7 | 1.19e-5 | 4.58 |
| 10240×5120 | 2 | 8.86e-7 | 3.34e-5 | 4.15 |
| 6144×5120 | 4 | 6.28e-7 | 1.81e-5 | 4.17 |
| 1024×5120 | 16 | 3.31e-7 | 6.68e-6 | 4.21 |

The contract this change must keep, and how it keeps it:

1. **Decode and speculative verify are untouched.** The GEMM fires only at `nCols === this.NC === 16`. A verify pass is 1+K ≤ 8 columns and stays on the GEMV. So "the speculative stream equals plain decoding" (`tests/test_mtp.js`) is preserved *structurally*, not by measurement. This is the single most important property of the design and the reason it is safe to land before Oct 30.
2. **Prefill was already not bit-identical to sequential.** `tests/test_batch_q38.js` gates at `rel < 2e-3` and `tests/test_b4.js` at `1e-3`; today's batched path sits around 1e-7. The GEMM adds 3e-7…1.6e-6 — three orders inside both gates.
3. **`tests/test_q38_full.js` does not exercise prefill at all** (it calls `forwardToken` per prompt token). The 27B golden text is therefore *not* a gate on this change today. §6.3 adds one.
4. **Cross-device determinism.** Every peer in a room must run the same kernel with the same `S` per shape, or hidden states diverge on the wire. `S` is a pinned constant table, never autotuned. Treat a change to that table as a protocol change under GOVERNANCE.md (bump the message version; older peers fail loudly at `ai-load`).
5. **Run-to-run determinism.** The reduce sums `s = 0..S-1` in fixed order; WGSL has no f32 atomics, so there is no nondeterministic path.

---

## 6. Validation plan

Ordered; each step gates the next. Harnesses used in this document are at a scratch directory (not committed) (`rs.js` per-shape sweep, `full.js` whole-inventory, `gemm/gemm.js` + `gemm/check.js` correctness). Promote `full.js` into `benchmarks/bench_gemm_inventory.js` — it is the only harness that measures the shape mix the engine actually runs.

**6.1 Kernel correctness (done, reproduce before landing).** `gemm/check.js`: compile the full module (`base + coop(COLS=16,ROWSB=1) + gemm + qwen35`), run the GEMM and `matvec_q4_coop_b` on the same weights with the engine's aligned strides, assert `relDiff < 5e-6` on all seven shapes. Add the missing case: **`gemm_red_s*_acc` into a pre-filled `y`**, compared against `matvec_q4_coop_b_acc` — the residual-accumulate path is load-bearing for `wo`, `wOut` and `ffn_down` and is the one branch not yet exercised.

**6.2 Edge cases.** `dOut` not a multiple of `TM` (the `min(..., dOut-1)` clamps and the `rb + r < dOut` epilogue guard); `S = 1` (no reduce, non-`acc` only); `nCols < NC` must fall through to the GEMV (assert the GEMM pipeline is never bound); `this.gemm = false` must reproduce today's output bit-for-bit.

**6.3 Engine equivalence.** `tests/test_batch_q38.js` with `BCOLS=16 ROWSB=1` — the 2e-3 gate should see ~1e-6. `tests/test_b4.js` extended to a three-way `seq / b16 / b8 / b4` comparison. **New test, the real gap:** a `test_q38_prefill_full.js` that prefills a ≥64-token prompt through `prefillTokens` (GEMM on) and generates 12 greedy tokens, compared against the existing `test_q38_full.js` golden text. Today nothing gates the 27B prefill path on token identity. Expect a match; a near-tie mismatch at 1e-6 is possible and must be triaged, not waived.

**6.4 Room / split equivalence.** `tests/test_batch_split.js` and `tests/test_q38_split.js` at `batchCols: 16`, with at least one run where the two engines use *different* `S` tables, asserted to **fail** — that is the regression test for invariant 5.4. Then a real two-tab room prefilling a 100-token prompt with per-round `basePos` logging.

**6.5 Performance, in this order.**
1. Confirm the `S` table with one interleaved `rs.js` run per shape (`CFGS` listing the candidates side by side) — pin the winners.
2. `bench_gemm_inventory.js` (`DISTINCT=1`) before/after: the numbers to reproduce are 506.8 / 552.6 / 199.6 ms and the 1-column control at 103.5 ms. Report every kernel as a fraction of the in-process streaming probe *and* record the control, because the GB10 clock governor moves the probe between 82 and 118 GB/s across processes and the ratio to the 1-column control is the only stable cross-run quantity.
3. `benchmarks/bench_breakdown.js` on the real 27B with `batchCols` 8 vs 16 — **this is the gate on the §4 projection**. The projection is calibrated from a microbenchmark and is good to ±20%; do not put a tok/s number in the bench-log until this run exists. Note that skip-family timing under-attributes latency-hidden kernels (`deltanet-prefill-spec.md`), so also record the whole-pass wall clock.
4. `MODEL=q38 benchmarks/bench.js` end-to-end prefill on the 86-token prompt used by the existing bench-log rows, so the new row is comparable to `27.3 tok/s`.
5. Mac (Chrome/Metal) and Safari 26: compile-time, `maxComputeWorkgroupStorageSize` (needs 8 KB, default 16 KB), and prefill tok/s. **UNCERTAIN**: nothing about this kernel's Metal behaviour is measured. The likely Metal-specific risks are MSL compile time for five fully-unrolled kernels and Tint's robustness clamps on the `gm_qs4` / `q4_sc` loads (2 clamped storage loads per 128 vec4 FMAs here, vs ~6 clamped *shared* loads per 32 FMAs in the prototype — the row-stationary form is far less exposed by construction).

**6.6 Bench-log rows required** (GOVERNANCE.md: every performance claim, including neutral ones): the inventory A/B, the `bench_breakdown` A/B, end-to-end prefill, decode unchanged (it must be, structurally — record it anyway), and the Mac row even if it is neutral.

---

## 7. Ranked by expected gain per engineering week

| # | Item | Gain | Cost | Gain/week |
|---|---|---|---|---|
| **1** | **The GEMM + the `_dop` ladder + NC=16** (§2–3, minus the tail fixes) | matvec 2.5×; pass 37 → ~76 tok/s | 1.0 wk (kernel is written and validated; the work is the nine edits and the test suite) | **~2.0× / wk** |
| **2** | **Prefill tail as a batched pass** (`prefillTokens` step-down + the room's `while` bound) | removes up to 15 single-token passes (1.7 s solo) or 15 network laps (up to 4 s cross-internet) per prompt | 0.15 wk | very high, and it is a **hard prerequisite** — NC=16 is a regression without it |
| **3** | **Batch the MTP prefill fill** (run the MTP layer as a 16-column pass instead of 16 single-column submits) | solo prefill 51 → ~70 tok/s (removes ~48% overhead) | 0.4 wk (reuse `_encodeLayerBatch` on `mtpLayer`; draft-only, cannot change output tokens) | **~0.35× / wk** |
| **4** | **Fuse the `g` transpose into the reduce epilogue** (`outT` flag: gate/up reduces write `[row][col]`, `silu_mul_mc` follows, `down` consumes directly) | removes the single largest transpose (17408 wide); measured ceiling 1.4 ms/pass total for *all* transposes | 0.3 wk | **~0.01× / wk — do not do this.** Listed to close it out: measured, not worth it |
| **5** | **Push the inner loop toward the roofline** (the GEMM is at 47% of 184 GB/s; the 1-column GEMV is at 100%) | up to 1.9× more on the matvec family → pass ~120 tok/s | 2–4 wk, outcome **UNCERTAIN** | ~0.2–0.4× / wk |
| **6** | **`dot4I8Packed` / int8 activations** | ~1.3–1.4× on the matvec share (repo's own GB10 microbench, `kernel-plan-3.md`) | 3+ wk, changes numerics (needs a default-off switch), Tint polyfills it on Metal so it is NVIDIA/AMD/Intel-only | ~0.1× / wk, and it breaks the wire contract |
| **7** | **Glue at 16 columns** (58 ms/pass: `dn_delta_mc` 24 ms, attention, norms) | after item 1 this is 27% of the pass; the register-resident `dn_delta_mc` spec is written and bit-identical at RG=1 | 1 wk for `dn_delta`, more for attention | ~0.1× / wk each, but it is what stands between ~76 and ~110 tok/s |

**Do items 1 + 2 together (they are one change), then 3.** That is ~1.5 engineering weeks for prefill 27 → ~65–70 tok/s end-to-end solo, which is the honest number for the MLSys row. Items 5–7 are the path to 100+ and do not fit before Oct 30.

---

## 8. Open uncertainties

- **The §4 end-to-end projection is calibrated, not measured.** The microbenchmark-to-engine factor is 1.26–1.39 depending on which of the engine's own figures you anchor on (its 4-column `matvec_b` figure agrees with the harness to 6%; its 8-column figure disagrees by 15% in the other direction, and that figure comes from skip-family timing, which the repo's own spec says under-attributes latency-hidden kernels). Gate on `bench_breakdown` (§6.5.3) before publishing a tok/s number.
- **Nothing about this kernel is measured on Metal or Safari.** The design is portability-safe by construction (8 KB shared, no subgroups, probe-gated `unpack4x`, literal-indexed shared arrays) but the Mac's unexplained shortfall is unexplained for prefill too.
- **`S = 12` for 5120×6144** measured 7% better than the S=8 in the table; **`T = 32` for 5120×17408** measured 30% better in isolation and neutral in the inventory. Both are inside run-to-run noise for the pass total; resolve with one interleaved confirm run rather than by reasoning.
- **The bit-exactness of the 27B prefill token stream is currently untested** (`test_q38_full.js` uses `forwardToken`). §6.3 adds the test; if it turns out that today's batched path already diverges from sequential on some prompt, that is a pre-existing finding this change will surface, not cause.
- **Shader compile time** for five unrolled GEMM variants on Tint/MSL is unmeasured; 93 ms for 11 pipelines on wgpu/naga is the only data point.