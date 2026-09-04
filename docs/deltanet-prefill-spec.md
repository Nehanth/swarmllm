# DeltaNet prefill: implementation specs (Sep 2 night; 38 agents, 24/32 findings verified)

# Spec: re-tiled register-resident `dn_delta_mc` (same math, same op order)

## 0. Premise and honest ceiling (verified)

- `dn_delta_mc` (`engine/qwen35.js:294-333`) is 48 WGs x 128 threads = 6144 threads (~8% occupancy on GB10), and does two global read-modify-write sweeps of S per token per column (4 x 64 KB per head-token, L2-resident, latency-bound).
- It is NOT what bounds prefill: matvec_*_b is ~91 ms of a 132.7 ms 4-col pass and ~170 ms of a ~222 ms 8-col pass; dn_delta_mc is 9.2 ms (4 col) / 14.1 ms (8 col) measured in isolation, 1.3 / 14.3 ms by skip-timing. Ceiling of this change end-to-end: ~4% of a prefill pass, ~2% of a decode token. Do it because it is ~100 lines, bit-identical (RG=1) or bit-close (RG=2), and gives 1.7-2.9x on the kernel itself; do not expect it to move tok/s much.

## 1. Design

One workgroup per value head (48 WGs, unchanged). Workgroup = RG row-groups x 128 columns, `T = 128*RG` threads. Thread `(r, j)` owns rows `[r*R, (r+1)*R)` of S column `j` (`R = 128/RG`) in a private array whose every index is a compile-time literal (codegen-unrolled in the JS template string; a WGSL `for` over a `const`/literal bound does NOT unroll on wgpu/naga and lands the array in local memory, measured 2.3x slower than today).

Per column (token), exactly today's statement order per element:
`sd = s*decay; s = sd; vhat += sd*k_i; sq += sd*q_i; kq += k_i*q_i; d = (v_j - vhat)*beta; s += k_i*d; o = (sq + d*kq)*scale`.

- q/k of the current column staged into 2 x 128 f32 workgroup arrays by row-group 0 (two barriers: one protecting the previous column's readers, one publishing).
- RG>1: per-row-group partial `vh`, `sq` go through `RG x 128 x 2` f32 workgroup memory, one barrier, then every thread sums the RG partials for its column in fixed order g=0..RG-1. `kq` is a full 128-term sequential sum from workgroup memory, redundantly per thread (same order as today).
- S loaded once before the column loop, stored once after; snapshot slots (`frame.snap`) written straight from registers per column.
- `scale = inverseSqrt(f32(dlm_dn.dState))` must stay a runtime uniform read (folding a literal 128 changes 1 ULP in ~90% of outputs).
- The early `if (h >= nVH || j >= dS) { return; }` is removed: it precedes barriers and `j` derives from `local_invocation_id`, so uniformity analysis would reject the kernel. Dispatch is exactly nVH WGs, so the guard was dead anyway.
- Bindings, bind group layout (`["ro","ro","ro","rw","rw","u","u","rw"]`), `MC`/`DN`/`Frame` uniforms, buffers: unchanged. Drop-in.

Workgroup memory: 1 KB (RG=1), 3 KB (RG=2), 5 KB (RG=4). Registers: R f32 for S + ~20 temporaries.

## 2. Complete WGSL (generator; validated compile + run on GB10)

Put this in `qwen35.js` next to `WGSL2` and splice `${dnDeltaTiledWGSL(RG)}` in place of lines 294-333 (keep the eight `dlm_*` declarations at 286-293). The generator is the WGSL; the unrolled bodies are shown as their templates because 64-128 literal statements are what the file must contain.

```js
// RG = row-groups per head (1, 2 or 4). Emits entry point `name` with workgroup_size(128*RG).
// Requires dState == 128 (assert in JS). All private-array indices are literals on purpose.
function dnDeltaTiledWGSL(RG, name = "dn_delta_mc") {
  const R = 128 / RG, T = 128 * RG;
  const rows = Array.from({ length: R }, (_, i) => i);
  const P = "dlt" + RG + "_";                       // unique var<workgroup> names inside the shared module
  const load   = rows.map((i) => `s[${i}u] = dlm_s[Sb + (r0 + ${i}u) * 128u + j];`).join("\n  ");
  const store  = rows.map((i) => `dlm_s[Sb + (r0 + ${i}u) * 128u + j] = s[${i}u];`).join("\n  ");
  const shadow = rows.map((i) => `dlm_shadow[so + (r0 + ${i}u) * 128u] = s[${i}u];`).join("\n        ");
  const loop1  = rows.map((i) => `{ let sd = s[${i}u] * decay; s[${i}u] = sd; vh += sd * ${P}k[r0 + ${i}u]; sq += sd * ${P}q[r0 + ${i}u]; }`).join("\n      ");
  const loop2  = rows.map((i) => `s[${i}u] += ${P}k[r0 + ${i}u] * d;`).join("\n      ");
  const partials = RG > 1 ? `
var<workgroup> ${P}pv: array<f32, ${RG * 128}>;
var<workgroup> ${P}pq: array<f32, ${RG * 128}>;` : "";
  const reduce = RG > 1 ? `
      ${P}pv[r * 128u + j] = vh; ${P}pq[r * 128u + j] = sq;
      workgroupBarrier();
      var vhat: f32 = 0.0; var sqt: f32 = 0.0;
      ${Array.from({ length: RG }, (_, g) => `vhat += ${P}pv[${g * 128}u + j]; sqt += ${P}pq[${g * 128}u + j];`).join("\n      ")}` : `
      let vhat = vh; let sqt = sq;`;
  return /* wgsl */ `
var<workgroup> ${P}k: array<f32, 128>;
var<workgroup> ${P}q: array<f32, 128>;${partials}
@compute @workgroup_size(${T})
fn ${name}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x; let j = lid.x % 128u; let r = lid.x / 128u; let r0 = r * ${R}u;
  let kh = h % dlm_dn.nKH;
  let kOff = kh * 128u; let vOff = h * 128u; let Sb = h * 16384u;
  let scale = inverseSqrt(f32(dlm_dn.dState));          // uniform read: keeps bit-identity with today
  let nCols = max(frame.nCols, 1u);
  let sSize = dlm_dn.nVH * 16384u;
  var s: array<f32, ${R}>;
  ${load}
  for (var col: u32 = 0u; col < nCols; col++) {
    let qo = col * dlm_mc.s0 + kOff;
    let ko = qo + dlm_dn.keyDim;
    let vo = col * dlm_mc.s0 + 2u * dlm_dn.keyDim + vOff;
    workgroupBarrier();                                  // previous column done reading k/q
    if (r == 0u) { ${P}k[j] = dlm_c[ko + j]; ${P}q[j] = dlm_c[qo + j]; }
    workgroupBarrier();
    let decay = dlm_decay[col * dlm_mc.s1 + h];
    let beta = dlm_beta[col * dlm_mc.s1 + h];
    let vj = dlm_c[vo + j];
    var vh: f32 = 0.0; var sq: f32 = 0.0;
      ${loop1}${reduce}
    var kq: f32 = 0.0;
    for (var i: u32 = 0u; i < 128u; i++) { kq += ${P}k[i] * ${P}q[i]; }   // same 128-term order as today
    let d = (vj - vhat) * beta;
      ${loop2}
    if (r == 0u) { dlm_o[col * dlm_mc.s2 + vOff + j] = (sqt + d * kq) * scale; }
    let dlSB = frame.snap & 0xffu;                       // snapshot slot base + 1 (0 = off)
    if (dlSB != 0u && dlSB + col < (frame.snap >> 8u)) {
      let so = (dlSB - 1u + col) * sSize + Sb + j;
        ${shadow}
    }
  }
  ${store}
}`;
}
```

Expanded shape for RG=2: `workgroup_size(256)`, `var s: array<f32, 64>`, 64 load / 64 loop1 / 64 loop2 / 64 shadow / 64 store statements, `dlt2_pv`/`dlt2_pq` of 256 f32 each, three barriers per column. Full source used for the measurements below: `scratch/verify_delta_tile.js` (also contains the shipped kernel verbatim as reference and the bit-compare).

## 3. Integration (qwen35.js)

1. `WGSL2` (line 7) is a plain template string; change lines 294-333 to `${dnDeltaTiledWGSL(DN_RG)}` where `const DN_RG = 2;` (module-level). Keep 286-293 (`dlm_*` bindings) as is.
2. In the constructor after `this.dims` (line 554): `if (dState !== 128) throw new Error("dn_delta_mc tiled kernel assumes dState=128");`.
3. Dispatch, line 1175: `this._dMC(p, "dn_delta_mc", M.delta, D.nVH * 128 * DN_RG, 128 * DN_RG, 1);` (`_dMC` does `ceil(threads/wg)` -> 48 WGs).
4. `G1.dn_delta_mc`, `M.delta` bind group (1076-1077), buffers, `frameBufsB` writes (1216): unchanged.
5. Limits: RG=2 (256 threads, 3 KB workgroup memory) is inside WebGPU defaults (`maxComputeInvocationsPerWorkgroup` 256, `maxComputeWorkgroupStorageSize` 16384) so `p2p.html:1052` `requestDevice` needs no change. RG=4 would need `maxComputeInvocationsPerWorkgroup`/`maxComputeWorkgroupSizeX: 512` in `requiredLimits` and was slower on GB10 anyway (below).
6. Optional decode path (`dn_delta`, lines 68-108, separate q/k/v bindings, 48x128 threads, same global-S structure): same transform applies; RG=2 `_mc` kernel at nCols=1 measured 3.3 ms vs 5.6 ms (48 layers). Not required for this change.

## 4. Dispatch shapes

| variant | workgroup_size | WGs/layer | threads/layer | rows per thread | var<workgroup> | limits needed |
|---|---|---|---|---|---|---|
| today | 128 | 48 | 6144 | 128 (in global S) | 0 | default |
| RG=1 | 128 | 48 | 6144 | 128 in registers | 1 KB | default |
| RG=2 (recommended) | 256 | 48 | 12288 | 64 | 3 KB | default |
| RG=4 | 512 | 48 | 24576 | 32 | 5 KB | maxComputeInvocationsPerWorkgroup >= 512 |

S traffic: today 4 x 64 KB per head-token (2.3 GB per 4-col pass through L2); all new variants: read once + write once per pass = 288 MB/pass regardless of nCols.

## 5. Measured (GB10, idle, 48 layers, engine bindings, synthetic data; two runs, repeatable to <0.1 ms)

| nCols | today | RG=1 | RG=2 | RG=4 |
|---|---|---|---|---|
| 1 | 5.58-5.62 ms | 4.50-4.53 (1.24x) | 3.17-3.29 (1.72x) | 3.31-3.36 (1.68x) |
| 4 | 9.16-9.23 | 5.93-6.01 (1.54x) | 4.25-4.35 (2.13x) | 4.64-4.68 (1.97x) |
| 8 | 13.92-13.99 | 7.83-7.85 (1.78x) | 5.75-5.83 (2.41x) | 6.37-6.39 (2.19x) |
| 16 | 23.64-23.71 | 11.64 (2.03x) | 8.68-8.71 (2.72x) | 9.77-9.81 (2.42x) |
| 32 | 42.95-42.96 | 19.07-19.16 (2.25x) | 14.57-14.62 (2.94x) | 16.49-16.53 (2.60x) |

Fit: today ~4.4 ms + 1.2 ms/col; RG=2 ~2.9 ms + 0.37 ms/col. The remaining fixed ~2.9 ms is 48 dispatches (~1.3 ms, measured with empty dispatches) plus one S load/store per pass with only 48 WGs in flight.

Numerics (nCols=8, snapshots on, S/out/all 8 shadow slots vs shipped kernel):
- RG=1: 0 of 786432 S, 0 of 49152 out, 0 of 6291456 shadow elements differ (bit-identical).
- RG=2: relDiff(L2) S 5.2e-8, out 1.9e-7, shadow 4.0e-8; max |diff| 1.2e-7 (partial-sum reordering only; large "max rel" values are on near-zero elements).
- RG=4: S 5.5e-8, out 2.0e-7, shadow 4.2e-8.

## 6. Expected end-to-end effect

- 4-col prefill pass (~132.7 ms): -4.9 ms (isolated) => ~1.04x. 8-col pass (~222 ms): -8.2 ms => ~1.04x. Decode token (~111 ms): -2.3 ms => ~1.02x.
- UNCERTAIN: in-engine skip-timing attributed only 1.3 ms to dn_delta_mc at 4 cols (vs 9.2 ms isolated), i.e. part of the kernel already hides behind submission bubbles; real in-engine saving is between ~1 and ~5 ms per 4-col pass. Measure with `bench_breakdown_deno.js` (and an 8-col variant) before/after.
- Prefill tok/s will remain bounded by matvec_*_b (verified); this change does not alter that conclusion.

## 7. Validation plan

1. `test_batch_q38_deno.js` prefill-vs-sequential relDiff: with RG=1 it must be exactly what it is today (bit-identical, including `dlm_shadow` slots and rollback); with RG=2 expect ~5e-8 (S) / 2e-7 (out) extra, well inside the test's 2e-3 gate and the ~1e-7 you observe today.
2. Greedy token stream on the standard prompt must match byte-for-byte (RG=1) / is expected to match (RG=2, UNCERTAIN only in the sense that no long-prompt run was done here).
3. Speculative rollback (`restoreDN`) exercised with `snapshot` on: shadow slots verified identical (RG=1) / 4e-8 (RG=2) in the harness.
4. Shader compile: check `getCompilationInfo()`; the RG=1 variant emits 128-statement bodies x 5, which compiled cleanly on wgpu/naga.

## 8. Uncertainties / portability

- UNCERTAIN: Apple (Safari/Metal via Tint or WebKit's compiler) and Chrome/Tint on other GPUs were not measured. Register-resident private arrays with literal indices are the standard approach (llama.cpp's own `gated_delta_net.wgsl` does it), but the RG=1 vs RG=2 ranking and absolute gains may differ. Keep `DN_RG` as a single knob; RG=2 is inside default limits everywhere.
- UNCERTAIN: bit-identity (RG=1) depends on both compilers making the same FMA-contraction choices for `a += b*c`; that is already the case for today's kernel across platforms, and the new kernel uses identical expression shapes.
- Not covered: `dn_conv_mc` (still a serial column loop; separable per (channel, column) as verified, but out of scope here), `dn_pre_mc` (1.1-1.2 ms/pass, fine).

Files: harness `scratch/verify_delta_tile.js`; target `engine/qwen35.js` (kernel 294-333, bind group 1076-1077, dispatch 1175, dims 545-554); device request `p2p.html:1052` (unchanged for RG<=2).

---

# Spec: Chunkwise-parallel Gated DeltaNet prefill (C = 4..16 tokens per pass) for the SwarmLLM WebGPU engine

Scope: replaces the token loop of `dn_delta_mc` (engine/qwen35.js:294-333, 48 WGs x 128 threads, S in a read_write storage buffer) with a single-dispatch chunk kernel. Everything else in the DeltaNet block (conv, L2 norm, beta/decay, gated RMSNorm) is per-token and stays. Only verified findings are used; items not established by measurement or source are marked UNCERTAIN.

## 0. Premise and what this buys (read first)

Measured on the GB10 (scratchpad/micro.txt, breakdown8.txt; 48 DeltaNet layers, 64 KB f32 state per head, nVH=48, nKH=16, dState=128):

| item | 1 col | 4 cols | 8 cols | 16 cols |
|---|---|---|---|---|
| current `dn_delta_mc` | 5.6 ms | 9.2 ms | 14.1 ms | 24.0 ms (standalone) |
| register-resident sequential (bit-identical, unmeasured in-engine) | 4.3 | 4.8 | 5.4 | 6.8 |
| whole batched pass | 113 | 131 | 210-222 | 459 |
| `matvec_*_b` family | - | 108 | 170 | 401 |

Fixed floor inside any variant: ~1.3 ms for 48 dispatches + ~3 ms to load/store 3 MB of S per layer once (warm same-S run 3.67 ms). The chunk form has the same FMA count as the sequential scan (~3*128*C + C^2 per thread); its only algorithmic gain is turning the token-serial dependency into C independent 128-long reductions plus a C-step triangular solve, and touching S once per pass.

Consequence: this kernel can take `dn_delta_mc` from 14 ms to an estimated 4.5-6 ms at 8 columns (UNCERTAIN: estimate, not measured; the register-resident sequential already gets 5.4 ms bit-identically). End-to-end prefill gain is capped at ~1.04-1.07x because `matvec_*_b` is ~77% of the 8-col pass. Build this only if (a) you intend to widen passes to 16+ columns where the sequential per-column chain (0.16 ms/col/pass) starts to matter, or (b) as groundwork for a 64-token chunk path. Otherwise the register-resident sequential rewrite (bit-identical) is the better first step.

## 1. Equations (engine layout: S is K x V, 128 x 128, thread j owns column S[:, j])

Per value head h, tokens r = 0..C-1 in the pass (0-indexed), S_0 = state after the previous pass.

Inputs per token (all f32, produced today by `dn_pre_mc`):
- q_r, k_r in R^128: L2-normalized, `x * rsqrt(sum(x^2) + 1e-6)`; k from key head kh = h / (nVH/nKH) = h/3 (GQA repeat_interleave).
- v_r in R^128, beta_r = sigmoid(b_r) in (0,1), alpha_r = exp(g_r) in (0,1], g_r = -exp(A_log)*softplus(a_r + dt_bias) <= 0. `dlm_decay` already holds alpha_r (exp applied), which is exactly what the running-product formulation needs.
- scale = 128^-0.5, applied to the output (today: `o = (sq + d*kq)*scale`).

Sequential (what `dn_delta_mc` computes, per token):
S <- alpha_r S; vhat = S^T k_r; d_r = beta_r (v_r - vhat); S <- S + k_r d_r^T; o_r = scale * S^T q_r.

Cumulative decays, 0-indexed:
- gamma_r = prod_{m=0..r} alpha_m (gamma_{-1} = 1).
- Lam[r][i] = prod_{m=i+1..r} alpha_m for i <= r (Lam[r][r] = 1), 0 for i > r. Note gamma_r = alpha_0 * Lam[r][0].
- lamC[r] = Lam[C-1][r]; gamma_C = gamma_{C-1}.

Chunk identity (verified vs fla naive_chunk_gated_delta_rule, HF torch_chunk_gated_delta_rule, arXiv 2412.06464 Sec 3.3, and a float64 check at scratchpad/verify.py: O err 4e-15, S err 8e-17):

(E1) A[r][i] = beta_r * Lam[r][i] * (k_r . k_i), i < r (strictly lower). Sign is I PLUS A.
(E2) R[r] = beta_r v_r - beta_r gamma_r (S_0^T k_r)  (C x 128; RHS with S_0 folded in)
(E3) (I + A) D = R, unit-lower forward substitution: D[r] = R[r] - sum_{i<r} A[r][i] D[i]
(E4) P[r][i] = Lam[r][i] * (q_r . k_i), i <= r (diagonal INCLUDED: o_r reads S_r after its own update)
(E5) o_r = scale * ( gamma_r (S_0^T q_r) + sum_{i<=r} P[r][i] D[i] )
(E6) S_C = gamma_{C-1} S_0 + sum_r lamC[r] k_r D[r]^T
(E7) snapshot S_r (for dlm_shadow) = gamma_r S_0 + sum_{i<=r} Lam[r][i] k_i D[i]^T

Everything is separable per V-column j: (E2), (E3), (E5), (E6), (E7) act on column j of S/V/D/O independently. Only (E1)/(E4) are cross-token dots over the K dimension (S-independent).

Stability: ||k||=1, beta<1 gives |A[r][i]| < 1 and |T[r][i]| <= beta_r < 1; fp32 forward substitution at C=16 stays within ~4e-7 of fp64 across 3000 adversarial trials. No pivoting; never form T explicitly.

## 2. Numerical conventions (required for bit-closeness to the sequential path)

1. Decays as running products of the per-token alpha_r read from `dlm_decay`, never exp(G_r - G_i) of a cumsum: cumsum in f32 costs ~(r-i)*|G|*6e-8 absolute -> 1e-6..1e-5 relative in the decay, and WGSL exp() is allowed 3+2|x| ULP error. Running products reuse the exact alpha_r the sequential kernel multiplies by. Only i <= r products are ever formed, so no inf*0 hazard and no select-masking needed.
2. Underflow: gamma_r can reach denormal/0 for G < -87/-103; sequential has identical behavior (repeated S*alpha). Backends may flush denormals; same on both paths.
3. scale = `inverseSqrt(f32(dlm_dn.dState))` from the uniform, NOT `inverseSqrt(128.0)`: the constant folds to exact 1/sqrt(128) and differs by 1 ULP from the runtime rsqrt, which alone perturbs 45k of 49k outputs at 1e-7 (measured, finding 21).
4. beta, alpha, L2 norm: reuse `dn_pre_mc` output unchanged, so per-token scalars are bit-identical across paths.
5. Accumulation order inside the 128-long dots differs from the sequential kernel regardless (there it is interleaved with the decay multiply); expect relDiff ~1e-7..1e-6 on S and o, not 0. The HF fp32 chunk reference vs its recurrent path is the right yardstick (~1e-6, UNCERTAIN exact figure).
6. Denormal/flush and fma contraction: f32 throughout; do not use f16 for S (rejected: 4.9e-4 per rounding, incompatible with state continuity).

## 3. Kernel sequence per DeltaNet layer per prefill pass (C = tokens in pass)

Dims: dim 5120, convDim 10240 (= 2*2048 q/k + 6144 v), nKH=16, nVH=48, dState=128, S buffer `dlm_s` = 48 x 128 x 128 f32 row-major S[h][i][j] (3 MB/layer), shadow `dlm_shadow` slots (same shape per slot).

| # | kernel | grid | reads | writes | change |
|---|---|---|---|---|---|
| K0 | `dn_conv` (2-D) | (10240 channels, C columns) | conv input [C][10240] + convState[3][10240] | conv out [C][10240], new convState | rewrite as 2-D dispatch, bit-identical math (conv is a 4-tap FIR, not a recurrence) |
| K1 | `dn_pre_mc` | as today (one WG per column) | conv out, gate projections | q,k L2-normed [C][16][128]; beta, alpha [C][48] | unchanged (1.1-1.2 ms/pass) |
| K2 | `dn_chunk_mc` (NEW) | 48 WGs x 128 threads (variant B: 256 threads) | q,k,v,beta,alpha for C tokens; S | S; o [C][48][128]; shadow slots | replaces dn_delta_mc |
| K3 | gated RMSNorm (`dn_gatenorm_mc` or equivalent) | as today | o, gate z | layer output | unchanged; y = norm(o)*w*silu(z) |

Tail passes with nCols not in the generated set: keep the existing `dn_delta_mc` pipeline as fallback (the engine already has a `_b4` twin pattern for 4-col fallback). Generate K2 per C in {4, 8, 16}.

Shape/work map for K2 per head (C=4/8/16):
- Workgroup memory: Ks[C][129] + Qs[C][129] f32 = 4.1/8.3/16.5 KB (pitch 129 avoids the 16-way bank conflict of a 128-float pitch in S1); A[C][C], P[C][C] = 128 B / 512 B / 2 KB; alpha_s, beta_s, gamma_s, lamC = 4*C*4 B. Totals ~4.3 / ~8.9 / ~18.7 KB. Default `maxComputeWorkgroupStorageSize` is 16384: C=16 needs `requiredLimits.maxComputeWorkgroupStorageSize >= 32768` (GB10 reports 49152; Apple exposes >= 32 KB) or Q read from storage (uniform-address broadcast reads) which drops C=16 to ~10.4 KB.
- Registers per thread: s[128] + R[C] + D[C] + Oi[C] (+ e[C]) f32, all with codegen-literal indices (a dynamically indexed `var<private>` array was measured 2.3x SLOWER than today on naga/NVIDIA; codegen-unrolled was 1.85-2.6x faster). At C=16: ~180 live f32 + temps, under the 255 cap but occupancy-hostile; use variant B for C=16.
- Dots in S1: C^2 total (C(C-1)/2 for A + C(C+1)/2 for P) = 16/64/256 length-128 dots -> at most 2 per thread.
- FMAs per thread: S3 2*C*128, S4 C(C-1)/2, S5 C(C+1)/2, S6 C*128, plus C*128 per snapshot token. Total ~3*C*128 + C^2 (+snapshots).
- Global traffic: S read once + written once per head per pass (128 KB), k/q rows C*2*512 B, v/o C*512 B per column.

Barriers: 3 (after staging; after Lam/gamma; after S1). No barrier in S2-S6. WGSL uniformity: no early `return` before a barrier (the existing `if (h >= nVH || j >= dS) { return; }` would be rejected); dispatch exactly nVH WGs.

## 4. WGSL sketches

Index helpers (`qIdx`, `kIdx`, `vIdx`, `betaIdx`, `alphaIdx`, `oIdx`, `shadowBase`) are UNCERTAIN: copy the exact MC column-stride expressions from the current `dn_delta_mc` and `dn_pre_mc` bindings (qwen35.js:294-333, bind group at :1076). Uniform struct name `dlm_dn` with field `dState` is taken from the existing kernel; `nCols`/`snap` fields UNCERTAIN.

### 4.1 K2 variant A: `dn_chunk_mc`, 128 threads, column-per-thread (generated by a JS template; `C`, `PITCH`, and every `s[<literal>]` line are emitted at codegen time)

```wgsl
// ---- codegen constants ----
const C     : u32 = 8u;      // tokens per pass, generated per variant (4/8/16)
const DS    : u32 = 128u;    // dState literal for loop bounds (scale still comes from the uniform!)
const PITCH : u32 = 129u;    // padded row pitch, kills bank conflicts in S1

// bindings: same group/layout as dn_delta_mc (UNCERTAIN: fill from qwen35.js)
@group(0) @binding(0) var<storage, read>       dn_q     : array<f32>;  // L2-normed q, [col][kh][128]
@group(0) @binding(1) var<storage, read>       dn_k     : array<f32>;  // L2-normed k, [col][kh][128]
@group(0) @binding(2) var<storage, read>       dn_v     : array<f32>;  // [col][h][128]
@group(0) @binding(3) var<storage, read>       dn_beta  : array<f32>;  // [col][h]
@group(0) @binding(4) var<storage, read>       dlm_decay: array<f32>;  // alpha = exp(g), [col][h]
@group(0) @binding(5) var<storage, read_write> dlm_s    : array<f32>;  // S[h][i][j], 48*128*128
@group(0) @binding(6) var<storage, read_write> dn_o     : array<f32>;  // [col][h][128]
@group(0) @binding(7) var<storage, read_write> dlm_shadow : array<f32>; // snapshot slots
@group(0) @binding(8) var<uniform>             dlm_dn   : DnParams;    // .dState, .nVH, .nKH, .snapMask (UNCERTAIN)

var<workgroup> Ks    : array<f32, C * PITCH>;
var<workgroup> Qs    : array<f32, C * PITCH>;
var<workgroup> A     : array<f32, C * C>;   // A[r*C+i], i<r
var<workgroup> P     : array<f32, C * C>;   // P[r*C+i], i<=r
var<workgroup> Lam   : array<f32, C * C>;   // Lam[r*C+i], i<=r
var<workgroup> alpha_s : array<f32, C>;
var<workgroup> beta_s  : array<f32, C>;
var<workgroup> gamma_s : array<f32, C>;
var<workgroup> lamC    : array<f32, C>;

fn dotK(r: u32, i: u32) -> f32 {          // k_r . k_i  (workgroup memory, dynamic index is fine)
  var acc = 0.0;
  for (var m = 0u; m < DS; m++) { acc = fma(Ks[r*PITCH + m], Ks[i*PITCH + m], acc); }
  return acc;
}
fn dotQK(r: u32, i: u32) -> f32 {         // q_r . k_i
  var acc = 0.0;
  for (var m = 0u; m < DS; m++) { acc = fma(Qs[r*PITCH + m], Ks[i*PITCH + m], acc); }
  return acc;
}

@compute @workgroup_size(128)
fn dn_chunk_mc(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h  = wid.x;                        // value head 0..47 (dispatch is exactly nVH WGs; no early return)
  let j  = lid.x;                        // V column 0..127
  let kh = h / (dlm_dn.nVH / dlm_dn.nKH); // key head (GQA)
  let scale = inverseSqrt(f32(dlm_dn.dState)); // uniform -> bit-identical to dn_delta_mc

  // ---- S0: stage K, Q rows (thread j loads element j of each row: coalesced) and per-token scalars
  for (var r = 0u; r < C; r++) {
    Ks[r*PITCH + j] = dn_k[kIdx(r, kh, j)];
    Qs[r*PITCH + j] = dn_q[qIdx(r, kh, j)];
  }
  if (j < C) { alpha_s[j] = dlm_decay[alphaIdx(j, h)]; beta_s[j] = dn_beta[betaIdx(j, h)]; }
  workgroupBarrier();

  // ---- Lam / gamma by running products (thread r < C owns row r). No exp(), no cumsum.
  if (j < C) {
    let r = j;
    var p = 1.0;
    Lam[r*C + r] = 1.0;
    var i = r;
    while (i > 0u) { p = p * alpha_s[i]; i = i - 1u; Lam[r*C + i] = p; }   // Lam[r][i] = alpha_{i+1}..alpha_r
    gamma_s[r] = p * alpha_s[0];                                             // gamma_r = alpha_0..alpha_r
  }
  workgroupBarrier();
  if (j < C) { lamC[j] = Lam[(C-1u)*C + j]; }   // may be folded into the block above by thread C-1

  // ---- S1: C*C pairwise dots, <=2 per thread. Grid cell (r,i): i<r -> A[r][i]; i>=r -> P[i][r].
  for (var p = j; p < C*C; p += 128u) {
    let r = p / C; let i = p % C;
    if (i < r) {
      A[r*C + i] = beta_s[r] * Lam[r*C + i] * dotK(r, i);
    } else {
      P[i*C + r] = Lam[i*C + r] * dotQK(i, r);   // P[i][r], r<=i, diagonal included
    }
  }
  workgroupBarrier();

  // ---- S2: load own column of S_0 into registers (codegen emits 128 literal-index statements)
  let sBase = (h * DS + 0u) * DS + j;
  var s0_0   = dlm_s[sBase + 0u*DS];
  var s0_1   = dlm_s[sBase + 1u*DS];
  // ... codegen ... s0_127 = dlm_s[sBase + 127u*DS];

  // ---- S3: R[r] = beta_r v_r[j] - beta_r gamma_r (k_r . s0);  Oi[r] = gamma_r (q_r . s0)
  // 2*C independent accumulators; Ks/Qs reads are uniform-address broadcasts (no bank conflicts).
  // codegen: for each r in 0..C-1:
  //   var ak_r = fma(Ks[r*PITCH+0u], s0_0, 0.0); ak_r = fma(Ks[r*PITCH+1u], s0_1, ak_r); ... 128 terms
  //   var aq_r = fma(Qs[r*PITCH+0u], s0_0, 0.0); ... 128 terms
  //   let R_r  = beta_s[r] * dn_v[vIdx(r, h, j)] - beta_s[r] * gamma_s[r] * ak_r;
  //   let Oi_r = gamma_s[r] * aq_r;

  // ---- S4: unit-lower forward substitution (I + A) D = R, in registers, no barrier
  // codegen: D_0 = R_0;
  //          D_1 = R_1 - A[1*C+0]*D_0;
  //          D_2 = R_2 - A[2*C+0]*D_0 - A[2*C+1]*D_1;   ... C(C-1)/2 FMAs total

  // ---- S5: o_r[j] = scale * (Oi_r + sum_{i<=r} P[r][i] D_i)
  // codegen: dn_o[oIdx(r, h, j)] = scale * (Oi_r + P[r*C+0]*D_0 + ... + P[r*C+r]*D_r);

  // ---- S7 (optional, per snapshot token r flagged in dlm_dn.snapMask; semantics UNCERTAIN):
  // codegen per flagged r and per row i:  dlm_shadow[slotBase(r) + i*DS + j]
  //     = gamma_s[r]*s0_i + Lam[r*C+0]*Ks[0*PITCH+i]*D_0 + ... + Lam[r*C+r]*Ks[r*PITCH+i]*D_r;
  // (S_0 is still intact in registers here; do this BEFORE S6.)

  // ---- S6: S_C = gamma_{C-1} S_0 + sum_r lamC[r] k_r D_r^T ; write once
  // codegen: let e_r = lamC[r] * D_r  (C scalars)
  //          for each row i: dlm_s[sBase + i*DS] = fma(Ks[0*PITCH+i], e_0, fma(Ks[1*PITCH+i], e_1, ... gamma_s[C-1u]*s0_i));
}
```

Notes on the template:
- The `s0_i` scalars can equivalently be `var s: array<f32,128>` indexed ONLY with literal constants; both measured equal. Never index it with a loop variable.
- S3 fully unrolled over r and i is 2*C*128 lines (4096 at C=16). Compile time on Tint/naga UNCERTAIN; if it is a problem, keep the r loop dynamic and hold R/Oi in `var<workgroup> Rw[C][128]` (C*512 B, +2 KB at C=4 .. +8 KB at C=16) instead of private arrays.
- With Q read from storage instead of Qs (to fit 16 KB at C=16), S1's P dots read `dn_q[qIdx(i,kh,m)]` with per-thread different rows: uncoalesced but only 136 dots; S3's reads are uniform-address per WG.

### 4.2 K2 variant B: 256 threads, 2 row-groups (P=2), 64 S rows per thread

Same algebra; thread (rg = lid/128, j = lid%128) owns S rows rg*64..rg*64+63 of column j (64 registers). Differences:
- S3 produces partial (ak_r, aq_r) over its 64 rows; exchange through `var<workgroup> part : array<f32, 2*C*128>` (rg=1 writes 2C values per column, barrier, rg=0 adds and writes the total back, barrier, rg=1 reads the total): 4/8/16 KB at C=4/8/16, 2 extra barriers. Both threads of a column then run S4/S5 redundantly (S5 store done by rg==0 only); S6/S7 each updates its own 64 rows.
- Workgroup memory: Ks+Qs 8.3 KB + part 8 KB + small = ~16.6 KB at C=8: just over the default 16384. Options: Qs pitch 128 (saves 32 B... insufficient), Q in storage (~12.3 KB, fits), or raise the limit. Within default limits, 256 threads/WG and maxComputeWorkgroupSizeX=256 are OK; P=4 (512 threads) needs `maxComputeInvocationsPerWorkgroup >= 512` (GB10 1024, Apple 1024; request with fallback).
- Expected: the measured 4-row-group sequential variant (V3) beat 1 and 2 row-groups (1.85x vs 1.14x/1.23x) because per-thread register pressure drops; the same should hold here (UNCERTAIN for the chunk kernel; measure A vs B vs P=4).

### 4.3 K0: 2-D causal conv (bit-identical, drop-in for `dn_conv_mc`)

```wgsl
// one thread per (channel c, column t). 4-tap depthwise FIR with 3 carried inputs. Tap order and
// weight/state layout: COPY from dn_conv_mc verbatim (UNCERTAIN here).
@compute @workgroup_size(256)
fn dn_conv2d(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x; let t = gid.y;                 // t < nCols; dispatch ceil(convDim/256) x nCols
  if (c >= dlm_dn.convDim) { return; }          // no barriers in this kernel, early return is fine
  // x_{t-3..t}: from the batch buffer for t-k >= 0, else from convState[3 + (t-k)]
  let x0 = tap(c, t, 3u); let x1 = tap(c, t, 2u); let x2 = tap(c, t, 1u); let x3 = tap(c, t, 0u);
  let y = w0(c)*x0 + w1(c)*x1 + w2(c)*x2 + w3(c)*x3;   // SAME association order as dn_conv_mc
  convOut[t*convDim + c] = y / (1.0 + exp(-y));         // silu, same expression as today
  // new state = last 3 inputs: columns nCols-3..nCols-1 write convState[k][c]
  if (t + 3u >= dlm_dn.nCols) { convState[(t + 3u - dlm_dn.nCols) * convDim + c] = x3; }
}
```
(For nCols < 3 the untouched state entries must shift from the old state; simplest: have every thread with t == nCols-1 write all three entries from its own x1,x2,x3 window.)

## 5. Numerical validation plan vs the sequential recurrence

Harnesses to extend: scratchpad/dnbench.js (48-layer synthetic Deno WebGPU bench with V0 = verbatim `dn_delta_mc`), scratchpad/verify_delta_reg2.js / reg3.js (bit-compare of S, out, 8 shadow slots against the shipped kernel), scratchpad/verify.py (float64 chunk-vs-sequential check of E1-E6), engine/test_batch_q38_deno.js (prefill-vs-sequential, threshold rel < 2e-3), ref_q38.mjs (greedy reference).

1. Float64 oracle (CPU, per head): sequential recurrence and E1-E7 both in f64 on the SAME inputs; require |diff| < 1e-12 (already demonstrated: 4e-15 / 8e-17). Re-run for each generated C and for 0-indexed gamma to catch the gamma_r = alpha_0*Lam[r][0] off-by-one.
2. Kernel vs shipped `dn_delta_mc` (Deno WebGPU, 48 layers, nCols in {4,8,16}): inputs drawn to match the model: q,k L2-normalized, beta = sigmoid(N(0,1)), alpha = exp(-exp(A_log)*softplus(N(0,1)+1)) with A_log ~ log U(0.01,16) (this yields alpha across (1e-40, 1), including underflow), S_0 = state after 64 sequential warm-up tokens (non-zero, realistic scale). Metrics: relDiff (L2) of S_C, o, and each shadow slot vs the shipped kernel and vs the f64 oracle. Acceptance (UNCERTAIN thresholds; set from the first run against the f64 oracle): S_C relDiff <= 1e-6, o <= 1e-5, and the chunk kernel must be at least as close to the f64 oracle as the shipped kernel is. Expected ~1e-7..1e-6 from f32 reordering only.
3. Adversarial cases: (a) alpha -> 0 for a middle token (gamma underflow; both paths must agree on which entries flush); (b) beta -> 1 - 1e-7; (c) all k identical (A entries = beta*Lam; T entries bounded by beta < 1); (d) k orthogonal (A = 0, D = R); (e) S_0 = 0 (first pass) and S_0 huge (long context). Compare vs oracle in each.
4. Determinism: two runs of the chunk kernel on the same inputs must be bit-identical (no atomics, fixed reduction order); shadow slots must equal an independent S_r computed by running the shipped kernel for r+1 tokens (relDiff <= 1e-6).
5. State continuity (the constraint that matters for decode): prefill N tokens with the chunk kernel, then decode 32 tokens with the unchanged single-token `dn_delta` kernel; compare per-step logits relDiff and the greedy token stream against the all-sequential run on >= 3 prompts of 100-500 tokens (ref_q38.mjs). Pass criterion: identical greedy stream; report logits relDiff (expect ~1e-6..1e-5 UNCERTAIN).
6. Integration: test_batch_q38_deno.js unchanged (2e-3 gate) plus a new tighter assertion at the level established in step 2; run both nCols=4 and 8 and the tail-pass path (prompt length not a multiple of C).
7. Performance attribution: dnbench.js-style microbench on an IDLE GPU (a concurrent GPU job doubles absolute numbers), NC=4/8/16, NL=48, 10 reps; then engine/bench_breakdown_deno.js with batchCols 8 (skip-family timing under-attributes latency-hidden kernels, so also keep the direct microbench).

## 6. Expected speedup

Kernel level (48 layers, per pass), measured baselines and estimates:
- Current `dn_delta_mc`: 9.2 ms @4 cols, 14.1 @8, 24.0 @16 (4.4 ms fixed + ~1.2 ms/col).
- Register-resident sequential (bit-identical, measured standalone): 4.8 / 5.4 / 6.8 ms.
- Chunk kernel: floor ~4.3 ms (1.3 dispatch + ~3 S load/store) plus S1 (<=2 dots/thread) and 3*C*128 ILP FMAs. Estimate 4.5-5.5 ms @8 and 5-6 ms @16 (UNCERTAIN, unmeasured). Speedup vs current: ~2.6-3x @8, ~4x @16; vs the register-resident sequential: ~1.0-1.2x, growing with C.
End-to-end: an 8-col pass is 210-222 ms with matvec_b at ~170 ms; the recurrence rewrite saves at most ~9 ms -> <= 1.04x prefill tok/s. At 16 cols the pass is 459 ms (matvec_b 401), saving ~18 ms -> ~1.04x. The recurrence is not what bounds prefill on this GPU; the 4-vs-8 col near-parity is matvec_b's superlinear column scaling plus leftover single-token passes in the host loop.

## 7. Implementation checklist / risks

- Codegen: emit the kernel from a JS template per C with literal indices for every private-array access; verify no `local memory` spill by timing against the register-resident sequential (a 2x slowdown = spilled).
- Limits: request `maxComputeWorkgroupStorageSize` (32768) and, for P>=4, `maxComputeInvocationsPerWorkgroup`/`maxComputeWorkgroupSizeX` in `requestDevice`; pick variant A/B/C=8 fallback from `device.limits` at pipeline creation. SwarmLLM's requestDevice currently raises only buffer limits.
- Uniformity: remove the early return; guard nothing that precedes a barrier on `local_invocation_id`.
- Pitch 129 for Ks/Qs; scale from the uniform; alpha from `dlm_decay`, not recomputed with exp().
- Snapshots (S7) must be computed before S6 overwrites the S_0 registers; snapshot slot semantics (`frame.snap`, slot index per column) UNCERTAIN -- mirror the shipped kernel's stores exactly and verify with the shadow bit-compare harness.
- GQA option: k_r.k_i and q_r.k_i are shared by the 3 value heads of a key head; only beta/Lam scaling differ. Not worth a cross-WG pass at C<=16 (S1 is <=2 dots/thread).
- Keep `dn_delta_mc` as the fallback pipeline for nCols not in the generated set and for decode (nCols=1).
- Full-attention layers and `matvec_*_b` are untouched; the real prefill bottleneck remains matvec_b (target that next: x-column staging in workgroup memory, flat ~55-60 ms/pass up to 16 cols).