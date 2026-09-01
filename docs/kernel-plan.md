# SwarmLLM Kernel-Optimization Week — Implementation Plan

**Baseline & ceiling math (calibrates every estimate below):** decode is weight-bandwidth-bound: ~15 GB of Q4_0/Q8_0 weights stream per token. At 2.5 tok/s we sustain ~38 GB/s effective — ~10% of a Max-class M-chip's ~400 GB/s. The hard ceiling is ~25 tok/s (400/15.2); browser stacks demonstrably reach 40-70% of roofline (WebLLM: 41.1 tok/s on 8B = 71% of native on M3 Max), so **10-13 tok/s is the realistic landing zone on Max-class; 15 is a stretch (needs 56% of peak); base-M/iPhone are capped at ~4-7 tok/s regardless of kernels; GB10 (~273 GB/s) lands 10-13.** Current bottleneck is NOT bandwidth — it's the maximally-uncoalesced 1-thread-per-row kernel (adjacent threads read addresses one full weight row apart, dIn bytes for Q8/dIn/2 for Q4) plus ~67 submits/token and a ~600KB/token (248320-vocab is ~1MB f32 — read back in full) logits sync.

**Global order (gain/effort):** A0 → A1 → C1 → C2 → A2 → B1 → B2 → A3 → A4.

---

## A. Kernel work (this week)

### A0. Shape-constant prelude + frozen bind-group layout (half day — pure enabler)
- **WHAT (JS):** generate a WGSL const prelude per model (`const D = 5120;`, `FFN_DIM = 17408;`, `NB`, `ROW_WORDS`, etc., emitted as AbstractInt) injected at pipeline creation; runtime-variable dims via one small uniform. Freeze one bind-group layout for the whole matvec family (`y@0, x@1, scales@2, qs@3, args@4`) so every variant below swaps only the pipeline object, never bind groups. Kill the `q8_shape` uniform loads in loop bounds.
- **WHY:** makes A1-A4 A/B-able in hours instead of days; const-folds loop bounds. Near-zero direct tok/s gain (weight bytes unchanged).
- **EXPECTED cumulative:** 2.5 tok/s (unchanged).
- **RISK:** none — core WGSL, all three runtimes.
- **VALIDATE:** golden tests pass unchanged; add a **non-zero output checksum** to the bench harness now ("a skipped dispatch always looks like a fast one" — a mis-bound bind group silently no-ops in WebGPU; zero-tvm hit this 3x).

### A1. Cooperative workgroup-per-row matvec — the single biggest lever (2 days)
- **WHAT (WGSL):** replace 1-thread-per-row with llama.cpp's shipped WebGPU shape: `@workgroup_size(256)`, **OUTPUTS_PER_WG = 4 rows per workgroup**, THREADS_PER_BLOCK = 4 per 32-elem block. Strided-coalesced loop: `for (block = tid/4; block < NB; block += 64)`, `thread_within_block = tid % 4`, each thread handling 8 elements — consecutive threads read **consecutive u32 words of the same row**. Each thread loads its 8 activations into a register array `x_block[8]` ONCE and reuses across all 4 rows (`acc[4]` per thread). Reduction: **portable shared-memory halving tree ONLY** as baseline — `var<workgroup> partial_sums: array<f32, 4*256>` (4 KB), layout `[row][thread]`, `stride=128; while(stride>0){ if(tid<stride) add; workgroupBarrier(); stride/=2 }` with the barrier OUTSIDE the `if` (uniform control flow, or Tint/Naga reject it). Same structure for Q4 (nibbles, -8) and Q8.
- **WHAT (JS dispatch):** `dispatchWorkgroups` count = `ceil(dOut/4)`, **flattened to 2D** (`wg_id.y*num_wg.x + wg_id.x` in-shader) for the LM head: 248320 rows → 62080 WGs, which fits per-dim but keep the 2D fold anyway for safety/uniformity. Buffer layouts unchanged — your repacked qs/scales already beat llama.cpp's spliced GGUF layout; keep them.
- **WHY:** converts fully-uncoalesced, latency-bound access (5120-17408 total threads, each a 5120-long serial loop) into coalesced 256-wide streaming with 4x activation amortization. This is verbatim what llama.cpp ggml-webgpu, web-llm/TVM-dlight (TS=1,TR=64), and zero-tvm all ship for decode.
- **EXPECTED cumulative:** **~6-8 tok/s** on Max-class (kernel-level 3-5x; e2e capped below that because ~67 submits/token + logits readback still cost tens of ms of the now-~130ms token). ~90-120 GB/s = 24-30% of peak — well under the 25 tok/s ceiling, headroom remains.
- **RISK/caveats:** core WGSL only (256 invocations, 4KB workgroup storage — inside guaranteed limits everywhere). **Do NOT use subgroups here** — absent from shipping Safari 26 (landed only in Safari TP 249). Do not shrink total WG count below current dispatch sizes on small matvecs (zero-tvm measured 15-19% loss from halving WG count on M2 Pro).
- **VALIDATE:** token-exact golden vs llama.cpp, greedy, ≥64 tokens, per-quant-type (Q4 and Q8 layers). **Expect possible near-tie token flips from reduction reorder** — add a logit-space check (max |Δlogit| < 1e-3 vs old kernel on stored activations) to distinguish benign drift from real bugs. Paired A/B benches only (alternate arms within a round; thermal drift is ~20%).

### A2. Word/vector loads + unroll + scale hoist inside A1's kernel (1 day)
- **WHAT (WGSL):** (a) bind activations as `array<vec4<f32>>`, unpack each weight u32's 4 values into a `vec4<f32>` once, accumulate with `sum += dot(w4, x4)` — 1 dot instead of 4 scalar FMAs, 8 fewer `extractBits` per word (use shifts+mask, not extractBits — avoids Naga polyfill paths); (b) fully unroll the per-thread word loop (4-8 words/iter — NOT the whole 5120 dim, Safari compile-time blowup); (c) hoist the per-block scale load out of the 8-word inner loop; (d) for Q4, consider repack-time **lo/hi nibble-plane splitting** (two u32 streams) so decode is mask+subtract with no shifts (Metal "yl-trick" analogue). dIn%4==0 holds (5120, 17408).
- **WHY:** post-coalescing the kernel is partially ALU/ILP-limited; this keeps ALU off the critical path.
- **EXPECTED cumulative:** **~8-10 tok/s** (with C1/C2 landed; +10-20% on kernel time — low end on desktop once bandwidth-bound, upper end on iPhone).
- **RISK:** core WGSL everywhere. `unpack4xI8` needs the `packed_4x8_integer_dot_product` language feature — feature-detect via `navigator.gpu.wgslLanguageFeatures` or just use manual shift/mask (identical codegen).
- **VALIDATE:** same golden + logit-diff harness; paired A/B per variant (this is exactly what A0's variant machinery is for).

### A3. TS/TR autotune knobs (half day, after C lands)
- **WHAT:** expose WG_SIZE {128, 256} × OUTPUTS_PER_WG {2, 4, 8} as generated-shader parameters; ~10 timed dispatches at startup per device (wall-clock around `onSubmittedWorkDone` — **never** request `timestamp-query`: requesting the feature alone cost zero-tvm ~3x decode); pick per device. MLC never tuned WebGPU (TS=1/TR=64 is their generic fallback), so headroom exists.
- **WHY:** Apple sometimes prefers 128 threads × more rows; GB10 differs.
- **EXPECTED cumulative:** **~10-12 tok/s** Max-class (1.1-1.3x realistic; approaches 1x as we near achieved-bandwidth ceiling).
- **RISK:** none (core). Keep the occupancy floor rule: never reduce total threads on already-small dispatches.
- **VALIDATE:** goldens must pass for every (WG_SIZE, OUTPUTS) combo — run the matrix in CI once.

### A4. MMVQ dp4a variant — GB10/Deno only (1 day, parallelizable)
- **WHAT:** port llama.cpp's MMVQ: a `quantize_q8` pre-pass (activations → q8_1 blocks: d=amax/127, `pack4xI8`, s-term via `dot4I8Packed(q, 0x01010101u)`), then matvec inner loop = `dot4I8Packed` pairs; Q4_0 via lo/hi nibble planes (`qs&0x0F0F0F0F`, `(qs>>4)&0x0F0F0F0F`) with the -8 folded algebraically: `acc += f32(row_sum)*(d_a*d_b) - 8.0*d_a*s_b/THREADS_PER_BLOCK`. **Gate on feature AND `adapter.info.vendor` ∈ {nvidia, amd, intel}** — never Apple.
- **WHY:** GB10 kernel is ALU/latency-bound (~48 GB/s of 273); native dp4a via wgpu/Vulkan replaces 32 extractBits+FMAs per block with 2-8 instructions.
- **EXPECTED cumulative:** **GB10: 12-15 tok/s** (roofline ~18-19). No change on Macs.
- **RISK:** Apple Metal *emulates* dp4a (Safari 26 compiles it but it's 4 scalar mul-adds — llama.cpp explicitly excludes apple vendor); wgpu needs ≥v25/26 for native OpSDot — feature-detect, keep f32 path as fallback.
- **VALIDATE:** goldens on GB10 under Deno; note activation quantization changes logits more than reduction reorder — use the logit-tolerance harness plus token-match on a longer (256-token) run.

---

## B. Prefill batching

### B1. NUM_COLS column loop in the cooperative matvec (half day — cheap first step)
- **WHAT:** extend A1's kernel with a compile-time NUM_COLS (≤4) loop: reuse each loaded weight word across up to 4 token columns (`x[col*D + …]`, `y[col*dOut + r]`); attention becomes causal over the small batch; KV appends M entries per pass. JS: activation buffer becomes `[M × dIn]`.
- **WHY:** prefill currently re-streams ~15 GB per prompt token; weight reuse is near-linear for small M since prefill is weight-bandwidth-bound.
- **EXPECTED:** up to ~4x prefill (6s → ~1.5-2s short prompt) for minimal kernel work. Zero effect on decode.
- **VALIDATE:** goldens on the *first generated token* after a batched prefill (KV correctness is the whole risk) vs token-at-a-time prefill and vs llama.cpp.

### B2. Tiled dequantize-into-shmem GEMM for M>4 (1.5 days)
- **WHAT:** separate prefill kernel (web-llm shape): `@workgroup_size(8,8)`, each thread a 4×4 register tile → 32×32 output tile per WG; K-loop steps 8; cooperatively load a 32×8 activation tile AND dequantize a 32×8 weight tile into two `array<f32,256>` workgroup buffers (each u32 dequantized once, amortized over 32 tokens); if/else zero-pad guards on `row < batch_size`. JS routes by M: M≤4 → B1 kernel, M>4 → GEMM. **Chunk prompts at ≤256 tokens** (bounds dispatch time under Safari/Metal watchdogs; zero-tvm's same-checkpoint qwen38 has a known chunked-prefill correctness quarantine above cap 256 — stay at/below it).
- **WHY:** weights read once per prefill instead of once per token; floor is one full weight pass (~0.4-0.5s at achieved bandwidth).
- **EXPECTED:** **prefill 6s → ~0.5-1s** short prompts (plus C2's readback removal). Also fixes P2P-mesh prefill scaling.
- **RISK:** f32 shmem (2KB) — core WGSL, portable everywhere; skip the chromium-experimental subgroup-matrix variant.
- **VALIDATE:** goldens: first generated token after M∈{2, 8, 33, 256}-token prefills must match llama.cpp exactly (edge M values catch tile-padding bugs).

---

## C. System-level (do immediately after A1 — cheapest big win)

### C1. One command encoder per token; submit once (1 day)
- **WHAT (JS):** `recordWholeForward(pos)`: embedding + all 64 layers + LM head + argmax into ONE `GPUCommandEncoder`, one `queue.submit` per token (replaces ~67 submits/token). Per-dispatch uniforms from a **reuse pool** (one buffer per dispatch slot — `queue.writeBuffer` executes immediately while passes are deferred, so a shared uniform would be clobbered; this is TVM's exact design). Pre-create every bind group/buffer at load; zero `createBuffer`/`createBindGroup` in the token loop.
- **WHY:** each submit costs ~0.08-0.35ms of validation/IPC; more importantly at the 10-15 tok/s target (66-100ms/token budget) per-stage submits would consume 15-60% of the budget.
- **EXPECTED cumulative (with A1):** **~7-9 tok/s**; ~10-50ms/token recovered.
- **RISK:** none — core WebGPU, identical on Dawn/Metal/wgpu; hundreds of dispatches per encoder is exactly what MLC ships.
- **VALIDATE:** goldens unchanged (pure restructure — any token diff is a real ordering/aliasing bug); paired A/B for the timing claim.

### C2. On-GPU argmax chained into next input + pipelined 4-byte readback; readback-free prefill (1 day)
- **WHAT:** portable (non-subgroup) argmax kernel over the 248320 logits → 4-byte `tokenOut`; encoder appends `copyBufferToBuffer(tokenOut → inputIds, 4)` so the next step reads its input on-GPU — no CPU in the decode loop. Token ids reach JS via a **PIPELINE_DEPTH=2 ring** of 4-byte MAP_READ buffers; submit step k+1 before awaiting token k; drain ring slots in a `finally` (mapAsync on a pending slot is a validation error). **Prefill:** submit every prompt token fire-and-forget (`wantReadback=false`), `onSubmittedWorkDone` every ~64 tokens for progress/cancel only; only the last prompt token reads back (it doubles as the first generated token).
- **WHY:** kills the ~1MB f32 logits copy + full pipeline drain per token; zero-tvm's profile shows pipelining hides essentially all residual submit/readback cost at 42 tok/s.
- **EXPECTED cumulative:** **~8-10 tok/s**; prefill loses its per-token sync bubbles immediately (small now, prerequisite for B).
- **RISK:** core WebGPU only. EOS check moves to id arrival (worst case wastes 1 in-flight token). Temperature sampling later = GPU sampler kernel, same chaining. **Scope:** on-GPU chaining is single-device decode; the P2P split path keeps its per-shard network hop (already one encoder per shard).
- **VALIDATE:** goldens vs the C1 build (argmax on GPU must reproduce JS argmax bit-for-bit — it's f32 compares, deterministic); checksum tokenOut ≠ 0.

---

## D. Explicitly rejected (this week)

| Idea | One-line reason |
|---|---|
| Subgroup reductions (`subgroupAdd`) as baseline | Not in shipping Safari 26 (only Safari TP 249); Chrome-only fast path worth ~5-10% of matvec — defer, feature-gate later. |
| shader-f16 activations for matvec | Weights dominate traffic; f16 x/scales cap at ~5-10% total bytes; optional feature + unverified Deno f16 — not worth the week's slots. |
| dot4I8Packed on Apple/Safari | Metal emulates dp4a as 4 scalar mul-adds; llama.cpp explicitly vendor-excludes apple. |
| int8 KV cache | Context-capacity win (f32 KV is 512KB/token), ~nothing for short-context decode tok/s — next week. |
| MoE/expert-style weight-traffic grouping | Measured 2.6x LOSS at decode-scale row counts (Apple caches already serve re-reads). |
| Shrinking dispatches / more rows-per-WG beyond tuned range | Occupancy floor: zero-tvm lost 15-19% tok/s halving WG count on small dispatches. |
| `timestamp-query` profiling in production | Requesting the feature alone measured ~3x decode slowdown; use wall-clock + onSubmittedWorkDone. |
| Fused QKV+RoPE+KV-append | Measured +2.3% shrinking to ~+0.1% once vec4 loads land; C1 already removes its submit benefit — revisit only if dispatch count shows up in profiles. |
| Multi-request micro-batching | Aggregate throughput only; zero single-stream gain (exo: single-stream got *slower* pipelined). |
| Speculative decoding | Needs B2's batched kernel + draft model + KV rollback; ~1.5-2x candidate for a later week. |
| Indirect dispatch | Enabler with no raw win; matvec dims are fixed — only needed for future pre-recorded variable-KV loops. |
| Wire/network optimizations (f16 activations on DataChannel, etc.) | ~1-5ms/token vs a 400→80ms GPU problem; decode hops are serial and can't be hidden anyway. |

---

## Single most likely failure mode

**The cooperative-matvec rewrite (A1) benches 3-5x faster on day 2, but the token-exact golden tests go red — and the week is lost triaging whether each mismatch is benign f32 reduction-order drift (256-thread tree vs. serial accumulation flips near-tie argmaxes, legitimately diverging from llama.cpp after ~20-50 tokens) or a real bug (Q4 nibble/-8 offset, scale indexing `r*nb+b` under the new thread mapping, or the LM-head 2D dispatch fold silently skipping rows).** Mitigate up front: build the layer-level logit-tolerance harness (compare stored per-layer activations, max |Δ| threshold) *before* touching the kernel, keep token-exact as the end-to-end gate but allow divergence-point analysis, and checksum every dispatch output non-zero so a silently-skipped dispatch can never masquerade as a fast, wrong kernel.