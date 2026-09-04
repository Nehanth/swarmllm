# The WebGPU engine: kernels, tricks and hacks

Everything here was measured on a DGX Spark (GB10, 184 GB/s achievable through WebGPU) and the maintainer's MacBook. Every row of the table below has a commit and a number in [bench-log.md](bench-log.md). Output is bit-identical to the reference at every step unless a row says otherwise.

## Where the time goes

A generated token on the 27B = ~111 ms on the GB10: **82 ms streaming 15 GB of weights through the matvec kernels** (183 GB/s: at the roofline) and ~30 ms of everything else (small kernels, command encoding, submit, logits readback). Prefill passes cost ~139 ms for 4 tokens. Measured with `benchmarks/bench_breakdown.js`, which re-times passes with one kernel family skipped at a time.

## Kernel families

| Family | Entry points | Role |
|---|---|---|
| Cooperative GEMV | `matvec[_q8|_q4]_coop[_acc]` | decode: `ROWS` rows per workgroup of `WG` threads; 64 threads sweep one row together |
| Batched GEMV | `matvec*_coop_b[4][_acc]` | prefill / speculative verify: `COLS` columns per weight read; `_b4` twin for ≤4 live columns |
| Fused gate/up | `matvec*_gu[_b[4]]` | FFN gate and up in one weight sweep, SiLU applied in the epilogue |
| Glue | `rmsnorm`, `qsplit`, `head_norm`, `rope_part`, `attn_scores/softmax/out`, `sigmoid_mul`, `silu_mul`, `add_res`, `argmax` + `_mc` multi-column forms | normalization, attention, activations, sampling prep |
| DeltaNet | `dn_pre` (gates + L2), `dn_conv`, `dn_delta`, `dn_gatenorm` + `_mc` forms | the gated delta-rule recurrence, with snapshot slots for speculative rollback |
| Legacy | `matvec[_q8|_q4]` | one thread per row; kept as the correctness reference and fallback |

## The tricks, with what each one bought

### Reading weights
1. **Cooperative rows (2.2× on the 27B).** One thread per row gives scattered memory traffic. Instead 64 threads sweep a row with interleaved 32-weight blocks, so every cycle the group touches one contiguous stripe and the GPU coalesces it. Partials merge in a shared-memory halving tree with `workgroupBarrier()` *outside* the `if`.
2. **Scalar accumulators (the 3× bug).** Four running totals in a `array<f32, 4>` indexed by a variable spilled to scratch memory and made the "fast" kernel 3× slower than the naive one. Named scalars `acc0..acc3` fixed it. Corollary: all unrolled code is generated with literal indices; a WGSL `for` over a constant bound does not unroll on wgpu.
3. **Dequantize in registers.** Nibbles are extracted from packed `u32` words and `(nibble − 8) × scale` happens in registers; full-precision weights never exist in memory. The scale multiplies each block's partial sum once, not each weight.
4. **f16 block scales end to end (~10% fewer bytes).** GGUF's native f16 scales stay packed two per `u32` and are decoded with `unpack2x16float`; requantized tensors round scales to f16 first, matching the spec.
5. **Load-time repacking.** The file interleaves `[scale][nibbles]`; at load they are split into a nibble array and a scale array so trick 1's stripes are contiguous. Layout, not compression.
6. **`unpack4xU8` / `unpack4xI8` dequant (neutral on NVIDIA, kept for Metal/Android).** One instruction per four packed bytes instead of shift/mask/convert per element; a runtime probe compiles a one-line shader and falls back to the shift form where the builtins are missing.
7. **vec4 aliased binding views.** The same buffer is declared as `array<u32>` for one entry point and `array<vec4<f32>>` for another; legal as long as no single entry point references both views.
8. **Device autotune (~1 s at load).** Times `(WG, ROWS)` candidates on the actual GPU and keeps the winner with a 3% noise guard. The GB10 prefers (64, 4).

### Fewer passes, fewer bytes moved
9. **One command submit per token.** All ~1,100 dispatches of a token are encoded into a single command buffer.
10. **Fused gate/up + SiLU.** Both FFN projections read the same input, so one kernel computes both and applies `silu(g)·u` before writing. Per-buffer helper functions are generated instead of storage-pointer parameters, which are unsafe on Safari. Neutral on the GB10 (bandwidth-bound), aimed at Metal's dispatch cost.
11. **Accumulate-into-residual matvecs (`_acc`).** `y[row] += W·x` folds the residual add into the o-proj, DeltaNet out-proj and FFN down-proj: 192 fewer dispatches per token. Measured neutral on Vulkan, which taught us small dispatches are nearly free there.
12. **Fused DeltaNet pre-pass (`dn_pre`).** Gates (sigmoid β, decay) and the per-head L2 norms of q and k run in one dispatch after the causal conv.
13. **GPU argmax.** A single-workgroup reduction over the 248,320 logits with lowest-index tie-breaking (matches the CPU loop exactly) so the draft chain reads back 8 bytes instead of 1 MB per draft.

### Batching
14. **Batched prefill (3 → 39 tok/s).** 4 (now up to 8) prompt tokens per GPU pass as columns; each weight block is read once for all columns. Causality inside a pass: recurrent kernels process columns strictly in order; K/V are appended before attending; per-column position uniforms.
15. **Multi-column glue kernels (`_mc`, 2× on batched passes).** One dispatch covers all live columns instead of one dispatch per column.
16. **4-column twin kernels.** With 8 batch columns, a 4-token verify would pay for 4 garbage columns of ALU; a second kernel set generated for 4 columns is chosen per dispatch by live column count.
17. **8-wide columns were neutral for prefill** (an 8-column pass costs 2× a 4-column pass), which is how we learned prefill is not weight-streaming-bound and needs a real GEMM.

### Speculation
18. **Multi-token prediction with exact rollback (9 → 16 tok/s).** The model's `nextn` layer drafts up to 7 tokens; one batched trunk pass verifies them; DeltaNet states are snapshotted after every non-final column into per-layer shadow slots (7 of them), written inside the recurrence kernel; a rejection copies the right slot back. KV caches need no rollback (position-capped). Any sampler stays lossless because the trunk decides.
19. **Snapshot slot base packed into the frame uniform.** `frame.snap` carries `(total << 8) | (base + 1)` so an 8-column verify split into chunks on any device writes global slot indices.
20. **Draft depth by measured throughput.** A lap-time threshold locked a Mac-hosted room at depth 7 (1.5 tok/s); the room now probes depths 3/5/7 and keeps the fastest measured tokens per second.

### Correctness and portability hacks
21. **2-D dispatch for tall matvecs.** WebGPU silently drops a dispatch over 65,535 workgroups in one dimension; the LM head at 2 rows per workgroup needs 124,160. Kernels derive `row0` from `(wg.y * 32768 + wg.x)`.
22. **No f16 accumulation** (corrupted output); **no timestamp queries** (3× slowdown when merely enabled); **no subgroups** (absent on Safari, hidden on Deno; measured worth 0–8% anyway).
23. **Feature probes at load** for `unpack4x` and the WGSL extensions, compiled as one-line shaders with `getCompilationInfo()`.
24. **Generated helper functions per buffer** instead of `ptr<storage>` parameters (Safari).
25. **Bank-conflict padding** in the GEMM prototype's shared tile (row stride 17 vec4s instead of 16): the only change that moved that kernel (1.25×).

### On the wire (the room)
26. **Binary frames (−33%) and f16 activations (−50%)** over WebRTC data channels; decoders accept f32 from older peers.
27. **16 prompt tokens per network round** (4 GPU passes per round) so prefill pays one round-trip per 16 tokens.
28. **Cache API with size stamps**: the cache refuses 206 responses, so range responses are stored as 200 with an `x-swarm-len` stamp and validated on read.
29. **Fire-and-forget prefill dispatches** with periodic `onSubmittedWorkDone()` syncs to keep the queue from growing unbounded.

## Things tried and rejected (measured)
- Q4 KV cache: −92.5% prefill.
- External 0.6B draft model: different vocabulary (151,936 vs 248,320), can't verify.
- RMSNorm fused into the GEMV: 0.91× on Metal.
- Tree/Medusa drafting: no trained heads, and a DeltaNet state per branch.
- Subgroup reductions: 0–8% (the shared-memory tree was never the bottleneck).
- 4×4-per-thread GEMM tiles: slower than 2×4 (occupancy beat reuse).

## Open items
- Prefill GEMM: `benchmarks/bench_gemm.js` is correct and 1.25× over the batched GEMV path; still latency-bound at ~30 GB/s. Next: register prefetch tuning, split-K for small matrices, 256-thread workgroups.
- Register-resident `dn_delta` (see [deltanet-prefill-spec.md](deltanet-prefill-spec.md)): ~4% of a pass, bit-identical.
