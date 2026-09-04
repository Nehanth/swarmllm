# The WebGPU engine

All kernels are WGSL, written by hand or generated from JavaScript templates in `engine/engine.js` (`coopWGSL`) so shapes can vary per device. Nothing is compiled at build time; browsers compile the shaders at load.

## Kernel families

| Family | Entry points | Role |
|---|---|---|
| Cooperative GEMV | `matvec[_q8|_q4]_coop[_acc]` | decode matvecs: `WG` threads per workgroup, `ROWS` rows per workgroup, 64 threads sweep a row; `_acc` adds into the output (residual folded in) |
| Batched GEMV | `matvec[_q8|_q4]_coop_b[4][_acc]` | prefill / speculative verify: `COLS` columns per weight read; a 4-column twin is used when fewer columns are live |
| Fused gate/up | `matvec*_gu[_b[4]]` | FFN gate and up projections in one pass with SiLU applied in the epilogue |
| Glue | `rmsnorm`, `qsplit`, `head_norm`, `rope_part`, `attn_scores/softmax/out`, `sigmoid_mul`, `silu_mul`, `add_res`, `argmax` (+ `_mc` multi-column variants) | normalization, attention, activations |
| DeltaNet | `dn_pre` (gates + L2), `dn_conv`, `dn_delta`, `dn_gatenorm` (+ `_mc`) | the gated delta-rule recurrence with snapshot slots for speculative rollback |

## Rules that came from measurements

- **Dynamically indexed local arrays spill to scratch memory.** A four-element accumulator array made a kernel 3× slower than the naive one; use named scalars, and generate unrolled code with literal indices. A WGSL `for` over a constant bound does not unroll on wgpu.
- **`workgroupBarrier()` stays outside conditionals.**
- **Accumulate in f32.** f16 accumulation corrupted output. f16 is fine for block scales and for the wire.
- **Feature-detect, never assume.** Subgroups are absent on Safari 26 and hidden on Deno; storage-pointer function parameters are unsafe on Safari (helper functions are generated per buffer instead); `unpack4xU8/I8` is probed at load with a shift/mask fallback.
- **Timestamp queries cost 3× just by being enabled.** Profile by skipping kernel families instead (`benchmarks/bench_breakdown.js`).
- **Dispatches over 65,535 workgroups in one dimension are silently dropped.** All matvecs dispatch in 2-D; the LM head at 2 rows per workgroup needs 124,160.
- **Small dispatches are nearly free on Vulkan.** Removing ~200 per token changed nothing on the GB10; fusions are kept for Metal, where dispatch cost is higher.
- **Bank-conflict padding matters** for shared-memory tiles (the GEMM prototype gained 20% from a 17-wide row stride).

## Quantization

Weights arrive as GGUF Q4_0 or Q8_0: blocks of 32 with one f16 scale. At load, blocks are repacked into separate packed-nibble/byte arrays and f16 scales (two per `u32`, decoded with `unpack2x16float`). Dequantization happens in registers: `(nibble − 8) × scale`, with the scale applied once per block. Requantized tensors round scales to f16 first, matching the GGUF spec.

## Autotune

At load, `autotuneCoop` times a few `(WG, ROWS)` shapes for ~1 s on the actual device and keeps the winner (with a 3% noise guard favoring the default). The GB10 prefers (64, 4); Apple GPUs differ.

## Testing a kernel change

1. `npm run test:gpu` — goldens on the 0.6B (bit-exact argmax and logit tolerance).
2. `npm run test:q38` — 27B goldens, batched-vs-sequential equality, and `test_mtp` (speculative stream must equal plain).
3. `npm run bench:q38` and add a row to `docs/bench-log.md`.
