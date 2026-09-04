# 02 · Prefill GEMM

**Phase:** now · **Status:** landed (Q4 shapes) · next: Q8 variant

## Why
Prompt processing is the biggest remaining gap to native: 27 tok/s vs 377 for llama.cpp on the same GB10. Decode is at the memory roofline; prefill is not, because the batched GEMV path costs ALU per column and re-reads activations per row.

## Design
- A tiled Q4_0 GEMM for 16 token columns: weight tile dequantized once into shared memory, activations staged as vec4s, each thread computing a 2×4 (rows × cols) tile, two 32-weight blocks per barrier pair, register prefetch of the next blocks.
- Measured so far: correct (2.5e-7 vs the batched GEMV), 1.25× on the FFN shape, parity on 5120×5120 (occupancy: only 80 workgroups). Bank-conflict padding of the shared tile was the only change that moved it; a 4×4 thread tile lost to occupancy.
- Next: split-K for small matrices, 256-thread workgroups, check the cost of naga's bounds checks, then a 16-column engine mode selected per dispatch like the existing 4-column twin kernels.
- Companion: a register-resident `dn_delta` (spec in `docs/deltanet-prefill-spec.md`) for ~4% of a pass, and the chunkwise DeltaNet formulation for longer prompts.

## Done when
- Prefill ≥ 100 tok/s on the GB10 and a visible improvement on the Mac, bit-exact goldens intact, rows in `docs/bench-log.md`.

## Update (Sep 2026 research round)

A complete kernel design with code now exists in [docs/research/prefill-gemm-v2.md](../docs/research/prefill-gemm-v2.md): the prototype was missing double-buffered shared memory, a wider register tile and 16 columns, and the design projects a 2.5x matvec improvement (pass 37 to ~76 tok/s). Batching the prefill tail is a hard prerequisite: at 16 columns a prompt currently falls back to up to 15 single-token passes.

## Landed (Sep 2026)

`engine/wgsl/gemm.js`: row-stationary Q4_0 GEMM, 16 columns, split-K pinned per shape, spliced into the shader module and reached through the `_dop` ladder (GEMM at the full batch width, GEMV twins below it, so decode and speculative verify are structurally untouched). Room runs at 16 batch columns with the prefill tail stepping down 16 → 8 → 4.

Measured on the GB10, whole 27B, pass level (no draft-cache fill): **34.6 → 56.7 tok/s (1.64× over the best GEMV width)**; end-to-end `bench.js` with the draft fill 30.2 → 43.7. Bit-close to the GEMV path (relDiff 5.4e-7 end to end, 1.6e-6 worst per-shape).

**Why not the projected ~2.4×:** in this GGUF, `ffn_down` (5120×17408, the single largest tensor by bytes, ~40% of a DeltaNet layer) and `ssm_out`/`attn_output` are stored **Q8_0**, so they stay on the GEMV. A Q8_0 variant of the same kernel (32 int8 = two `vec4<u32>` per block, sequential layout so no lo/hi interleave, `unpack4xI8` dequant) is the highest-value follow-up; then the MTP draft-cache fill as one 16-column pass (48% of solo prefill after this change).
