# 02 · Prefill GEMM

**Phase:** now · **Status:** in progress (`benchmarks/bench_gemm.js`)

## Why
Prompt processing is the biggest remaining gap to native: 27 tok/s vs 377 for llama.cpp on the same GB10. Decode is at the memory roofline; prefill is not, because the batched GEMV path costs ALU per column and re-reads activations per row.

## Design
- A tiled Q4_0 GEMM for 16 token columns: weight tile dequantized once into shared memory, activations staged as vec4s, each thread computing a 2×4 (rows × cols) tile, two 32-weight blocks per barrier pair, register prefetch of the next blocks.
- Measured so far: correct (2.5e-7 vs the batched GEMV), 1.25× on the FFN shape, parity on 5120×5120 (occupancy: only 80 workgroups). Bank-conflict padding of the shared tile was the only change that moved it; a 4×4 thread tile lost to occupancy.
- Next: split-K for small matrices, 256-thread workgroups, check the cost of naga's bounds checks, then a 16-column engine mode selected per dispatch like the existing 4-column twin kernels.
- Companion: a register-resident `dn_delta` (spec in `docs/deltanet-prefill-spec.md`) for ~4% of a pass, and the chunkwise DeltaNet formulation for longer prompts.

## Done when
- Prefill ≥ 100 tok/s on the GB10 and a visible improvement on the Mac, bit-exact goldens intact, rows in `docs/bench-log.md`.
