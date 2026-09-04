# Changelog

All notable changes to SwarmLLM. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Multi-token-prediction speculative decoding using the model's built-in draft layer; output is bit-identical to plain decoding. Solo host and device chains.
- Adaptive draft depth (3/5/7) chosen by measured tokens per second; 8 batch columns so a depth-5 verify is a single pass.
- Batched prefill: 4 (now up to 8) tokens per GPU pass, 16 tokens per network round.
- Device autotune for the cooperative GEMV shape at load.
- Binary WebRTC frames and f16 wire format for activations (one third of the original bytes per hop).
- Fused gate/up GEMV with in-kernel SiLU; fused DeltaNet gate + L2-norm pre-pass; accumulate-into-residual matvec variants.
- Bench log, kernel-family profiler, GEMM prefill prototype, research plans under `docs/`.

### Changed
- Cooperative GEMV kernels (64 threads per row, coalesced loads, shared-memory reduction): 27B decode 3.6 → 9.0 tok/s plain, 16 tok/s with speculation on a GB10.
- f16 block scales end to end; `unpack4x` dequantization with runtime probe.

### Fixed
- Tall matvec dispatches over 65,535 workgroups (the LM head at 2 rows per workgroup) were silently dropped; all matvecs now dispatch in 2-D.
- Draft depth selection by lap time could lock a Mac-hosted room at maximum depth (1.5 tok/s); replaced by throughput-based selection.

## [0.1.0] - 2026-08-29

Initial public demo: Qwen 3.8 27B split across browser tabs over WebRTC, from-scratch WebGPU engine, Qwen3 and SmolLM support.
