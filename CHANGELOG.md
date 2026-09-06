# Changelog

All notable changes to SwarmLLM. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Room topology is linear, not a mesh: every device keeps one link to the host; chain neighbours connect only when the layers are dealt (`ensureLink`). A 64-device room opens 126 links instead of 2,016. Workers draw the device list from the host's roster. `?signal=host:port` points the room at a self-hosted PeerServer.
- Room emulator (`npm run e2e`): `--devices N --phones K`, local PeerServer, 27B from local disk, room-log error capture, fail-fast on load errors. 16 devices on the 27B verified on GB10.
- Hidden-state wire (`room/transport.js`): activations travel on a dedicated data channel as ≤4.6 KB slices, striped over several peer connections (`?wire=stripeN`, default 4; `?wire=off` to disable). On a 100 ms link a token's hidden state now crosses a hop in 51 ms instead of 152, and a depth-5 verify block in 52 ms instead of 355 (docs/bench-log.md, transport section). Bytes only, output unchanged.
- Multi-token-prediction speculative decoding using the model's built-in draft layer; output is bit-identical to plain decoding. Solo host and device chains.
- Adaptive draft depth (3/5/7) chosen by measured tokens per second; 8 batch columns so a depth-5 verify is a single pass.
- Batched prefill: 4 (now up to 8) tokens per GPU pass, 16 tokens per network round.
- Device autotune for the cooperative GEMV shape at load.
- Binary WebRTC frames and f16 wire format for activations (one third of the original bytes per hop).
- Fused gate/up GEMV with in-kernel SiLU; fused DeltaNet gate + L2-norm pre-pass; accumulate-into-residual matvec variants.
- Bench log, kernel-family profiler, GEMM prefill prototype, research plans and four implementation designs under `docs/` and `docs/research/`.
- A roadmap (`roadmap/`) of 27 items, each mirrored by a tracking issue.

### Changed
- Repository restructured for open source: `engine/` split into focused modules behind a compatible `engine.js` barrel (dense, qwen35, wgsl/*, gguf, tokenizer, sampling, quant, autotune, selftest, safetensors); the room page split into `p2p.html` (markup) + `room.js` + `room/*` helpers; tests, benchmarks, goldens and references moved under `tests/` and `benchmarks/`; `BelloEngine` renamed `DenseEngine`.
- Cooperative GEMV kernels (64 threads per row, coalesced loads, shared-memory reduction): 27B decode 3.6 → 9.0 tok/s plain, 16 tok/s with speculation on a GB10.
- f16 block scales end to end; `unpack4x` dequantization with runtime probe.

### Fixed
- Generation no longer runs past the context window (#31): the room stops before the KV cache would overflow and says "stopped: context full", refuses prompts that leave no room for an answer, and the context is 2048 tokens (was 512). Cache size does not change logits (`tests/test_ctx.js`, bit-identical).
- **`f32ToF16` dropped the rounding carry**, silently halving ~0.03% of all values (e.g. -1.9999911 to -1.0): about 1.4 corrupted numbers per 5,120-float activation frame on every hop of a split room, plus occasional halved Q4/Q8 block scales. The carry now propagates into the exponent; all 65,536 representable f16 values round-trip exactly.
- Tall matvec dispatches over 65,535 workgroups (the LM head at 2 rows per workgroup) were silently dropped; all matvecs now dispatch in 2-D.
- Draft depth selection by lap time could lock a Mac-hosted room at maximum depth (1.5 tok/s); replaced by throughput-based selection.

## [0.1.0] - 2026-08-29

Initial public demo: Qwen 3.8 27B split across browser tabs over WebRTC, from-scratch WebGPU engine, Qwen3 and SmolLM support.
