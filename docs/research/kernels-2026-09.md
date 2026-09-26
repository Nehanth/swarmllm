# Outside techniques vs. this engine (September 2026 survey)

A survey of current inference-kernel and scheduling techniques, checked against what this repo has already measured and built. Nothing here was measured on a GPU by the survey itself; numbers are the repo's own (docs/bench-log.md, docs/research/*) or the cited sources'. Ranked by expected gain on the 27B × feasibility here.

**Two corrections to common assumptions.** Prefill is not bound by the DeltaNet recurrence (`dn_delta_mc` is 1–4% of a 4-column pass, ~24 ms of a ~210 ms 16-column pass; deltanet-prefill-research.md §9, prefill-gemm-v2.md §7): it is the matvec family, above all the **Q8_0 tensors (`ffn_down`, `ssm_out`, `attn_output`) that still run on the batched GEMV**. And ~11.6 ms of the GB10's "30 ms of other work" per token is Deno's `mapAsync` poll sleep, which Chrome does not pay, so Deno-only savings do not transfer 1:1 to the Mac.

## 0. The split-device "gibberish" and the f16 wire

Already fixed (CHANGELOG): workers never reset DeltaNet state between questions; the `spec` flag was not forwarded past the first worker (stale rollback); rollback/reset could be overtaken by the next frame; `f32ToF16` dropped the rounding carry (wire error 2.93e-3 → 1.98e-4). Each alone produces wrong output in split rooms.

f16 overflow looks unlikely at the one measured boundary: max|x| = 278.7 at layer 32 with 3 massive channels (decode-overhead-and-wire.md §2.3), ~235× headroom; a 2026 survey of 27 checkpoints puts Qwen3.5-family maxima at 10²–10³ ([arXiv 2605.15572](https://arxiv.org/abs/2605.15572)); massive activations enter early and persist to late layers ([arXiv 2603.05498](https://arxiv.org/pdf/2603.05498)). Qwen3 dense models can overflow fp16 *inside* layers ([Qwen3-Embedding-8B thread](https://huggingface.co/Qwen/Qwen3-Embedding-8B/discussions/21)); this engine keeps f32 inside a device, so only the wire is exposed.

Open risks: (1) only layer 32 was measured, not the last hop (after layer 63) nor the early spike layers; (2) `badF32` sampled every 97th element, so an Inf in one of ~3 massive channels was missed ~96% of the time, and WGSL's finite-math assumption ([gpuweb #5109](https://github.com/gpuweb/gpuweb/issues/5109)) can turn Inf into finite garbage on Metal with no error; (3) no 27B test covers the f16 wire (the split tests pass f32 arrays); (4) split ≠ solo by design (the host's heads see f16-rounded hiddens); (5) the packer rounds ties away from zero, not to even as its comment said, which matters if packing moves to the GPU (`pack2x16float` is RNE).

## Ranked items

| Rank | Item | Status | Expected gain | Exact? |
|---|---|---|---|---|
| 0 | f16 wire audit: full finite + range scan, per-boundary max\|x\| telemetry, f16-wire split test | new | correctness | test only |
| 1 | Q8_0 variant of the prefill GEMM (`ffn_down`, `ssm_out`, `attn_output`); llama.cpp MMQ q8_0 tiles ([PR #12135](https://github.com/ggml-org/llama.cpp/pull/12135)) | new | pass-level prefill 56.7 → ~72 tok/s | prefill tolerance |
| 2 | Batched draft-cache (MTP) fill during prefill: one 16-column pass instead of one submit per column (6.4 ms/token, ~48% of solo prefill after the GEMM) | designed (prefill-gemm-v2 §7 #3) | solo prefill 43.7 → ~55–65 tok/s | yes (drafts only) |
| 3 | Flash-decoding (split-K over sequence) for the 16 attention layers ([Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)); `attn_softmax` is `@workgroup_size(1)`, attention grew 0 → 8.7 ms/token from pos 8 to 500, ~30 ms extrapolated at 2048 | new | −25% token time near full context | re-baseline goldens |
| 4 | One-submit speculative step (GPU embedding gather from the argmax buffer) + encode-ahead | designed (roadmap 26) | GB10 ceiling −52 ms per K=3 step (~16 → 20 tok/s); encode-ahead −7.45 ms/token | yes |
| 5 | Dispatch fusion for Metal: 32–71 µs per dispatch on Metal vs 24–36 on Vulkan ([arXiv 2604.02344](https://arxiv.org/pdf/2604.02344), [2608.08730](https://arxiv.org/abs/2608.08730)); at 898 dispatches/token ≈ 29–64 ms of the Mac's 149 ms. Fuse `dn_conv`+`dn_pre`+`dn_delta`+`dn_gatenorm` per value head (−3 × 48 dispatches); `qsplit`+`head_norm`×2+`rope_part`×2; K/V append in `rope_part` | partly designed | Mac +10–20% | yes if per-element order kept |
| 6 | Chunked WY/UT DeltaNet prefill at C=16 ([FLA](https://github.com/fla-org/flash-linear-attention), [Gated DeltaNet](https://arxiv.org/abs/2412.06464)); register-resident `dn_delta` for decode (5.2 → 3.3 ms/token, RG=1 bit-identical) | specs written | prefill +8%, decode +2% | tolerance / yes |
| 7 | Replay-based rollback instead of per-column state snapshots (SGLang ReplaySSM, [v0.5.16](https://github.com/sgl-project/sglang/releases/tag/v0.5.16); SpecLA [arXiv 2607.16673](https://arxiv.org/abs/2607.16673)): today (n−1) × 144 MB of snapshots per verify and ~1 GB of shadow slots per device | new | ~0.9 GB VRAM freed per device, +1–3% spec | yes (same kernel, inputs, order) |
| 8 | DP4a with Q8_1 activations, prefill GEMM only (kernel-plan-3 A.4); Apple emulates dp4a | designed | prefill matvec ×1.3–1.4 (NVIDIA/AMD/Intel) | prefill tolerance |
| 9 | Subgroup-matrix GEMM (`chromium-experimental-subgroup-matrix` → Metal simdgroup_matrix / Vulkan coopmat); ORT MatMulNBits Phi-3.5 1K prefill 15 s → 5.4 s ([PR #23729](https://github.com/microsoft/onnxruntime/pull/23729)) | new, Chrome-flagged | the route to llama.cpp-class prefill | prefill tolerance |

Lower priority or covered: pipelined laps (PipeInfer, [arXiv 2407.11798](https://arxiv.org/abs/2407.11798), designed in kernel-plan-3 and network-scheduler §3; prefill pipelining is built); frequency-ranked draft vocabulary ([FR-Spec](https://arxiv.org/abs/2502.14856); the prefix version is `?draftvocab=N`; the draft head is ~1.27 GB ≈ 6.9 ms per draft); f16 KV cache (~0.7 ms/token at 2048, changes numerics). Not recommended: EAGLE-2/3 trees (no heads for Qwen 3.8, a DeltaNet state per branch, native MTP already ~85% acceptance), lookahead decoding (recurrent state per branch), streaming partial hidden states mid-layer (refuted, exact-forward-pass-ideas #22/#25), and the already-rejected f16 activation storage, subgroup reductions in the GEMV, int8 on the wire, and llama.cpp Metal's `yl` pre-scale (breaks single/batched bit-identity).

## What landed on the `kernels` branch, and how to measure it

Correctness was checked on SwiftShader with synthetic models (tests/e2e/*_synth.mjs); none of the speed effects below has been measured on a GPU yet. Each has a switch so one build can A/B it:

| Change | Switch (off) | Expected effect | Measure with |
|---|---|---|---|
| Q8_0 prefill GEMM | `Qwen35Engine.create({ gemm8: false })`, or `engine.gemm8 = false` at runtime | prefill pass 56.7 → ~72 tok/s on the GB10 (item 1) | `MODEL=q38 deno run --unstable-webgpu --allow-read --allow-env benchmarks/bench.js` (prefill line), `tests/test_gemm.js` for tolerance |
| Batched draft-cache fill | `engine.mtpBatchFill = false`, room `?mtpbatch=0` | solo prefill 43.7 → ~55–65 tok/s (item 2) | same bench, prefill line; `tests/test_mtp.js` must stay equal |
| Replay rollback | `replayRollback: false` | ~0.9 GB less GPU memory for a whole-model device; spec tok/s ±small | `tests/test_mtp.js`, `tests/test_mtp_split.js` (spec == plain); GPU memory in the browser task manager |
| Draft head over the first N vocab rows | `?draftvocab=N` (off by default) | cheaper drafts; acceptance may drop | room stats line "N% drafts accepted" and tok/s with N = 32768 / 65536 vs off |
| One-submit draft chain | on only with `draftChain: true` / room `?draftchain=1` (default off: +~675 MB of GPU memory for the embedding table, or its first `draftvocab` rows) | GB10 ceiling −52 ms per K=3 step (item 4) | spec tok/s with `?draftchain=1` vs without, best with `&draftvocab=32768` |
| Register-resident `dn_delta` / `dn_delta_mc` | none (bit-identical) | kernel 1.24× at 1 column, 2.03× at 16 (spec microbenchmark); ~2% decode, ~4% prefill end to end | `benchmarks/bench_breakdown.js` DeltaNet family |
| Workgroup attention softmax | `softmaxWG: false` (bit-identical) | grows with context: largest near 2048 tokens | decode tok/s at a long context |
| Fused attention glue (`attn_glue`: qsplit + q/k `head_norm` + partial rope) | `attnGlue: false` or `engine.attnGlue = false` (bit-identical) | −4 dispatches per attention layer (−64 per token on the 27B, ~2–4.5 ms on Metal at 32–71 µs each, item 5); the norms also move from one thread per head to a workgroup | `node tests/e2e/attnglue_synth.mjs` (identity + ms/token); on a GPU, decode tok/s with and without |
| Prompt-lookup drafts | room `?lookup=0` | faster on answers that repeat the context | room tok/s on a summarise/quote prompt, on and off |

The dense-engine fix (batched GEMVs dispatching half their workgroups when autotune picks 8 rows per workgroup) is a correctness fix with no switch: `node tests/e2e/engine_dense_synth.mjs` checks batched vs one-token hiddens for every autotune shape.
