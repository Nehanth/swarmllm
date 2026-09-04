<p align="center">
  <a href="https://swarmllm.ai"><img src="favicon.svg" width="72" alt="SwarmLLM"></a>
</p>
<h1 align="center">SwarmLLM</h1>
<p align="center"><b>Every device brings a slice. Together they run the whole model.</b></p>
<p align="center">
  <a href="https://swarmllm.ai">Site</a> ·
  <a href="https://swarmllm.ai/room">Start a swarm</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="docs/bench-log.md">Benchmarks</a> ·
  <a href="roadmap/">Roadmap</a> ·
  <a href="SECURITY.md">Threat model</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>
<p align="center">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-2b4eff">
  <img alt="runtime" src="https://img.shields.io/badge/runs%20on-WebGPU%20%2B%20WebRTC-16171c">
</p>

SwarmLLM runs large language models across the devices in a room, in their browser tabs. Each device holds a slice of the model; a 10 KB activation vector passes between them over direct WebRTC connections. Nothing to install, no accounts, no server does any thinking.

- **27B in browser tabs.** Qwen 3.8 27B (15 GB of Q4_0 weights) across laptops, phones and PCs that individually can't hold it.
- **Native-competitive decode.** A from-scratch WebGPU engine (~50 WGSL kernels) at the memory roofline: 9.0 tok/s plain and 16 tok/s with speculative decoding on a GB10, where native llama.cpp measures 8.0 on the same file and GPU. ([bench log](docs/bench-log.md))
- **Bit-exact by construction.** Every optimization is gated on golden tests; the speculative path produces the same stream as plain decoding.
- **Cross-network.** Rooms span networks via WebRTC; prefill sends 16 tokens per round trip and decode chains speculative drafts so a slow link still moves several tokens per lap.
- **Text stays on the host.** Peers see only mid-model activations (which are *not* private against a determined peer, see [SECURITY.md](SECURITY.md)).

## Quick start

**Use it:** open [swarmllm.ai/room](https://swarmllm.ai/room), create a room, share the code, pick a model, start. Every device downloads only its layers (cached for next time).

**Run it locally:**

```bash
git clone https://github.com/Nehanth/swarmllm && cd swarmllm
npx -y serve -l 8080 .        # any static server works; then open http://localhost:8080/room
```

**Hack on the engine** (needs [Deno](https://deno.com) 2.x and a WebGPU-capable GPU; model files go under `models/`, see [docs/models.md](docs/models.md)):

```bash
npm test              # unit tests, no GPU
npm run test:gpu      # golden tests on Qwen3 0.6B
npm run test:q38      # 27B suites incl. speculative-vs-plain equality
npm run bench:q38     # decode / prefill tok/s
```

## How it works

```
host      embed the last token → hidden state (5,120 floats)
   ↓ 10 KB over WebRTC
peer A    layers 0–21           ─┐
peer B    layers 22–42           ├─ each device runs its slice on its own GPU
peer C    layers 43–63          ─┘
   ↓ back to the host
host      final norm → LM head → sample → next token (and the draft head guesses the one after)
```

Generating a token is memory-bound: every token reads all of the weights once. So the engine's job is reading fewer bytes (4-bit blocks with f16 scales) and reading them well (64 threads sweep each row together, dequantize in registers, reduce in shared memory), with one command submit per token. The runtime's job is making network laps carry more: batched prefill, and multi-token-prediction speculation verified in a single batched pass with exact rollback of the recurrent state. Details: [docs/architecture.md](docs/architecture.md), [docs/kernels.md](docs/kernels.md), [docs/protocol.md](docs/protocol.md).

## Supported models

| Model | Format | Notes |
|---|---|---|
| Qwen 3.8 27B | GGUF Q4_0 | hybrid Gated-DeltaNet + attention; MTP speculation |
| Qwen3 0.6B / 1.7B / 4B | GGUF Q8_0 / Q4_0 | dense; used for golden tests |
| SmolLM2 135M | safetensors f32 | smallest demo |

Browsers: Chrome/Edge 113+, Safari 26+. Headless: Deno 2 (wgpu). See [docs/models.md](docs/models.md).

## Performance

Qwen 3.8 27B Q4_0, greedy, bit-identical output at every row. Full history with commits in [docs/bench-log.md](docs/bench-log.md).

| Device | Decode plain | Decode speculative | Native llama.cpp (same GGUF) |
|---|---|---|---|
| NVIDIA GB10 (Deno / Vulkan) | 9.0 tok/s | 16.1 tok/s | 8.0 tok/s |
| MacBook Pro (Chrome / Metal) | 6.7 tok/s | 10.8 tok/s | — |
| Cross-internet room (host + peer) | — | 3.5–6 tok/s | — |

Prefill is the known gap (27 tok/s vs 377 native on the GB10); a tiled GEMM for prefill is in progress ([benchmarks/bench_gemm.js](benchmarks/bench_gemm.js)).

## Repository layout

```
engine/            the runtime (ES modules; engine.js re-exports the public API)
  dense.js         DenseEngine: dense Llama-architecture models (Qwen3, SmolLM)
  qwen35.js        Qwen35Engine: hybrid Gated-DeltaNet + attention, batched paths, MTP speculation
  gguf.js          GGUF parsing, quantization repacking, streaming upload
  wgsl/            base.js (shared kernels) · coop.js (generated GEMV family) · qwen35.js (DeltaNet kernels)
  tokenizer.js · sampling.js · quant.js · autotune.js · selftest.js · safetensors.js
room.js + room/    the room: signaling, mesh, weight streaming, generation loop; wire/markdown/sampling/models helpers
index.html         landing page          p2p.html   the room's markup (served at /room)
tests/             GPU golden tests · unit/ (no GPU) · golden/ · reference/ · run.sh
benchmarks/        tok/s harnesses, kernel-family profiler, GEMM prototype
docs/              architecture · tech-stack · kernels (every trick, measured) · protocol · models · bench log · research
roadmap/           one file per planned item with status, design and done-criteria
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [GOVERNANCE.md](GOVERNANCE.md). Benchmark reports from hardware we don't have are especially welcome (there's an issue template).

## Citation

```bibtex
@software{swarmllm2026,
  author = {Narendrula, Nehanth},
  title  = {SwarmLLM: peer-to-peer LLM inference across browser tabs},
  year   = {2026},
  url    = {https://github.com/Nehanth/swarmllm}
}
```

## Acknowledgements

Model weights and the GGUF format come from the [Qwen](https://huggingface.co/Qwen) team and [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp), whose speculative-decoding graph for Qwen 3.5/3.8 was the reference for ours. Prior work that shaped this: [Petals](https://github.com/bigscience-workshop/petals), [exo](https://github.com/exo-explore/exo), [WebLLM](https://github.com/mlc-ai/web-llm), [LlamaWeb](https://arxiv.org/abs/2605.20706), and the Gated DeltaNet and PipeInfer papers.

## License

[MIT](LICENSE).
