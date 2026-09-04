# Tech stack

What SwarmLLM is built from, and why each piece was chosen. There is no build step and no framework: the site is three HTML/JS entry points plus ES modules.

| Layer | Technology | Why |
|---|---|---|
| GPU compute | **WebGPU** (WGSL compute shaders) | The only GPU API a web page can use on Chrome, Edge and Safari. Same shaders run headless in Deno (wgpu) for tests and benchmarks. |
| Kernels | Hand-written WGSL, generated from JS templates | Shapes (threads per row, rows per workgroup, batch columns) vary per device; generation keeps one source of truth. No dependency on an ML compiler. |
| Model format | **GGUF** (Q4_0 / Q8_0), safetensors for the tiny demo | GGUF is the lingua franca of local models: every popular model has one, and the block-quantized layout (32 weights + f16 scale) is ideal for bandwidth-bound decode. |
| Weight delivery | HTTP range requests to Hugging Face, streamed straight into GPU buffers | Each device fetches only its layers' byte spans; RAM never holds the whole model. |
| Weight cache | Browser **Cache API** with a size stamp | Rejoining a room reloads from disk in seconds; the stamp guards against truncated entries. |
| Peer connections | **WebRTC** data channels via PeerJS signaling | Direct browser-to-browser, cross-network (STUN hole-punching), encrypted (DTLS). The broker only introduces peers; no model traffic touches a server. |
| Wire format | Binary frames, activations as f16 | 10 KB per hop for a 5,120-wide hidden state; one third of the original JSON/f32 bytes. |
| Speculation | The model's own multi-token-prediction (`nextn`) layer | No second model needed; verification by the full model keeps output bit-identical. |
| Hosting | Static files on Vercel (`/room` rewrites to `p2p.html`) | Nothing runs server-side. Staging deploys from the feature branch, production from `main`. |
| Testing | Deno (WebGPU headless) + golden JSON references + CPU reference implementations | GPU tests compare argmax and logits against references; CI runs the no-GPU unit tests. |
| Native reference | llama.cpp (`llama-bench`) on the same GGUF | The honest baseline for "how fast is native on this hardware". |

## Runtime targets

| Target | Status | Notes |
|---|---|---|
| Chrome / Edge 113+ | primary | subgroups and `shader-f16` used only when probed |
| Safari 26+ | supported | no subgroups, no storage-pointer function parameters, no Web Bluetooth; everything feature-detected |
| Deno 2 (wgpu, Vulkan/Metal) | tests and benchmarks | `deno run --unstable-webgpu`; exposes no `subgroups` in `adapter.features` even when the device supports them |
| Firefox | untested | WebGPU shipping progressively |

## Models

Qwen 3.8 27B (hybrid Gated-DeltaNet + attention, `qwen35` architecture family), Qwen3 0.6B/1.7B/4B (dense), SmolLM2 135M. See [models.md](models.md).

## Things deliberately not used

- **No ML compiler (TVM/MLC), no ONNX Runtime.** They don't support layer-splitting mid-model, hybrid DeltaNet architectures, or our wire formats; owning the kernels made every later optimization possible.
- **No f16 accumulation.** Corrupted output; f16 is used only for block scales and the wire.
- **No timestamp queries.** Enabling them cost 3× on the GB10; we profile by skipping kernel families instead.
- **No accounts, tokens, ads, or servers doing inference.** See [../GOVERNANCE.md](../GOVERNANCE.md).
