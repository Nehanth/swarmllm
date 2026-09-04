# Models

SwarmLLM ships no weights. Browsers fetch tensors by HTTP range request from public Hugging Face repositories; local development and tests read the same files from `models/` (git-ignored).

## Supported

| Model | File | Engine | Notes |
|---|---|---|---|
| Qwen 3.8 27B | GGUF Q4_0 (~15 GB), includes the `nextn` draft layer | `Qwen35Engine` | 64 layers: 48 Gated DeltaNet + 16 attention; MTP speculation |
| Qwen3 4B / 1.7B / 0.6B | GGUF Q4_0 / Q8_0 | `BelloEngine` | dense; 0.6B is the golden-test model |
| SmolLM2 135M | safetensors f32 | `BelloEngine` | smallest demo; quantized to Q8 at load |

Architectures sharing Qwen 3.5/3.6/3.8's hybrid layout (e.g. Qwen 3.5 2B/9B) need only a config entry; see `p2p.html` `MODELS`.

## Local layout for tests and benchmarks

```
models/
  qwen/    model.gguf (Qwen3-0.6B Q8_0), model-q4.gguf, tokenizer.json, config.json
  qwen17/  model.gguf, tokenizer.json, config.json
  q38/     model.gguf (Qwen 3.8 27B Q4_0)
  model/   SmolLM2-135M: model.safetensors, tokenizer.json, config.json
```

## Adding a model

1. Confirm the GGUF tensor names and dims match one of the two engines (`gguf.js` has the name maps).
2. Add a `MODELS` entry in `p2p.html` (URL, layer count, bytes per layer for the split planner).
3. Generate a golden with the reference implementation in `tests/reference/` and add a test.
4. Record tok/s in `docs/bench-log.md`.
