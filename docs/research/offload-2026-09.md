# MoE expert offload and SSD streaming: research for SwarmLLM (2026-09-26)

Tags: **[M-ours]** we measured it today (GB10 box, headless Chromium 131 / Deno WebGPU, engine from `origin/tabby-new-idea`); **[M]** measured and published by the authors, with a reproducible setup; **[C]** a claim or projection; **[E]** my estimate.

## TL;DR

- Both X posts stream MoE experts from SSD with MLX on Apple Silicon. **Edge0** gets 15–20 tok/s for a *modified* Qwen3.6-35B-A3B on a 24 GB Mac mini. It does this by cutting top-8 routing to top-4 and replacing the router with a trained predictor, which is lossy: −3.9 points on average. The **@thefp4brain** DeepSeek V4.1 Flash run is exact, but it takes **~23 s per token** on a 16 GB M1.
- Real routing traces from Qwen3.6-35B-A3B **[M-ours]**:
  - A per-layer LRU cache holding 25% of experts hits 76–87% of lookups; 50% hits 94–96%; 75% hits 98–99%.
  - Pinning a fixed set of "hot experts" profiled on other text is near useless (hit rate ≈ cache fraction), because routing is context-specific.
  - Applying layer *l+1*'s router to layer *l*'s output predicts the next layer's experts with 77–87% recall at top-8 and 92–98% at top-16. It needs no training.
- Browser I/O is not the bottleneck **[M-ours]**. OPFS `createSyncAccessHandle` reads come off the warm page cache at ~20 GB/s. Cold reads are ~1.2 GB/s per worker, 4.8 GB/s with 4 workers. Chrome's WebGPU limits on this box are maxBufferSize 4 GiB and maxStorageBufferBindingSize 4 GiB−4.
- **Expert parallelism across devices is a bad fit for batch-1 decode over WebRTC.** With experts split across 2 devices, both devices have work in >99% of layer-steps, so every layer needs a network round trip. Keep the layer split, and use SSD spill per device.
- **Blocker found:** the branch's loader cannot load bartowski's Qwen3.6-35B-A3B "Q4_0" GGUF. The file has 60 shared-expert tensors in Q5_0 (ggml type 6) and BF16 MTP router tensors (type 30), so `ggmlTypeBytes` returns −1. This is the file the docs recommend.

---

## 1. The two X posts

Both posts were read through the fxtwitter / vxtwitter / syndication JSON mirrors, since x.com returns 402.

### @SamuelZengML, 2026-09-10: Edge0

Post text: *"A 35B language model running on an iPhone using only 1–2.5 GB of peak memory… we're open-sourcing Edge0"* (10.3k likes, 720k views). Sources: [post](https://x.com/SamuelZengML/status/2097861839287927139), repo [Edge0-AI/edge0](https://github.com/Edge0-AI/edge0) (MLX, Apple Silicon only, CUDA "on the roadmap"), paper [arXiv 2609.18063](https://arxiv.org/abs/2609.18063), "The Other Half of the Memory Wall".

**Technique** (from `docs/streaming.md`, `docs/prerouter.md` and `streaming/options.py`):
- **SSD streaming.** Weights are 4-bit affine (group 64) safetensors, mmapped. Per-expert byte ranges are read on demand into a cross-layer LRU (`cache_slots=64`) plus a small set of fixed GPU "staged slots". A slot table maps expert id to slot on the GPU, so "indices never leave the GPU": there is no host sync per layer.
- **Prerouter** (the key idea). A small MLP head per layer (hidden 512; 33 heads for layers 6–38) takes layer N's MoE input at token *t* and predicts **layer N+1's routing for token t+1**. This "double shift" gives the loader a whole token of lead time.
- **The prediction is consumed as the routing itself.** The paper says the head "entirely replaces the standard router", so a miss is impossible by construction. Output is therefore not the base model's.
- **Top-k cut from 8 to 4** (`staged_k4`, `top_k=4`). The stated reason: "narrowing from K=8 to K=4 nearly doubles decode (3.3 to 6.4 tok/s)".
- **Recovery LoRA** (r=16, unmerged, 42 MB), distilled from the fp16 teacher to win back the quality lost to int4, K=4 and the prerouter.
- **I/O tricks.** `madvise(WILLNEED)` over the predicted byte ranges cut staging from 162 ms to 27 ms per step on a 16 GB M2: faults serialise on the VM map lock, and kernel bulk readahead avoids that. Incremental in-place stacking of the staged experts also helped.

**Numbers** (**[M]** unless marked):

| Setup | Decode | Memory | Source |
|---|---|---|---|
| edge0-35b (K=4), Mac mini M4 Pro 24 GB | 14.9–17.7 tok/s (README); 20.4 (paper) | 2.9 GiB "peak active" | README, paper |
| Same machine, fully resident MLX baseline | 3.9 tok/s | 18.2 GiB | paper |
| 16 GB M2, K=4, prerouter vs on-demand | 6.4 vs 3.5 tok/s | — | paper table |
| 16 GB M2, K=8, prerouter vs on-demand | 3.3 vs 1.8 tok/s | — | paper table |
| 16 GB M2, prod profile after I/O fixes | 5.33 tok/s (187.6 ms/step) | 2.6 GiB MLX + 4.7 GiB page cache | `options.py` docstring |
| iPhone, "1–2.5 GB peak" | not benchmarked in the repo | — | tweet **[C]** |

Quality, OpenCompass **[M]**: average 79.2 vs 83.2 for fp16 Qwen3.6-35B-A3B. AIME 2026 is 86.6 vs 92.7.

**Caveats:**
1. "Peak active memory" excludes the page cache. On the 24 GB Mac most of the 19.5 GB checkpoint stays in the page cache ("warm" runs), and that is where the 15–20 tok/s comes from. On 16 GB it is 5–6 tok/s.
2. Adjacent tokens "agree on only about a quarter of a layer's expert set". The paper attributes the gain to moving load time off the critical path (blocked time −58%), not to prediction accuracy.
3. It is not the same model: K=4 plus the replaced router plus LoRA.

### @thefp4brain, 2026-09-11: DeepSeek V4.1 Flash on a 16 GB M1 Mac mini

The linked post is a reply. The parent post says: *"got deepseek V4.1 flash running locally on a 16GB m1 mac mini… original FP4/FP8 weights, ssd streaming + custom mlx runner… 108s ttft and about 23s/token."* Sources: [post](https://x.com/thefp4brain/status/2098519817812660288), repo [atbender/deepseek-v41-flash-mac-mini](https://github.com/atbender/deepseek-v41-flash-mac-mini).

**Technique:**
- The runner reads the original 475 GiB checkpoint with bounded `pread` calls: only the 6 routed experts per layer, plus the needed rows of the huge Engram lookup tables.
- It keeps a 4 GiB first-fit cache of decoded dense matrices and compiles the FP4/FP8 dequant.
- It does **no prefetch and no adaptive expert cache**; the README says so.

**Numbers [M]** (short runs, the author says not rigorous):

| Configuration | Decode |
|---|---|
| Baseline | 30.7 s/token |
| + allocator reuse | 27.9 s/token |
| + 4 GiB dense cache | 26.3 s/token |
| + compiled dequant | 22.6–22.8 s/token, TTFT 108 s, 5.65 GiB MLX peak |

The 4 GiB cache pushed the system into 1.8 GiB of swap.

**Takeaway:** the run is exact, but it is an existence proof, not a usable speed.

## 2. Prior work: offloading, prediction, caching

| Work | Core idea | Prediction / cache | Result | HW |
|---|---|---|---|---|
| Eliseev & Mazur, [2312.17238](https://arxiv.org/abs/2312.17238) | Offload Mixtral experts | Per-layer **LRU** (k=2–4 of 8); **speculative load by applying the next layer's gate to the current hidden state** (~60–70% recall for 1–2 experts) | 2.1–3.1 tok/s Mixtral at 2–3-bit experts **[M]** | A100, 3080M, 3060, T4 |
| LLM in a flash (Apple), [2312.11514](https://arxiv.org/abs/2312.11514) | Flash-resident dense FFN; sparsity predictor; windowing; row-column bundling | Predicts active neurons; keeps a sliding window resident | Runs models 2× DRAM; 4–5× CPU, 20–25× GPU speed vs naive loading **[M]** | Mac, M1 Max |
| PowerInfer, [2312.12456](https://arxiv.org/abs/2312.12456) | Hot neurons on GPU, cold on CPU | Offline profile plus online predictors | Up to 11.69× vs llama.cpp **[M]** | RTX 4090 |
| PowerInfer-2, [2406.06282](https://arxiv.org/abs/2406.06282) | Neuron-cluster compute; segmented cache plus I/O pipeline | Neuron cache | 47B at 11.68 tok/s on a phone; up to 27.8× **[M]** | Smartphone |
| Pre-gated MoE, [2308.12066](https://arxiv.org/abs/2308.12066) | **Retrain** the gate to select the next block's experts | Exact lookahead by construction | Faster, less memory (Switch-Transformer) **[M]** | 1 GPU |
| MoE-Infinity, [2401.14361](https://arxiv.org/abs/2401.14361) | Batch-1 activation tracing | Request-level activation matrix drives prefetch and cache | 3.1–16.7× per-token latency vs vLLM, Ollama, DeepSpeed **[M]** | Personal GPU |
| Fiddler, [2402.07033](https://arxiv.org/abs/2402.07033) | Run missed experts **on the CPU** instead of moving weights | — | 1.26× single batch; 11.57× beam search **[M]** | 1 GPU + CPU |
| HOBBIT, [2411.01433](https://arxiv.org/abs/2411.01433) | Load **low-precision** copies of less critical missed experts | Layer-level prefetch; multi-dimensional cache | Up to 9.93× decode **[M]** | Jetson, consumer GPUs (llama.cpp) |
| ProMoE, [2410.22134](https://arxiv.org/abs/2410.22134) | Proactive cache | Learned predictor from intermediate results | 2.20× prefill, 2.07× decode on average **[M]** | Consumer GPU |
| AdapMoE, [2408.10284](https://arxiv.org/abs/2408.10284) | Adaptive number of experts | Sensitivity-based gating plus prefetch and cache | −25% experts, 1.35× **[M]** | Edge |
| ExpertFlow, [2410.17954](https://arxiv.org/abs/2410.17954) | Predict the whole routing path; group tokens by route | Transformer route predictor; predictive cache | Up to −93.7% GPU memory, up to 10× **[M]** | 1 GPU |
| MoE-Lightning, [2411.11217](https://arxiv.org/abs/2411.11217) | CPU-GPU-I/O pipeline; roofline model | Paged weights | Up to 10.3× throughput, Mixtral on a T4 (batch) **[M]** | T4 |
| Klotski, [2502.06888](https://arxiv.org/abs/2502.06888) | Expert-aware multi-batch pipeline | Correlation-aware prefetcher | Up to 85× throughput (batch) **[M]** | GPU+CPU+disk |
| Cache-conditional experts, [2412.00099](https://arxiv.org/abs/2412.00099) | **Cache-aware routing**: prefer resident experts when scores are close; training-free | — | ~2× on phone; small quality cost **[M]** | Mobile |
| KTransformers, [repo](https://github.com/kvcache-ai/ktransformers), SOSP'25 | Attention and hot experts on GPU; experts on CPU (AMX/AVX-512 int4/int8) | Static placement | Server-class (e.g. DeepSeek-R1 on 8×L20 + Xeon) | Needs CPU SIMD: N/A in browser |
| llama.cpp | `mmap` (page-cache "offload"); `-ot/--override-tensor exps=CPU`, `--n-cpu-moe` | OS LRU | Page-fault path is slow for models larger than RAM | CPU+GPU |
| llama.cpp [disc. #27149](https://github.com/ggml-org/llama.cpp/discussions/27149) | Expert-aware reads, expert-contiguous layout | LRU | Qwen3-30B-A3B on a 16 GB M1: 4.7 tok/s vs 0.9 for whole-layer reads; 3.8 tok/s at 75% hit, 4.1 at 88% **[M]** (prototype) | M1 16 GB |
| [flash-moe](https://github.com/danveloper/flash-moe) | C/Metal; parallel `pread` | **OS page cache only**. Custom Metal LRU, LZ4 and F_RDADVISE all lost; temporal prediction −18% (25% accurate) | Qwen3.5-397B-A17B (209 GB) at 4.36 tok/s, **K cut 10→4**, ~71% page-cache hit **[M]** | M3 Max 48 GB, 17.5 GB/s SSD |
| [ssd-moe/deepseek-v4-flash-mlx](https://github.com/ssd-moe/deepseek-v4-flash-mlx) | Dense 8-bit resident; mxfp4 experts on SSD; LRU; parallel `pread` | LRU | ~4.5–5 tok/s **[M]** (author) | 48 GB Mac |
| SSD-LLaMA, [2609.18110](https://arxiv.org/abs/2609.18110) | SSD/RAM/VRAM tiers; no pruning or substitution | Expert delivery pipeline | Trillion-parameter MoE at 1+ tok/s; decode 2.1–15.6× vs baselines **[M]** | RTX 5090, 32 GB RAM |
| [quantumnic/ssd-llm](https://github.com/quantumnic/ssd-llm) | Layer streaming with `madvise` | LRU plus pinning | Table is estimates, tok/s "TBD" **[C]** | — |

Also relevant: SSD offload costs a lot of energy ([2508.06978](https://arxiv.org/pdf/2508.06978)).

**Patterns:**
1. The fast "SSD" systems (Edge0, flash-moe) **reduce K** and lean on the page cache. Their headline speeds do not come from exact routing.
2. Next-layer-gate lookahead (Eliseev) works without training.
3. Previous-token prediction is weak: flash-moe measured 25%, and we measured 34–46% (below).
4. Big resident caches beat cleverness.

## 3. In a browser

### Limits and I/O throughput [M-ours]

Measured on GB10, Linux, Chromium 131 headless; scripts in `/tmp/offbench`, not in the repo.

- **WebGPU limits (Chromium on this box):** maxBufferSize = 4,294,967,296 and maxStorageBufferBindingSize = 4,294,967,292. A stacked Q4_0 expert tensor is 144 MiB (35B) or ~1.3 GB (122B), so it binds in one piece on desktop. Phones cap bindings far lower, and Mac values must be checked on the device.
- **OPFS sync handles** (dedicated worker), random reads of 1.77 MB (one Q4_0 expert's gate+up+down):
  - **Warm page cache:** 0.08 ms per read, ~21 GB/s, i.e. native speed. Native `pread` is 0.06 ms.
  - **Cold** (evicted with `fadvise`): ~1.2 GB/s per worker; **4.8 GB/s with 4 workers** reading 4 files.
  - Native `O_DIRECT` at queue depth 1 on this NVMe is 2.5 GB/s.
  - Only one sync handle per file is allowed, so **shard the experts into several files** (e.g. per layer) to get parallelism.
  - Beware that an incognito or ephemeral profile stores OPFS in memory. It measured ~1 GB/s, which is not disk.
- **User-picked file** (`File.slice().arrayBuffer()`, the File System Access / input path, no copy into OPFS): ~0.35–0.46 GB/s serial, **~1.3 GB/s ceiling** even when warm (IPC-bound). Use it to import a GGUF into OPFS once, not as the hot path.
- **Disk to GPU:**
  - `queue.writeBuffer` of 1.77 MB pieces: 3.8 GB/s (0.46 ms each).
  - Larger `writeBuffer` calls (28–64 MB) drop to ~1 GB/s.
  - A 4-deep `mapAsync(WRITE)` staging ring plus `copyBufferToBuffer`: **9 GB/s**.
  - Upload is therefore not the limit. A cold expert costs ~1.5 ms of disk plus ~0.2–0.5 ms of upload per worker, and the workers run in parallel.

### How often experts hit the cache, from real traces [M-ours]

**Method:**
- We patched a scratch copy of `origin/tabby-new-idea` (loader: added Q5_0; engine: copy `moeB.sel`/`selw` and the residual after every layer).
- We ran bartowski's Qwen3.6-35B-A3B Q4_0 (the file must be the full 20,836,243,072 bytes) under Deno WebGPU on GB10, one token at a time, with greedy decoding.
- Three runs, each with its prompt as warm-up: code (700-token prompt + 250 decode tokens), README prose (700 + 250) and a chat story (67 + 500).
- Output was coherent. The model is 40 layers × 256 experts, top-8.
- **Limit:** short sessions; one model.

| Per-layer cache (fraction of experts) | 6% | 12% | 25% | 38% | 50% | 75% |
|---|---|---|---|---|---|---|
| LRU hit rate, decode | 0.45–0.58 | 0.61–0.73 | **0.76–0.87** | 0.88–0.94 | **0.94–0.96** | **0.98–0.99** |
| Static "hot set" profiled on the *other* texts | 0.02–0.07 | 0.05–0.12 | 0.15–0.26 | 0.30–0.43 | 0.42–0.58 | 0.65–0.82 |
| Half static + half LRU | 0.28–0.35 | 0.50–0.59 | 0.68–0.77 | 0.79–0.88 | 0.87–0.94 | 0.98 |

**Other trace results:**
- Adjacent tokens share 34–46% of a layer's 8 experts. This matches Edge0's "about a quarter".
- Within one text, the top 64 of 256 experts cover 83–90% of that text's decode activations. Across texts, the hot set moves.
- **Prefill touches almost everything.** A 700-token prompt uses 202–211 of 256 experts per layer; 64 tokens use 86–109.
- Misses carry about their share of routing weight: at a 25% cache, 11–21% of routing weight lands on missed experts. Dropping misses is therefore not free.
- Top-4 of the 8 selected experts carry only 65% of the routing weight on average (p10 = 57–59%). **K=4 is a real model change.**

**Training-free lookahead:** predict layer l+a's experts as top-m of `W_router[l+a] · rmsnorm(x_out[l]) · postnorm[l+a]`.

| Layers ahead | top-8 recall | top-16 | top-24 |
|---|---|---|---|
| 1 | 0.77–0.87 | 0.92–0.98 | 0.95–0.99 |
| 2 | 0.67–0.80 | 0.83–0.95 | 0.89–0.97 |
| 4 | 0.58–0.74 | 0.73–0.90 | 0.80–0.94 |
| 6 | 0.51–0.67 | 0.66–0.85 | 0.73–0.90 |
| Previous token, same layer | 0.34–0.46 | — | — |

The ranges span code (the lower end) and chat (the higher end). The predictor costs one extra 256×2048 GEMV per predicted layer (2 MB of F32), which is negligible.

**Timing catch:** one layer of 35B-A3B decode is ~0.5 ms on a fast GPU, which is less than one cold 1.5 ms read. One-layer lookahead cannot hide a cold read, but 4–6 layers of lookahead with top-16/24 overfetch, or cross-token lead time as in Edge0, can.

### Exactness and sync

The engine today runs a whole token in one submit, with routing kept on the GPU. Exact streaming needs the CPU to learn about a miss before that layer's expert kernel runs. The options:
- **(i)** One `mapAsync` sync per layer (40 per token; each is a GPU drain plus IPC, ~0.2–1 ms **[E]**).
- **(ii)** Chunked encode (e.g. 4–8 layers per submit). At each chunk boundary, read back the lookahead predictions, load the predicted misses, run the chunk with a GPU "miss flag", and on a flagged miss restore the DeltaNet snapshot (the machinery already exists for speculative decoding) and replay the chunk.
- **(iii)** Lossy: cache-conditional routing or dropping missed experts with renormalisation, as Edge0 and flash-moe do. Off by default, labelled.

With a 75% cache, a token makes ~320 lookups and ~3–6 of them miss. **Almost every token has at least one miss**, so a design with no sync at all has to be lossy.

### Expert parallelism vs layer split [M-ours for the split statistics]

With experts split between 2 devices (by id halves, or balanced by frequency), the busier device gets 61–64% of a layer's 8 experts. All 8 land on one device in only 0.2–0.6% of layer-steps.

So every layer needs a scatter and gather across the network: **80 crossings per token** for 40 layers, vs **1 per boundary** with a layer split. At 1–5 ms per Wi-Fi hop that is 80–400 ms per token.

Expert parallelism only pays off for batched or throughput work, such as large prefill with token grouping. Roadmap 10's "replicate hot experts" also does not survive the data: there is no stable global hot set.

## 4. What to build (ranked)

### (a) Qwen3.6-35B-A3B on two 24 GB Macs

The 20.8 GB Q4_0 file split by layers is ~10.4 GB per Mac, so **it fits without any SSD**.

| # | Build | Benefit | Cost | Risk |
|---|---|---|---|---|
| 1 | **Loader: Q5_0 (type 6) and BF16 (type 30).** Dequant, then requantise to Q8_0 like Q6_K. Test on the real file. | Unblocks the recommended file (today it throws `Invalid typed array length: -1`) | Hours | Low |
| 2 | **Measure on the Macs:** decode tok/s solo and split, Chrome's allocatable GPU memory on 24 GB | Replaces estimates | 1 day | Low |
| 3 | **Expert-grouped batched prefill and verify:** sort a batch's (token, slot) pairs by expert, one GEMM per touched expert, fixed-order combine | Prefill reads ~210 experts per layer instead of 8 × 700. Up to ~25× fewer expert bytes for a 700-token prompt [E]. Speculative verify windows also get cheaper. | 1–2 weeks | Medium: keep batched == single bit-exact |
| 4 | **Single-Mac mode with an OPFS expert cache** (see b-1). One 24 GB Mac holding ~70–75% of experts should hit ≥98% [M-ours stats] with ~3–6 misses per token, ~5–20 ms of I/O [E]. | Removes the network hop; one device suffices | See b-1 | See b-1 |
| — | Expert parallelism across the 2 Macs | Negative for decode | — | Don't |

### (b) Qwen3.5-122B-A10B

Model shape: 48 layers, 256 experts, top-8. An expert's gate+up+down is 5.3 MB at Q4_0, so experts total ~65 GB, the whole model ~70 GB [E], and a token touches ~2.0 GB of routed weights.

| # | Build | Benefit | Cost | Risk |
|---|---|---|---|---|
| 1 | **Per-device expert slot bank + OPFS spill, exact:** GPU slot table (expert id → slot), per-layer LRU, experts sharded into per-layer OPFS files, 4 read workers, `mapAsync` staging ring. v1 syncs per layer (option i). | 4 × 24 GB (~16–18 GB usable each [E]) needs ~10–20% spill: expect ≥99% hits and only a few ms per token. A 64 GB Mac alone (~70% cached): ~98% hits, ~50 MB per token, ~10–40 ms per token of I/O [E]. | 2–3 weeks | Medium: Chrome memory caps; OPFS quota (70 GB) and eviction (`persist()`); first-run copy |
| 2 | **Lookahead prefetch** (training-free router lookahead, 2–6 layers, top-16/24) plus **chunked submit with miss-flag and DN-snapshot replay** (option ii) | Removes most per-layer syncs and hides the I/O. The 32 GB single-machine case (~25% cached, ~76–87% hits, ~300–500 MB per token) moves from I/O-stall to overlap: ~2–6 tok/s instead of ~1–3 [E] | 2–3 weeks | Medium-high: replay complexity; needs 122B traces |
| 3 | **Layer placement aware of memory and SSD speed:** give devices with bigger caches and faster SSDs more layers; probe OPFS cold GB/s when a device joins | Balances the pipeline | 1 week | Low |
| 4 | **Opt-in lossy fallback on a miss:** a low-bit copy of experts (HOBBIT) or cache-aware re-routing (2412.00099); never Edge0-style K=4 by default | Bounds the worst case on a 16–32 GB single device | 1–2 weeks plus evals | Quality; must be labelled |
| 5 | **Trained cross-token prerouter** (Edge0) | A full token of lead time | Weeks, plus a training pipeline | Changes the model. Last. |

**Preconditions for (b):**
- Re-run the trace tool on 122B (and on GLM-5.x); hit rates may differ with 1024-wide experts.
- Check the 122B GGUF's tensor types. The same Q5_0 issue is likely.

**Not recommended:** a static hot-expert set (it measured at about the cache fraction), whole-model page-cache reliance (browsers cannot `mmap` into the GPU), or relying on the user-picked-file path for hot reads (~1.3 GB/s ceiling).

## Reproduction

- Traces: a scratch export of `origin/tabby-new-idea` in `/tmp/tabby`. It adds `trace_moe.js`, `analyze.py`, `pred.py` and `miss.py`, plus two small patches (Q5_0 dequant in `engine/gguf.js`; trace copies in `engine/qwen35.js`).
- Browser I/O: `/tmp/offbench` (`run.mjs`, `run2.mjs`, `run3.mjs`, `run4.mjs`).
- Nothing was committed.
