# Multi-device decode: parallelism choices, hop latency, and the other SwarmLLM

Researched 2026-09-26. Every number is tagged **[M]** (measured, by us or by the cited source on stated hardware) or **[C]** (claimed in a paper abstract or a README, not re-checked). Our own measurements come from `docs/bench-log.md`: GB10, Qwen 3.8 27B Q4_0, one machine emulating the devices over loopback. This note builds on `network-scheduler.md`, `decode-overhead-and-wire.md` and `exact-forward-pass-ideas.md` and does not repeat them.

## 0. The short version

- Over Wi-Fi, the internet or WebRTC in a browser, **pipeline parallelism (PP) is the only split that makes sense for one decode stream**. Tensor parallelism (TP) and expert parallelism (EP) need 80–128 synchronisations per token for our two models. Each sync costs milliseconds on Wi-Fi and in browsers, and TP has only about 50 ms of compute per token to save. TP also changes the order of the floating-point adds, so it breaks our bit-exact contract unless we redesign it for that.
- TP pays off only when a sync costs microseconds, as with RDMA over Thunderbolt 5 in exo 1.0, or when the devices are so slow that their compute dwarfs the syncs (distributed-llama on Raspberry Pis).
- Our emulator data fits roughly **t ≈ 62 ms + (12–19 ms × extra devices) per token**. The two biggest exact levers are therefore: **use fewer devices**, and **make each hop cost less**. After those comes pipelining speculative windows.
- enapt/SwarmLLM is a much broader product: a native binary with a persistent swarm, failover, APIs and security docs. Our engine is faster, it runs the Qwen 3.5/3.8 hybrid models and MTP (which theirs refuses), it runs phones as compute nodes, and it keeps a bit-exact contract. Their maintainer's point about phones holds up against our own numbers.

---

## 1. TP vs PP vs EP for decode over consumer networks

### 1.1 Communication per token (batch 1, f16 activations, one token)

| | Qwen 27B dense (d=5120 → 10 KB, 64 layers) | Qwen3.6-35B-A3B MoE (d=2048 → 4 KB, 40 layers, 256 experts, top-8 + 1 shared) |
|---|---|---|
| **PP, D devices** | D messages of 10 KB, **D sequential latencies** | D messages of 4 KB, D sequential latencies |
| **TP, D devices (Megatron: 2 all-reduces per layer)** | **128 all-reduces**. Each device sends about 2(D−1)/D × 10 KB per all-reduce, ≈1.3 MB/token/device at D=2, which is ~170 Mbit/s at 16 tok/s | 80 all-reduces of 4 KB each |
| **EP, D devices** | n/a | **80 all-to-alls** (dispatch + combine per layer). Each carries up to min(8, D−1) × 4 KB each way, ≈1 MB/token at D=4. Attention/DeltaNet still has to be replicated or TP'd |

Bytes are not the problem for PP: 10 KB is about 7 packets. The problem is **the number of sequential sync points**: PP has D, TP has 2L, EP has 2L.

### 1.2 Latency math: when can TP ever win?

Take 27B on one GB10: plain decode is about 100 ms/token **[M]**. That is memory-bound, so TP over 2 devices saves at most ~50 ms. Spread over 128 all-reduces, a sync has to cost **under ~0.4 ms** for TP2 just to break even.

| Link | Per-sync cost | 128 syncs | Verdict for 27B TP2 |
|---|---|---|---|
| RDMA over Thunderbolt 5 (macOS 26.2, exo 1.0) | ~3–50 µs **[C]** ([Geerling](https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5/), [AppleInsider](https://appleinsider.com/articles/25/12/20/ai-calculations-on-mac-cluster-gets-a-big-boost-from-new-rdma-support-on-thunderbolt-5)) | <7 ms | **TP wins** |
| TCP over Thunderbolt / 10 GbE | ~0.1–0.3 ms | 13–38 ms | marginal |
| Wi-Fi LAN (3–7 ms link latency, prima.cpp testbed) | ≥3 ms | ≥384 ms | loses: capped at ~2.6 tok/s vs ~10 on one device |
| Browser WebRTC + WebGPU readback/upload | ~15 ms per hop today **[M]** | ~1.9 s | never wins |
| Internet, 20–100 ms RTT | ≥10–50 ms | 1.3–6.4 s | never wins (enapt's arithmetic: Australia–Belgium ≈9.4 s/token for 28 layers at the fibre floor) |

The same math applies to EP: 80 syncs against about 8 ms of expert compute per token for a 3B-active model on a good Mac. **EP buys memory capacity, not speed**, and PP buys the same capacity with D syncs instead of 80.

**Exactness.** PP runs each layer's arithmetic unchanged. TP sums partial dot products from different devices, which changes the reduction order, so its output differs from single-device output unless the single-device kernels are rewritten to use the same split-K partition. That makes the reference depend on D. PP is the only one of the three that is exact by construction.

### 1.3 How other systems handle it (measured where available)

| System | Split | Numbers |
|---|---|---|
| **exo** (pre-1.0) | PP over LAN/Wi-Fi | Llama 3.2 3B on 1/2/3× M4 Pro: single request **49.3 → 44.4 → 39.7 tok/s**, multi-request 49.3 → 95.7 → 108.8 **[M]** ([blog](https://blog.exolabs.net/day-1/)). This is our curve: PP adds capacity and throughput, never single-stream speed |
| **exo 1.0 + MLX, TB5 RDMA** | TP | DeepSeek V3.1 671B at 21.1 / 27.8 / 32.5 tok/s on 1/2/4 Mac Studios **[M, third party]** (Geerling). Kimi-K2 1T: TP4 14.82 vs PP4 14.49 tok/s, a 2.3 % difference, but PP timed out on prompts over ~1.5K tokens **[M]** ([mlx#2990](https://github.com/ml-explore/mlx/discussions/2990)) |
| **distributed-llama** | TP (2ⁿ nodes, capped by the number of KV heads) | Llama 3 8B on Pi 5 over a GbE switch: 1.77 / 2.25 / 3.01 tok/s on 1/2/4 nodes **[M]** ([#33](https://github.com/b4rtaz/distributed-llama/discussions/33)). Qwen3 30B-A3B on 4× Pi 5: **13.04 tok/s** predict, ~636 kB sent + 1,057 kB received per token, sync 15–94 ms **[M]** ([#255](https://github.com/b4rtaz/distributed-llama/discussions/255)). TP works here because a Pi's compute dwarfs a switched-GbE sync |
| **prima.cpp** | piped-ring PP + disk offload, Halda scheduler | Six home devices on Wi-Fi (320–610 Mbps, 3–7 ms), phone and tablet included. 70B at 674 ms/token; 32B with spec decoding at 26 tok/s; claims 5–17× over llama.cpp. Estimates dllama TP on Wi-Fi at ~150 ms per all-reduce and seconds per token **[M on their rig / C]** ([2504.08791](https://arxiv.org/html/2504.08791)) |
| **TPI-LLM** | TP, star all-reduce | Finding: "link latency, not bandwidth, is the main issue"; going from 300 Mbps to 1 Gbps barely helped **[M]** ([2410.00531](https://arxiv.org/abs/2410.00531)). Targets memory-starved 70B, not speed |
| **llama.cpp RPC** | layer split; one network RPC per graph op | Gets slower as nodes are added on Mac Studios (Geerling) **[M, qualitative]** |
| **Petals** | PP over the internet | Llama 2 70B at 2.29 steps/s (<5 ms RTT) and 1.57 steps/s (100 ms RTT); BLOOM-176B at 0.83 steps/s across 14 real servers on two continents **[M]** ([2312.08361](https://arxiv.org/html/2312.08361)) |
| **EdgeShard** | PP, DP over device selection + partition (Jetsons + cloud) | up to 50 % less latency, 2× throughput **[C]** ([2405.14371](https://arxiv.org/abs/2405.14371)) |
| **LinguaLinked** | PP on phones, LP assignment, runtime rebalancing | 1.11–1.61× single-thread, 1.73–2.65× multi-thread **[C]** ([2312.00388](https://arxiv.org/abs/2312.00388)) |
| **Helix** | MILP max-flow placement over heterogeneous GPUs/links | up to 3.3× throughput, −24 % decode latency (24–42 nodes) **[C]** ([2406.01566](https://arxiv.org/abs/2406.01566)) |
| **HexGen** | asymmetric TP inside a stage + PP across stages | 2.3× lower latency deadlines **[C]** ([2311.11514](https://arxiv.org/abs/2311.11514)) |
| **BloomBee** (2026) | DP layer assignment + micro-batching + compression + SD | LLaMA-30B on 3 nodes: 67 tok/s aggregate at 20 Mbps (batch 32). SD **hurt** at 250 Mbps (75 vs 95 tok/s) **[M, throughput not latency]** ([2604.21072](https://arxiv.org/html/2604.21072v2)). enapt quotes "8.7–9.3 tok/s at ~80 ms RTT" from this paper; I could not find that figure |
| **mesh-llm** | llama.cpp RPC; PP for dense models, experts spread over nodes for MoE | +38 % on code with a local draft model at 75 % acceptance **[C]** ([repo](https://github.com/IvGolovach/mesh-llm)) |

**Pattern:** every system aimed at consumer networks does PP, puts its effort into **placement** (EdgeShard, Helix, Parallax, prima.cpp, BloomBee) and into **speculation**, and restricts TP to µs-class links.

---

## 2. Techniques that beat hop latency

| Technique | What it does for a PP decode stream | Evidence | Bit-exact? |
|---|---|---|---|
| **Speculative decoding, one draft block per lap** (what we ship: MTP, K=3/5/7) | one lap verifies K+1 tokens, so hop cost per token drops by the tokens accepted per lap | ours: 9.07 → 15.86 tok/s solo, 85 % accepted **[M]**; DSD formula saves (N−1)·t₁·(k−1)/k ([2511.11733](https://arxiv.org/abs/2511.11733)) | **Yes** with greedy verification and column-invariant kernels (our tests check this). DSD's "adaptive semantic thresholds" (+15–20 %) relax acceptance, so **not exact** |
| **Pipelined / continuous speculative windows** (several verify windows in flight, cancel on reject) | fills the idle stages. Each device is busy ~1/D of a lap today | PipeInfer 1.5–2.15× and tolerant of low acceptance **[C]** ([2407.11798](https://arxiv.org/abs/2407.11798)); FlowSpec 1.36–1.77× vs naive PP on 5 Jetson Orin Nano over LAN, 9.35 tok/s Llama2-7B **[M on their rig]** ([2507.02620](https://arxiv.org/html/2507.02620v1)); SpecPipe 4.19–5.53× TBT on 8 stages **[C]** ([2504.04104](https://arxiv.org/abs/2504.04104)); Speculative Pipeline Decoding ([2605.30852](https://arxiv.org/abs/2605.30852)) | **Yes**, provided rollback is exact. Our `tabby-next-2026-09.md` design uses replay from S_pre. Arithmetic: ~1.6 steps/lap with 2 windows at 85 % acceptance |
| **Token trees** (EAGLE-2, SpecPipe) | more accepted tokens per lap | as above | Exact, but **hard with DeltaNet**: each branch needs its own recurrent state. Stick to chains |
| **Several concurrent sequences** | fills the pipeline for **throughput**; per-stream latency does not improve | exo multi-request 49 → 109 tok/s on 3 devices **[M]**; enapt continuous batching 1.34–1.55× **[M]** | Yes, given batch-invariant kernels ([Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)). Ours already are per column |
| **Early exit** (CALM-style) | skips late stages | enapt priced it: untrained models need ~73 % of layers | **No.** Only LayerSkip-style self-speculation is exact, and it needs an early-exit-trained model ([2404.16710](https://arxiv.org/abs/2404.16710)). MTP is already a better draft |
| **Lookahead / Jacobi decoding** | draft-free n-gram guesses | ~1.5–2× on one GPU **[C]** ([2402.02057](https://arxiv.org/abs/2402.02057)) | Yes. Just another draft source; prompt/suffix lookup ([SuffixDecoding](https://arxiv.org/abs/2411.04975)) is cheaper for agent/code traffic |
| **Activation quantisation** (Q8 wire) | smaller frames | enapt: 3.76×, RMS <0.005, **−13.5 % decode on localhost** **[M]** | **No.** Not worth it: 10 KB is not the bottleneck |
| **Fewer hops** (smallest set of devices that fits) | removes D directly | ours: 3 devices 10.4 vs 16 devices 3.8–4.7 tok/s **[M, loopback]** | **Yes** |
| **Cheaper hops** (GPU-side pack, one readback, encode-ahead) | cuts the ~15 ms fixed cost per hop | encode-ahead −7.45 ms/token **[M]**; enapt's split costs ~47 ms/token on localhost **[M]** | **Yes** (bytes only) |

**Exactness caveat to settle before claiming "bit-exact vs single device" in a room.** `docs/kernel-plan-2.md` notes that the f16 wire breaks split-vs-solo bit-exactness, because a single device never rounds the residual stream to f16 at the split points. Either keep an f32 wire mode for the golden test, or define the contract as "identical to solo *with the same split points rounded*". Speculative vs plain equality holds either way.

---

## 3. enapt/SwarmLLM ([repo](https://github.com/enapt/SwarmLLM), read at 86deb40, 2026-09-26; ideas and citations only)

**What it is.** One Rust binary (Tokio, libp2p 0.56 Kademlia/GossipSub/QUIC, Axum, candle, optional llama.cpp backend). It is a P2P node, an OpenAI/Anthropic/MCP API server and a dashboard. Nodes download **only their own shards** (BLAKE3-verified on arrival and on every load) and run each model in a separate worker subprocess. Pipeline only. TP with a star all-reduce exists but is **off by default and LAN-only** (`tp_max_latency_ms` 10), with their own "9.4 s per token" argument against using it over a WAN.

**Scheduling.** Parallax-style shortest-path DP over EMA per-layer latencies, which are gossiped (top 32). Liveness filter; pipeline affinity for multi-turn KV; hot-standby nodes per segment; shards drift toward demand after ≥3 stable ticks. Direct peer chaining has been on by default since v0.3.109. A "regional pipelines" plan (2026-09-20) uses Vivaldi network coordinates to form nearby teams. The scheduler follows Parallax ([2509.26182](https://arxiv.org/abs/2509.26182)).

**Churn.** A single retry re-runs the scheduler without the dead peer. Failover to a stand-in **replays the retained inputs to rebuild KV**: token agreement P = 0.9965 vs 0.9966 intact, where before the fix it had drifted to 0.119 **[M]**. Chained and TP segments cannot be restored and end the request.

**Verification.** Shards are hash-checked. Results only get a well-formedness check; their docs say plainly that "fluent but wrong output passes", and trust scores move ±0.01–0.20. Speculation is documented as "not bit-identical", and a benchmark notes "split non-determinism".

**Measured performance.** Live swarm: **0.35 tok/s** on a 4-segment chain with peers 105–1043 ms away. TinyLlama 2-node GPU split: 40.0 → 13.85 tok/s, i.e. ~47 ms fixed split cost per token on localhost. Qwen-7B CPU split: ~10 % penalty. After a fix, decode is bound by CUDA submission count (1,085 submissions/token, 17.6 of 23 ms spent in the driver). Most other wins are TinyLlama/7B throughput numbers on an RTX 3070 (prefix cache 29.4×, cross-node prefix KV 12.9× TTFT on CPU). They list no WAN decode win as measured yet.

**Phones.** Phones are **clients** ("the app works on a phone"). They are not compute nodes. Their maintainer's claims, as relayed to us: "cover phones already at higher performance" is true in the sense that a phone reaching a desktop node's API gets desktop speed, with no hop through the phone. "Scaling up with phones never gets decent t/s … latency would be the issue" **matches our own measurements**: a phone at a 0.5 GiB pledge holds 2 of 64 layers and adds a full hop (plus ~45 ms Wi-Fi wake latency on an iPhone per `exact-forward-pass-ideas.md`), and 16 devices ran at 4 tok/s against 10 with 3. The caveat: single-stream latency cannot scale with phones, but room **throughput** can when many sessions fill the pipeline.

**Threat model.** A contributor's note (security.md, "Misuse of the Network") argues: "a limit counts only if someone other than the party it is meant to stop enforces it." Limits in a requester's own code bind only honest operators; limits applied by *serving* nodes survive a fork. For us this means our pledge caps and queue rules live in the host's JS, so a modified host ignores them. Today our rooms are invite-only, ephemeral and need a visible open tab, which limits the "swarm as a botnet resource" risk. Any public matchmaking would need **worker-side** enforcement: a worker refuses frames beyond its pledge or rate and stops when its tab is hidden.

| They do better | We do better |
|---|---|
| Install-once persistent swarm, DHT discovery, per-shard downloads and storage | Zero install (a browser tab), phones as real compute |
| Hot standby + KV-replay failover (we end the answer when a device leaves) | Faster engine: WebGPU 9.0 vs llama.cpp CUDA 7.99 tok/s on the same GGUF **[M]** |
| Latency-aware DP scheduling; regional grouping | Qwen 3.5/3.8 hybrid DeltaNet + MTP (they refuse Qwen 3.5) |
| Continuous batching, cross-node prefix KV, OpenAI/Anthropic/MCP APIs | MTP speculation inside the split pipeline, 85 % acceptance **[M]** |
| Written threat model: activation inversion (~81 % of text recoverable), "boomerang" host-holds-ends, misuse | Bit-exact contract with tests; wire-level SCTP measurements; per-hop cost ~15 ms vs their ~47 ms (different setups) |

**Worth adopting (ideas, not code):** (1) warm spares that take over a segment by **replaying the retained inputs** of the current answer instead of stopping it; our bit-exactness makes the takeover verifiable. (2) DP placement on *measured* per-link latency. (3) Serving-side limits. (4) Their measurement discipline ("verify the mechanism fired, not just that the number improved"). (5) A new idea our design makes possible: **exact spot checks**. A warm spare with the same kernels recomputes a random layer range and compares bytes. That catches wrong-but-fluent output, which enapt cannot detect. It works only between devices whose WGSL results agree bitwise, so compare within a vendor/driver class.

---

## 4. What to build next, ranked

Gains are modelled from our emulator fit (t ≈ 62 ms + ~12–19 ms per extra device per token) unless marked **[M]**. Every item below keeps output bit-exact.

| # | Build | Expected gain | Effort | Exact |
|---|---|---|---|---|
| **1** | **Fewest-hops planner.** In `room/plan.js`, choose the smallest set of devices whose memory covers the model, weighted by speed. Everyone else becomes a warm spare, and phones join the chain only when capacity requires it. Show "N devices idle as spares" | 16-device room → 3–4-device chain: **~4 → ~10 tok/s** (both ends **[M]** on loopback) | days | yes |
| **2** | **Cut the per-hop fixed cost** from ~15 ms to ~3–5 ms. Instrument the hop (unpack/upload/compute/readback/pack), pack f16 on the GPU before readback (it has to match the JS f16 rounding exactly), use one `mapAsync` per hop, `writeBuffer` on receive, and encode-ahead on workers | D=3: ~10 → ~13 tok/s; D=16: ~4 → ~8 tok/s | 1–2 wk | yes |
| **3** | **Two pipelined speculative windows** (`tabby-next-2026-09.md` §1: step ids in the frame, two replay slots, cancel on reject) | ~1.3–1.6× on 2–4-device rooms, more as D grows (PipeInfer/FlowSpec range) | 3–5 wk | yes if rollback is exact; gate on `test_mtp` equality |
| **4** | **Warm-spare failover by replay** (enapt's approach, plus an exact byte comparison) | an answer survives a leave instead of stopping; enables spot checks | 1–2 wk | yes |
| **5** | **Latency-aware chain order and host choice** from measured per-link RTT (Parallax/EdgeShard-style DP; brute force for D ≤ 6) | 0 on one LAN; up to ~2× on mixed-site rooms | ~1 wk | yes |
| **6** | **Better drafts for agent traffic**: suffix/prompt lookup combined with MTP; adaptive K from per-hop telemetry | +20–40 % tokens/lap on code **[C]** | 1–2 wk | yes |
| **7** | **Batched multi-session decode** (`tabby-next` §2) | room throughput ~B× (exo: 2.2× on 3 devices **[M]**); single-stream unchanged | 3–4 wk | yes (column-invariant kernels) |
| **8** | Loss handling on real Wi-Fi: unordered channel + FEC (issue #34) | fixes the p90 (a 1 % loss turns 10 KB p50 152 ms into p90 253–356 ms **[M]**) | 1–2 wk | yes |

**Do not build:** TP or EP across devices (§1.2: loses on every link we will see, and inexact); Q8/f16-delta activations; early exit; relaxed acceptance (DSD-adaptive); outline decoding (Jupiter, [2504.08242](https://arxiv.org/abs/2504.08242), changes the text).

**Before any of this,** settle the f16-wire exactness definition (§2), and repeat the 3- vs 16-device measurement on real Wi-Fi devices. The loopback emulator shares one GPU, so its per-device slope mixes hop cost with contention.
