# Roadmap

One file per item. Each has a status, the reason it matters, a design sketch, and what "done" means, so anyone can pick one up. Order within a phase is priority order. The long-form reasoning behind these choices is in [docs/master-plan.md](../docs/master-plan.md).

| # | Item | Phase | Status |
|---|---|---|---|
| 01 | [Relay fallback for strict NATs](01-relay-fallback.md) | now | planned · [#1](https://github.com/Nehanth/swarmllm/issues/1) |
| 02 | [Prefill GEMM](02-prefill-gemm.md) | now | landed (Q4) · Q8 variant next · [#2](https://github.com/Nehanth/swarmllm/issues/2) |
| 03 | [Spare layer copies and swarm recovery](03-swarm-recovery.md) | next | planned · [#3](https://github.com/Nehanth/swarmllm/issues/3) |
| 04 | [`npx swarmllm serve`: an OpenAI-compatible local endpoint](04-local-endpoint.md) | next | planned · [#4](https://github.com/Nehanth/swarmllm/issues/4) |
| 05 | [Offline rooms: hotspot + QR signaling + peer weight sharing](05-offline-rooms.md) | next | planned · [#5](https://github.com/Nehanth/swarmllm/issues/5) |
| 06 | [More models on the hybrid engine (Qwen 3.5 2B/9B, 3.6 27B)](06-more-models.md) | next | planned · [#6](https://github.com/Nehanth/swarmllm/issues/6) |
| 07 | [Persistent rooms and contribution stats](07-rooms-and-stats.md) | next | planned · [#7](https://github.com/Nehanth/swarmllm/issues/7) |
| 08 | [Randomly audited compute](08-verification.md) | later | research · [#8](https://github.com/Nehanth/swarmllm/issues/8) |
| 09 | [Overlapping laps across the network (PipeInfer-style)](09-lap-overlap.md) | later | research · [#9](https://github.com/Nehanth/swarmllm/issues/9) |
| 10 | [Expert-split mixture-of-experts across a room](10-moe-expert-split.md) | later | research · [#10](https://github.com/Nehanth/swarmllm/issues/10) |
| 11 | [Native peer for headless GPUs](11-native-peer.md) | later | planned · [#11](https://github.com/Nehanth/swarmllm/issues/11) |
| 12 | [Stop, fail fast, re-deal: the room survives launch day](12-room-survives-launch-day.md) | now | planned · [#12](https://github.com/Nehanth/swarmllm/issues/12) |
| 13 | [Multi-turn conversation and an honest context limit](13-multi-turn-context.md) | now | planned · [#13](https://github.com/Nehanth/swarmllm/issues/13) |
| 14 | [Pre-flight check, join links, and a model ladder that says what this room can run](14-preflight-and-join-links.md) | now | planned · [#14](https://github.com/Nehanth/swarmllm/issues/14) |
| 15 | [Self-hosted signaling, vendored PeerJS, and a status canary](15-self-hosted-signaling-and-status.md) | now | planned · [#15](https://github.com/Nehanth/swarmllm/issues/15) |
| 16 | [Fail loudly and legibly: actionable errors, a diagnostic report, build and protocol version](16-actionable-errors-and-versioning.md) | now | planned · [#16](https://github.com/Nehanth/swarmllm/issues/16) |
| 17 | [Security sweep before Show HN: host-authoritative protocol, honest SECURITY.md, hardened page](17-security-sweep.md) | now | planned · [#17](https://github.com/Nehanth/swarmllm/issues/17) |
| 18 | [Download path: Hugging Face backoff and mirrors, parallel range streams, phone cache](18-download-path.md) | now | planned · [#18](https://github.com/Nehanth/swarmllm/issues/18) |
| 19 | [Pinned model revisions, weight integrity, and cache management](19-pinned-revisions-and-cache.md) | now | planned · [#19](https://github.com/Nehanth/swarmllm/issues/19) |
| 20 | [Public demo room: ask-only guests, question queue, quotas](20-public-demo-room.md) | now | planned · [#20](https://github.com/Nehanth/swarmllm/issues/20) |
| 21 | [Reproducible benchmarks and the paper's evidence: protocol, `/bench` page, device matrix, per-hop telemetry](21-reproducible-benchmarks.md) | next | planned · [#21](https://github.com/Nehanth/swarmllm/issues/21) |
| 22 | [Extract the runtime: `engine/generate.js`, `room/pipeline.js`, request/stream contract, headless multi-peer driver](22-extract-the-runtime.md) | next | planned · [#22](https://github.com/Nehanth/swarmllm/issues/22) |
| 23 | [Contributor on-ramp: labels, seeded issues, no-GPU track, hardware-verifier role](23-contributor-on-ramp.md) | now | planned · [#23](https://github.com/Nehanth/swarmllm/issues/23) |
| 24 | [Close the Mac gap](24-close-the-mac-gap.md) | now | planned · [#24](https://github.com/Nehanth/swarmllm/issues/24) |
| 25 | [Cross-network quick wins: draft cache, frame chunking, prefill acks, per-hop telemetry](25-cross-network-quick-wins.md) | now | planned · [#25](https://github.com/Nehanth/swarmllm/issues/25) |
| 26 | [Host-side overhead: encode-ahead, one-submit speculation, GPU sampling, fused glue](26-host-side-overhead.md) | next | planned · [#26](https://github.com/Nehanth/swarmllm/issues/26) |
| 27 | [Placement, chain order and host election](27-placement-and-chain-order.md) | next | planned · [#27](https://github.com/Nehanth/swarmllm/issues/27) |

Items 12–23 came from the [roadmap gap review](../docs/roadmap-review.md); 12, 13 and 17 contain launch-week bug fixes and go first. Items 24–27 came from the [performance research round](../docs/research/), whose four designs live in `docs/research/`.

**Explicitly not on the roadmap** (see [GOVERNANCE.md](../GOVERNANCE.md)): accounts, tokens or ads, open swarms of strangers by default, distributed training, noise/permutation "privacy" features, frontier-scale natives that cannot fit a room. Native and headless peers *are* on it (11); the browser path just never stops being enough on its own.

## Status meanings
- **research**: needs a design note before code; open an issue with the *Research / design proposal* template.
- **planned**: design agreed in the item file; ready to build.
- **in progress**: someone is on it (named in the file).
