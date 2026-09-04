# Roadmap

One file per item. Each has a status, the reason it matters, a design sketch, and what "done" means, so anyone can pick one up. Order within a phase is priority order. The long-form reasoning behind these choices is in [docs/master-plan.md](../docs/master-plan.md).

| # | Item | Phase | Status |
|---|---|---|---|
| 01 | [Relay fallback for strict NATs](01-relay-fallback.md) | now | planned |
| 02 | [Prefill GEMM](02-prefill-gemm.md) | now | in progress (prototype at 1.25×) |
| 03 | [Spare layer copies and swarm recovery](03-swarm-recovery.md) | next | planned |
| 04 | [`npx swarmllm serve`: an OpenAI-compatible local endpoint](04-local-endpoint.md) | next | planned |
| 05 | [Offline rooms: hotspot + QR signaling + peer weight sharing](05-offline-rooms.md) | next | planned |
| 06 | [More models on the hybrid engine (Qwen 3.5 2B/9B, 3.6 27B)](06-more-models.md) | next | planned |
| 07 | [Persistent rooms and contribution stats](07-rooms-and-stats.md) | next | planned |
| 08 | [Randomly audited compute](08-verification.md) | later | research |
| 09 | [Overlapping laps across the network (PipeInfer-style)](09-lap-overlap.md) | later | research |
| 10 | [Expert-split mixture-of-experts across a room](10-moe-expert-split.md) | later | research |
| 11 | [Native peer for headless GPUs](11-native-peer.md) | later | planned |

**Explicitly not on the roadmap** (see [GOVERNANCE.md](../GOVERNANCE.md)): accounts, tokens or ads, open swarms of strangers by default, distributed training, noise/permutation "privacy" features, frontier-scale natives that cannot fit a room.

## Status meanings
- **research**: needs a design note before code; open an issue with the *Research / design proposal* template.
- **planned**: design agreed in the item file; ready to build.
- **in progress**: someone is on it (named in the file).
