# 18 · Download path: Hugging Face backoff and mirrors, parallel range streams, phone cache

**Phase:** now · **Status:** planned · _now · days · high_

## Why

Merges: *Hugging Face fetch resilience*; *Weight download: parallel range streams, compile overlap*; *Phone weight cache: chunked Cache API entries*.
`rangeFetch` throws on any non-206 with no retry (444), one tensor is fetched at a time per device (engine/gguf.js sequential `await entry()`), phones sleep 300 ms per tensor and skip the cache store entirely (445, 603) so every phone rejoin re-downloads ~470 MB, and nothing measures MB/s. An office or classroom behind one NAT pulling 15 GB from HF is the other half of "I joined and nothing happened" that 01 does not cover. A pinned-revision R2 mirror is the same class of infra as 01's VPS; roadmap 05's peer sharing becomes the LAN mirror later. Sits directly after 01 as a sibling launch-readiness item.

## Design

See the merged proposals in [docs/roadmap-review.md](../docs/roadmap-review.md) under item 18; turn them into a design note before building.

## Done when

- Acceptance criteria to be written with the design note.
