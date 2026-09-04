# 19 · Pinned model revisions, weight integrity, and cache management

**Phase:** now (pinning, stamps, persist) / next (manager UI, per-tensor hashes) · **Status:** planned · _now (pinning, stamps, persist) / next (manager UI, per-tensor hashes) · days · high_

## Why

Merges: *Weight cache versioning*; *Weight integrity: pin to a commit and verify hashes*; *Model manager: cached list, delete, persist, pre-download*.
All 13 URLs in room/models.js are `resolve/main`, the cache key is URL + range and validity is `x-swarm-len` only (429, 438), the 27B comes from a community account, and `navigator.storage.persist()` is never called. A routine upstream re-upload makes a returning device mix old and new ranges and produce garbage while every local golden passes: a field failure of non-negotiable 4 with no way to detect it, plus a surprise 15 GB re-download when Chrome evicts unpersisted storage. SECURITY.md and master plan §6 both already promise the hash ("the exact GGUF SHA runs, displayed per session"). Items 03, 05 and 07 all assume the cache is correct and durable, so this is their silent prerequisite.

## Design

See the merged proposals in [docs/roadmap-review.md](../docs/roadmap-review.md) under item 19; turn them into a design note before building.

## Done when

- Acceptance criteria to be written with the design note.
