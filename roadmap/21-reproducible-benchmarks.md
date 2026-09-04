# 21 · Reproducible benchmarks and the paper's evidence: protocol, `/bench` page, device matrix, per-hop telemetry

**Phase:** next (Oct 30 hard date) · **Status:** planned · _next (Oct 30 hard date) · weeks · high_

## Why

Merges: *Reproducible benchmark protocol*; *Mac/Metal profile and roofline*; *Load time and TTFT as bench-log columns*; *Per-hop lap telemetry and throughput-weighted split*; *Phone-class peers: GB/s, thermal*; *Heterogeneous device matrix*; *Baseline comparison harness*; *Ablation suite*; *Output-equivalence at 27B*; *Scaling and system-cost experiments*; the schema/copy-report half of *Hardware report bundle and community benchmark database*; the matrix half of *Phone and tablet device-check page*.
Thirteen proposals describe one gap: every number comes from one GB10 and one MacBook, hand-transcribed into a table (one row is "partly contaminated"), from a single "quick brown fox" prompt, with no load/TTFT, phone, Safari, discrete-GPU, per-hop or thermal column, no external llama.cpp equivalence at 27B, and roadmap 09 explicitly gated on per-hop data the host does not collect (only a whole-lap EMA at 974). Non-negotiable 3 is currently enforced by discipline, not tooling. One fixed prompt set, one JSON schema with the token-id hash, a `/bench` page anyone can run from a URL, per-hop timestamps in frames, and a generated bench-log give the MLSys artifact, the device matrix and the community's "reports from hardware we don't have" path in one build.

## Design

See the merged proposals in [docs/roadmap-review.md](../docs/roadmap-review.md) under item 21; turn them into a design note before building.

## Done when

- Acceptance criteria to be written with the design note.
