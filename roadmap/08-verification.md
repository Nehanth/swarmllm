# 08 · Randomly audited compute

**Phase:** later · **Status:** research

## Why
Nothing today detects a peer returning wrong activations. Between friends that is acceptable; any wider trust model needs at least probabilistic auditing. Petals proposed spot-checks and never shipped them.

## Design sketch
- Precompute reference activations for canary inputs per model and layer range; the host occasionally interleaves a canary token and compares the returned activation within a relative-L2 tolerance (never exact: GPUs and autotune shapes differ).
- Report "randomly audited" in the room; never claim "verified" or "trustless". Fingerprint-based checks (TOPLOC-style) as a second step.
- Needs a written security argument before implementation (issue with the research template).

## Done when
- A peer returning corrupted activations is flagged within N tokens with a documented false-positive rate.
