# 10 · Expert-split mixture-of-experts across a room

**Phase:** later · **Status:** research

## Why
Mixture-of-experts models (e.g. Qwen 3.5 122B-A10B) activate a fraction of their weights per token and are built from many "experts" that split naturally across devices. A 122B model no consumer device can hold, running in browser tabs, is the second demo people will not believe.

## Design sketch
- Expert-parallel: each device holds a subset of experts for a range of layers; the host runs the router (it already runs sampling); tokens are dispatched to the devices holding the chosen experts.
- Composes with the existing layer split; GGUF stores experts as contiguous tensors, so range-fetch works unchanged.
- Placement by measured latency and memory; expert popularity is skewed, so replicate hot experts.

## Done when
- A MoE model runs across a room with output matching a single-device reference.
