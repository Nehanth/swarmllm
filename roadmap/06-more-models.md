# 06 · More models on the hybrid engine

**Phase:** next · **Status:** planned

## Why
Qwen 3.5 2B and 9B and Qwen 3.6 27B share Qwen 3.8's architecture (`qwen35` family), so they are configuration entries, not engine work. 9B fills the gap between 4B and 27B; 2B gives weak devices a real role; 3.6-27B is a strong general model.

## Design
- Add `MODELS` entries (URL, layer count, bytes per layer) in `room/models.js`; verify tensor names against `gguf.js`; generate goldens with `tests/reference/`.
- Distilled variants of larger models (e.g. DeepSeek-style distills onto Qwen 3.5 9B) come for free once 9B works.

## Done when
- Each model has a golden test and a bench-log row on at least one device.
