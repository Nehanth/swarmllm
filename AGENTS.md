# Working on SwarmLLM with an AI agent

Instructions for coding agents (Claude Code, Codex, Cursor, …). Humans: see [CONTRIBUTING.md](CONTRIBUTING.md).

## What this is
A from-scratch WebGPU inference engine (`engine/`) and a browser room runtime (`p2p.html` + `room/`) that split one LLM's layers across devices over WebRTC. No build step, no framework, ES modules only. Read [docs/architecture.md](docs/architecture.md) first, then [docs/kernels.md](docs/kernels.md).

## Non-negotiables (from GOVERNANCE.md)
- Output must stay **bit-exact**: golden tests must pass, and the speculative stream must equal plain decoding (`tests/test_mtp.js`). Approximations need a default-off switch.
- Every performance claim gets a row in `docs/bench-log.md` with hardware + commit, including neutral results.
- Zero install, no accounts/tokens/ads, honest privacy claims (`SECURITY.md`).

## How to run things
```bash
npm test                              # unit tests, no GPU (CI runs these)
cd tests && deno run --unstable-webgpu --allow-read --allow-env test_qwen.js   # one GPU test
tests/run.sh quick | q38 | all        # suites; GPU tests read models from ../models/
cd benchmarks && MODEL=q38 deno run --unstable-webgpu --allow-read --allow-env bench.js
```
Model files live in `models/` (git-ignored). The 27B loads in ~2 minutes; plan runs accordingly.

## Rules learned the hard way (do not relearn them)
- **Never run two GPU jobs at once** (tests, benches, or a CUDA build): the GB10 shares memory bandwidth and every number becomes garbage.
- Do not edit `engine/*.js` while a queued test chain is still starting new processes; each process loads the files from disk when it starts.
- A shell loop that waits with `pgrep -f "<name>"` matches its own command line and waits forever. Wait on a completion line in a log instead.
- `dispatchWorkgroups(n)` with n > 65,535 is silently dropped; use the 2-D form (`_dop` does this).
- Dynamically indexed local arrays in WGSL spill (3× slower); generate unrolled code with literal indices. Barriers stay outside conditionals. Accumulate in f32.
- Feature-detect (`probeUnpack`, subgroup probes); Safari 26 has no subgroups and no storage-pointer function parameters.
- Profile by skipping kernel families (`benchmarks/bench_breakdown.js`); timestamp queries cost 3× just by being on.
- Vercel refuses deploys whose commits have an email that isn't on GitHub; `git config user.email` must match.
- The room page is served at `/room`; `curl` the alias's `/room`, not `/p2p.html`, to verify a deploy.

## Layout
`engine/engine.js` is a barrel; implementation is in `engine/{dense,qwen35,gguf,tokenizer,sampling,quant,autotune,selftest,safetensors}.js` and `engine/wgsl/*.js`. Tests in `tests/` (goldens in `tests/golden`, CPU references in `tests/reference`, no-GPU tests in `tests/unit`). Benchmarks in `benchmarks/`. Docs in `docs/`.

## Style
Plain modern JS, two-space indent, no reformatting of untouched code, descriptive commit subjects with the measured effect for performance work.
