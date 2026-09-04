# CLAUDE.md

See [AGENTS.md](AGENTS.md) for the full agent guide; the short version:

- Bit-exact output is the contract: run `tests/run.sh quick` (and `q38` for engine changes) before claiming a change works; `tests/test_mtp.js` must still say the speculative stream is identical.
- One GPU job at a time. Model files are in `models/`; GPU tests run from the `tests/` directory.
- Performance changes add a row to `docs/bench-log.md` (hardware, commit, before/after, neutral results included).
- Kernel rules: no dynamically indexed local arrays, barriers outside conditionals, f32 accumulation, 2-D dispatch for tall matvecs, feature-probe anything not in core WebGPU.
- Never push to `main` without being asked (it deploys production). Branch pushes get automatic preview URLs.
- Commit author email must be the GitHub email or Vercel blocks the deploy.
