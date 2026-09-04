# Contributing to SwarmLLM

Thanks for helping. SwarmLLM is a from-scratch WebGPU inference engine plus a peer-to-peer runtime that runs large models across browser tabs. Most useful contributions fall into: kernels, model support, the room protocol, the site, docs, and benchmark reports from hardware we do not have.

## Before you start

- Read [GOVERNANCE.md](GOVERNANCE.md) for what needs a design note first, and [SECURITY.md](SECURITY.md) for the trust model.
- Search existing issues. For anything beyond a small fix, open an issue before writing code so the approach can be agreed on.
- One change per pull request. Keep refactors separate from behavior changes.

## AI usage policy

Contributions written with AI assistance are welcome. You are responsible for the code you submit: read it, run it, and be able to explain it in review. Pull requests that show no sign of having been run (failing tests, invented APIs, unverified performance claims) will be closed. Do not paste raw model output into issues or reviews.

## Development setup

The engine runs in browsers and, for testing, in [Deno](https://deno.com) (which ships WebGPU). You need a GPU with WebGPU support to run most tests.

```bash
git clone https://github.com/Nehanth/swarmllm && cd swarmllm
curl -fsSL https://deno.land/install.sh | sh          # Deno 2.x
npm test                                               # quick unit tests (no GPU)
```

Model files are not in the repository. Put GGUFs under `models/` (see [docs/models.md](docs/models.md)); the full test suite and benchmarks expect `models/qwen/model.gguf` (Qwen3 0.6B, Q8_0) and `models/q38/model.gguf` (Qwen 3.8 27B, Q4_0).

```bash
npm run test:gpu          # golden tests on the small model
npm run test:q38          # 27B tests (needs ~16 GB of GPU memory)
npm run bench             # decode/prefill tok/s
```

To run the site locally, serve the repository root with any static server (`npx serve .`) and open `/room`.

## Pull requests

**Preparing your PR**

- Kernel and engine changes must keep the golden tests passing **bit-exactly**. The speculative decoding path must produce a stream identical to plain decoding (`tests/test_mtp.js`). If a change is intentionally approximate, it needs a switch that defaults to off, and a note in the PR.
- Performance changes must add a row to [docs/bench-log.md](docs/bench-log.md) with the hardware, commit, and before/after numbers, including neutral results. "Faster" without a number is not reviewable.
- Protocol changes must keep older peers failing loudly rather than silently (bump the message version).
- Add or update a test. GPU tests live in `tests/`, unit tests that need no GPU in `tests/unit/`.
- Keep the diff readable: no reformatting of untouched code, no drive-by renames.

**Commit messages**: imperative subject line under 72 characters, a body that says *why*, and the measured effect for performance work.

**After submitting**: CI runs syntax checks and the unit tests, and Vercel posts a preview URL for your branch on the PR (every branch gets `swarmllm-git-<branch>-nehanths-projects.vercel.app`). GPU tests run on the maintainer's hardware before merge; say which tests you ran locally and on what GPU. Merging to `main` deploys production.

## Coding guidelines

- Plain modern JavaScript (ES modules), no build step, no framework. The site must work by opening the HTML files.
- WGSL kernels are generated from JavaScript templates in `engine/engine.js` when they vary by shape; keep generated code deterministic and readable.
- Kernel rules learned the hard way (see [docs/kernels.md](docs/kernels.md)): never dynamically index a local array (it spills), keep `workgroupBarrier()` outside conditionals, accumulate in f32, no subgroup or pointer-parameter features without a probe and a fallback, and dispatch tall matvecs in 2-D (a single dimension over 65,535 workgroups is silently dropped).
- Everything must run on Chrome, Safari 26, and Deno. Feature-detect at runtime; never assume an extension.
- Prefer measuring to reasoning: `benchmarks/bench_breakdown.js` skips kernel families to show where time goes.

## Naming guidelines

- Files: `snake_case.js`; tests `test_<what>.js`; benchmarks `bench_<what>.js`.
- Kernels: `<op>[_<quant>][_<variant>]`, e.g. `matvec_q4_coop_b`, `dn_delta_mc`. Suffixes: `_coop` cooperative rows, `_b` batched columns, `_acc` accumulate into output, `_mc` multi-column glue, `_gu` fused gate/up.
- Protocol messages: `ai-<verb>` with a `t` field.

## Documentation

User-facing behavior changes update the relevant page in `docs/`. New models go in [docs/models.md](docs/models.md).

## Resources

- [docs/architecture.md](docs/architecture.md): how a token flows through the system.
- [docs/kernels.md](docs/kernels.md): the WebGPU engine.
- [docs/protocol.md](docs/protocol.md): the room protocol.
- [docs/bench-log.md](docs/bench-log.md): every performance change with its numbers.
