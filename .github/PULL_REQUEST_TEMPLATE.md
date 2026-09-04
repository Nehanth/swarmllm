## What

<!-- One paragraph: what changes and why. Link the issue. -->

## Correctness

- [ ] Golden tests still pass bit-exactly (`npm run test:gpu` / `npm run test:q38`), or this PR adds a documented, default-off approximation switch.
- [ ] Speculative and plain decoding still produce identical streams (`tests/test_mtp.js`), if the engine changed.

## Performance (delete if not applicable)

- [ ] Added a row to `docs/bench-log.md` with hardware, commit, and before/after numbers (neutral results included).

## Tested on

<!-- Browser/runtime, OS, GPU, model. e.g. "Deno 2.9 on a GB10 (Vulkan); Chrome 140 on M4 Pro". -->
