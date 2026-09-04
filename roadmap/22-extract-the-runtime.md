# 22 · Extract the runtime: `engine/generate.js`, `room/pipeline.js`, request/stream contract, headless multi-peer driver

**Phase:** next · **Status:** planned · _next · weeks · high_

## Why

Merges: *engine/generate.js: DOM-free generation core + swarmllm run*; *Room-in-a-box: pluggable transport + headless driver*; *Request and stream contract on ai-ask*; *examples/ + docs/engine-api.md*; *@swarmllm/client* as a follow-on.
The only code that turns a prompt into tokens is `aiGenerate`, which reads the DOM, calls PeerJS and mutates the chat (880–1036); `benchmarks/bench.js` re-implements the loop; `tests/test_mtp_split.js` simulates the protocol in-process. Roadmap 04 (bridge), 11 (native peer), master plan NOW #5 (solo landing demo) and NEXT #9 (SDK) all silently depend on a programmable generator and a wire contract carrying more than `{text, name}`; 09 needs a latency-injecting harness; and the K=7 Mac lock-up was found by a user because nothing exercises the runtime headless. A byte-identical refactor gated on the existing goldens, with the transport injected the way `spec.runTrunk` already is, unblocks four items at once. The SDK and examples follow from it and are not separate slots.

## Design

See the merged proposals in [docs/roadmap-review.md](../docs/roadmap-review.md) under item 22; turn them into a design note before building.

## Done when

- Acceptance criteria to be written with the design note.
