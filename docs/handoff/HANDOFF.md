# Handoff: SwarmLLM → "Tabby" (working name)

Branch `tabby-new-idea`: everything from the Claude Code session in one place, for picking up in
another session. The full conversation is in [chat-export.md](chat-export.md); the loop logs are
[kernels-loop.md](kernels-loop.md) and [tabby-loop.md](tabby-loop.md).

## The idea

SwarmLLM runs a big model (Qwen 3.8 27B) across several people's devices in browser tabs
(WebGPU compute, WebRTC between tabs, layers split over the devices). As a general chatbot it is
not very useful. The new direction:

**Tabby (not the final name): a peer-to-peer coding harness in the browser.** You open a tab, I
open a tab, we pool a model together, and coding agents run in the tabs on the hardware people
already own. Private (code never leaves the devices), no API bill, nothing to install.

- One-liner: "Two laptops. Two tabs. One free, private coding agent."
- Category line: "a peer-to-peer coding harness that runs in your browser".
- Name: "Tabby" clashes with TabbyML (an existing open-source coding assistant) and the Tabby
  terminal; alternatives discussed: Tabpool, Tabmates, Potluck, Quorum.
- SwarmLLM would be deprecated; its inference engine becomes the base of the new project.

## Models

- **Qwen3.8-27B (dense)**: what the engine runs today; best coding quality that fits two 24 GB
  Macs (~16.5 GB at 4-bit). SWE-bench Pro 61.7%.
- **Qwen3.6-35B-A3B (MoE, 3B active)**: ~4x faster per token in principle, 4-20 points weaker on
  coding benchmarks (SWE-bench Verified 73.4%, Pro 49.5%). ~21 GB at 4-bit, fits two 24 GB Macs.
  Engine support now built (see below). Plan: 27B for quality, A3B as the fast mode; switch to a
  Qwen 3.8 MoE if one ships (none officially as of Sept 2026).
- Bigger: Qwen3.5-122B-A10B (~69 GB at 4-bit: two 64 GB Macs, or 4+ smaller devices),
  GLM-5.3-Flash (320B/18B active, strongest open coder), too big for two 24 GB Macs.
- Speed: nothing has been timed on a real GPU (the dev container only has SwiftShader).
  Bandwidth estimates: 27B ~10-30 tok/s, 35B-A3B ~40-80 tok/s on a good Mac. ~60 tok/s is
  plausible only for the MoE.
- SSD: conversation state caching to disk (OPFS) is built and is a big win for resuming
  (~minutes of re-prefill vs ~1-2 s). Streaming *weights* from SSD (the Edge0 trick from the
  @SamuelZengML post) only makes sense for MoE and only when a model does not fit in the room's
  memory; not needed for the 35B-A3B on two 24 GB Macs.

## What is built (all on this branch)

Details: [../tabby-kernel.md](../tabby-kernel.md), research: [../research/](../research/).

Engine (WebGPU, `engine/qwen35.js`, `engine/wgsl/*`):
- Multi-device gibberish fixes (dense ROWS=8 half-dispatch, worker DeltaNet reset, spec flag past
  the first worker, rollback ordering).
- Kernel work, bit-identical unless noted: register-resident DeltaNet, workgroup softmax, fused
  attention glue, fused DeltaNet delta + gated norm, batched attention, Q8 prefill GEMM, replay
  rollback, batched draft-cache fill, one-submit draft chain (opt-in).
- Long context: f16 KV cache + split-K flash attention (27B rooms: 8192 tokens, any maxSeq
  works); int8 KV option (`?kv=q8`); two-column tiled prefill attention.
- Sessions: exportState / importState, GPU slots, OPFS state cache, room checkpoints (regenerate /
  branch resumes from the last answer on every device; protocol 4).
- Prompt-lookup drafts up to 15 tokens.
- **MoE (`qwen35moe`)**: router, top-k experts, shared expert (`engine/wgsl/moe.js`), loader for the
  real GGUF layout, one-token and batched paths.

Harness (`harness/`): tool-call format + streaming parser (Qwen XML and JSON), tool-name
constraint while sampling, prefix index, session manager (GPU slots, spill to disk), workspace
(a local folder via the File System Access API), coding tools (list / read / search / edit /
write), agent loop (approval for edits, context budget), engine adapter with prefix reuse.

## Test status at handoff

All on the 27B-shaped / small synthetic models under SwiftShader (no GPU here):
- Validated together (the `tabby-work/agent2` head, i.e. everything except MoE): unit 54/54,
  engine_synth 45/45, fusion, drafts, agent, sessions, state, workspace: all pass. The 3-device
  room run with regenerate + checkpoint reuse was still running when the session was stopped
  (each piece passed its own room run earlier).
- MoE: `tests/e2e/moe_kernels.mjs` and `tests/e2e/moe_synth.mjs` pass (each layer's MoE FFN vs a
  float64 reference ~4e-7; spec == plain; batched prefill bit-identical). `engine_synth.mjs --moe`
  (the full suite on a MoE model) had not finished when the session was stopped: run it first.
- `tabby-kernel` on GitHub has the work up to the session manager; this branch has everything.

## How to run the tests (what the Claude Code session used)

No GPU needed: headless Chromium with SwiftShader WebGPU.
- `npm i playwright@1.49.1 peerjs@1.5.4 peer jsqr deno` somewhere, then `NODE_PATH=<that>/node_modules`.
- Chromium: `/opt/pw-browsers/...` or any Chromium; flags in `tests/e2e/engine_synth.mjs` (GPU_ARGS).
- Unit: `deno test --no-check tests/unit/{prefix,tools,constrain,agent,engine_model,room,transport,visibility}_test.js`
- Engine: `node tests/e2e/engine_synth.mjs [--q8] [--moe]` (~20-25 min under SwiftShader).
- Focused: `fusion_synth`, `flash_synth [--q8] [--prompt-len 2300 --max-seq 4096]`, `state_synth [--q8]`,
  `sessions_synth`, `drafts_synth`, `agent_synth`, `workspace_browser`, `moe_kernels`, `moe_synth`.
- 27B shapes: `node tests/e2e/synth.mjs out.gguf --shape 27b` then `--model out.gguf` on any test.
- Rooms: `node tests/e2e/room_synth.mjs --model <gguf> --compare --devices 3 --rounds 2 --greedy --regen --expect-reuse`.
- Gotchas: PeerServer needs `--host 127.0.0.1` in IPv6-less containers; do not edit engine files
  while a room test is loading tabs; weights over ~100 MB must be served over HTTP, not
  `route.fulfill`.

## Next steps (agreed direction)

1. Measure on two real Macs: the 27B in a room (tok/s, `?fuse=0` A/B, short vs long context).
2. Finish MoE: run `engine_synth --moe` and a MoE room test, then try the real Qwen3.6-35B-A3B
   "Q4_0" GGUF (bartowski, ~21 GB) on two Macs; then expert-parallel splitting, grouping a
   batch by expert, SSD spill of cold experts.
3. Build the Tabby app page on the harness: pick a folder, agent chat, diff approval, sessions.
4. Designs written, not built: pipelined speculative windows across devices, batched
   multi-session decode ([../research/tabby-next-2026-09.md](../research/tabby-next-2026-09.md)).
5. Pick the name; decide new repo vs rename; deprecate SwarmLLM.
