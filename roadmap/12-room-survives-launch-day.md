# 12 · Stop, fail fast, re-deal: the room survives launch day

**Phase:** now · **Status:** in progress (PR: re-deal on leave/join/manual, single-turn; guest Stop button still open)

## Why
Two ways a room dies on Monday, both permanent. A friend closes their tab mid-answer: `conn.on("close")` (room.js:190–198) only removes the card, so the host waits out the 30 s / 90 s lap timeouts (865, 930, 968), `aiGenerate`'s catch leaves `ai.engine` set (1029–1033), `aiStart` early-returns on `ai.engine` (715), and nothing re-enables the start button (only `updateNeed` when `!ai.engine`, or the load-failure path at 805). Everyone reloads and re-types the code; guests whose host left still read "cluster online". Or a wrong-direction answer: the decode loops run to EOS or a literal 400 tokens (999, 1016) with no abort path, and at 3.5–6 tok/s cross-network that locks every screen behind `ai-busy` (1156) for up to two minutes. Roadmap 03 (spare copies, replay) is the right end state but is weeks away; this is the floor it sits on, and the master plan's NEXT metric ("median room survives one peer departure") is unreachable without it.

## Design (as shipped on the re-deal PR; `docs/protocol.md` § Room states is the reference)
- **Fail fast on departure.** A chain member's leave (PeerJS close, ICE `failed`, ICE `disconnected` > 1.5 s, 7.5 s of silence idle, 2 s of silence while an answer is in flight) rejects every outstanding lap with `stopped: <name> left (layers a–b)`; the host emits `ai-gendone` with that text so every Send box unlocks, keeps its engine, and re-plans. There is **no `ai-stop` message** in v1: the stop is local to the host and the room learns it through `ai-gendone`.
- **Re-deal, not "degraded".** The room states are `redealing` (a new plan is out, devices reload their deltas) and `waiting` (the survivors cannot hold the model). `planSplit()` in `room/plan.js` runs over the remaining pledges with the previous plan as `prev` (stable order, host pin), versioned `ai-load {v}` / `ai-wait {v}` go out, and the chain serving an answer is frozen until `ai-gendone`. Joins and the host's re-deal button run the same planner; the button reads "re-deal after this answer" while one is in flight. Single turn: every answer starts from `ai-reset`, nothing is re-prefilled.
- **Guest side.** A joiner whose host link closed, failed at ICE, or went silent for 7.5 s shows "the host left — this room is over" (`body[data-state]="over"`), ends the answer on screen with "stopped: the host left" and unlocks Send.
- **Read `died`.** The `hello` handler logs "X came back — its tab died 40 s ago during …" from the localStorage crumb.
- **Stop button** (still open): an `ai.abort` flag checked between `specStep` iterations and between `aiPipeToken` calls; Stop replaces Send while busy on every screen; guests send `ai-stop`, honoured from the current asker or the host; the host emits `ai-gendone` (`stats: "stopped at N tok"`). `ai-stop` is a new message, so that follow-up carries its own GOVERNANCE note and `docs/protocol.md` row. Roadmap 03 adds spares and automatic replay on top of the same `redealing` state.

- **Automatic re-deal, prepared in the background, flipped at the next question.** The chain that is serving an answer is frozen for that answer. Joins and leaves both trigger a re-plan (`planSplit()` over measured time per layer and memory, not pledged memory alone):
  - *Join:* the newcomer takes a fair share and the others give up layers, so every device gains headroom (this is what "more devices scale what fits" means; each extra hop costs a little speed and the docs say so). The newcomer and any device gaining layers preload their new slices while the current answer streams; when every preload reports ready the host flips the plan at the next question and re-prefills the conversation on the new chain. Devices that gave up layers free the GPU memory but keep the bytes in the Cache API, which makes them warm spares.
  - *Leave:* the answer in flight stops (its layers are gone). The departed range goes first to whoever still has it cached, reloaded in seconds, and only hits the network if nobody does. If a spare already holds that slice (roadmap 03) the host flips to it directly. The next question works without anyone reloading.
  - *Manual re-deal button* on the host for "find the best split now", same planner, for when the automatic one was skipped or the room changed shape.

## Done when
- [x] In a three-device room, closing one tab mid-answer surfaces the failure within 2 s (emulator: `stopAfterMs` ≈ 2.1–2.2 s), the room re-deals on its own (the button is there for "find the best split now"), and the next question is answered without anyone reloading.
- [ ] Pressing Stop on any screen ends generation within one lap and every Send box unlocks.
- [x] A guest whose host left sees "this room is over" rather than "cluster online" (emulator: `--leave host --expect-over`).
- [x] A device joining a three-device room is serving a slice by the next question (emulator: `--join-after`); a join during an answer never changes that answer's output (the chain is frozen; the plan is applied after `ai-gendone`).
- [x] A device leaving mid-answer stops that answer within 2 s; the next question is answered by the remaining devices, with the departed range reloaded from cache where any device still has it (survivors keep their order, so only the delta loads).
- [x] `docs/protocol.md` documents the room states (`redealing`, `waiting`, `over`) and the versioned plan messages. `ai-stop` is documented with the Stop-button follow-up.
```

### `roadmap/13-conversation.md`

```markdown
