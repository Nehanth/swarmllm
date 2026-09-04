# 12 · Stop, fail fast, re-deal: the room survives launch day

**Phase:** now · **Status:** planned

## Why
Two ways a room dies on Monday, both permanent. A friend closes their tab mid-answer: `conn.on("close")` (room.js:190–198) only removes the card, so the host waits out the 30 s / 90 s lap timeouts (865, 930, 968), `aiGenerate`'s catch leaves `ai.engine` set (1029–1033), `aiStart` early-returns on `ai.engine` (715), and nothing re-enables the start button (only `updateNeed` when `!ai.engine`, or the load-failure path at 805). Everyone reloads and re-types the code; guests whose host left still read "cluster online". Or a wrong-direction answer: the decode loops run to EOS or a literal 400 tokens (999, 1016) with no abort path, and at 3.5–6 tok/s cross-network that locks every screen behind `ai-busy` (1156) for up to two minutes. Roadmap 03 (spare copies, replay) is the right end state but is weeks away; this is the floor it sits on, and the master plan's NEXT metric ("median room survives one peer departure") is unreachable without it.

## Design
- **Stop.** An `ai.abort` flag checked between `specStep` iterations and between `aiPipeToken` calls; a Stop button replaces Send while busy on every screen; guests send `ai-stop`, honoured from the current asker or the host. The host still emits `ai-gendone` (`stats: "stopped at N tok"`) so every box unlocks, drops any in-flight `ai-hiddenret-b` by position (the `waiters` map already keys by pos) and leaves `ai.pos` consistent, which matters once item 13 keeps state across turns.
- **Fail fast on departure.** On `close`, if the peer is in `ai.chain` or is `ai.hostId`, reject every outstanding `ai.waiters` entry immediately with "X left (layers a–b)" and end the generation cleanly; drop the peer from `ai.chain` so `aiMaybeReady` can still fire when someone leaves during download.
- **Degraded → re-deal.** After a chain failure the room enters a "degraded" state. The host shows "Re-deal layers": the split planner in `aiStart` is factored into `planSplit()` and re-run over the remaining pledges, fresh `ai-load`s go out (cached ranges reload in seconds), and the host keeps the conversation text so it can re-prefill. The start button is re-enabled in the catch path regardless.
- **Guest side.** When the host connection closes, replace the pane with "the host left; this room is over" and a Create-room button.
- **Read `died`.** The joiner already sends `hello.died` from the localStorage crumb (361–362) and the `hello` handler ignores it; surface it as "phone came back; its tab was killed 40 s ago while streaming blk.30" so people learn that backgrounding Safari kills the peer.
- `ai-stop` is a new message and "degraded" is a new room state, so this file is the GOVERNANCE design note; `docs/protocol.md` gets a row for each. Roadmap 03 then adds spares and automatic replay on top of the same degraded → re-deal state and should be reworded to say so.

## Done when
- In a three-device room, closing one tab mid-answer surfaces the failure within 2 s, offers "Re-deal", and the next question is answered without anyone reloading.
- Pressing Stop on any screen ends generation within one lap and every Send box unlocks.
- A guest whose host left sees "this room is over" rather than "cluster online".
- `docs/protocol.md` documents `ai-stop` and the degraded state. Fail-fast, the guest message and the start-button fix are ordinary bug-fix PRs and land first.
```

### `roadmap/13-conversation.md`

```markdown
