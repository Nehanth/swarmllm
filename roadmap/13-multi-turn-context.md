# 13 · Multi-turn conversation and an honest context limit

**Phase:** now · **Status:** planned

## Why
The room renders a chat thread, but every Send wipes the model: `aiGenerate` calls `engine.reset()`, sets `ai.pos = 0` and broadcasts `ai-reset` (room.js:884–886), then builds a single bare `user` turn with the think block pre-closed (891–896). "What did I just ask?" fails, which reads as "the model is dumb" rather than "the UI is single-turn", and it is the first thing an HN commenter tries. Nothing enforces the 512-token cap (`MAX_SEQ`, room/models.js:26): a pasted paragraph plus a 400-token answer runs positions past the KV cache with no error; the prompt box is a single-line `<input>` (p2p.html:382) that collapses newlines; answers stop dead at 400 tokens with no notice. The master plan markets 256K context, so a silent 512-token overflow is the opposite of non-negotiable 3. Items 03 (replay needs the token history) and 04 (`messages[]`) both assume conversation state that does not exist; this is their prerequisite.

## Design
- **No reset between turns.** The host keeps `ai.history = [{role, text, ids}]` and the running `ai.pos`. A new turn appends `<|im_end|>\n<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n` and prefills only the new ids; workers already hold the KV and DeltaNet state and advance by frame order, so nothing changes on the wire. `ai-reset` moves behind a visible "new chat" button. First fix the EOS asymmetry: the speculative path breaks on EOS without advancing state (1006) while the plain path pipes it through (1018); both must leave state at `pos` identically.
- **Template builder.** `room/template.js` exporting `buildIds(tok, {system?, messages, thinking})`: the one place item 04's `messages[]`, a later system-prompt setting and a thinking budget plug in, with a unit golden for the rendered ids.
- **`maxSeq` per model.** 48 of 64 layers are constant-memory DeltaNet; only 16 attention layers hold f32 KV at `maxSeq * kvDim * 4` per K and V (engine/qwen35.js:218–219), so 4096 costs tens of MB per attention layer. Put `maxSeq` in the `MODELS` entry, count KV bytes for the attention layers in each device's range inside `aiStart`'s pledge math, and send `maxSeq` in `ai-load` so workers size buffers identically.
- **Counter and refusal.** A live "143 / 4096 tokens" line under the box from the host tokenizer; refuse to send when prompt + reply cap would overflow ("start a new chat or shorten the prompt") instead of producing garbage; when the reply cap hits, show "[stopped at N tokens]" with a Continue button that resumes decode from `ai.pos` without re-prefill.
- **Textarea.** Auto-growing; Enter sends, Shift+Enter inserts a newline; phones get a Send button only.
- Bit-exactness gate: a golden asserting a two-turn chat equals a one-shot prompt containing both turns (`tests/test_reset.js` is the template).

## Done when
- A 10-turn chat on a two-device room never re-prefills history and the second answer refers to the first question.
- The room shows "context N / M"; an over-long prompt is refused with a message, never silently truncated.
- `tests/test_q38_multiturn.js` passes bit-exact on the GB10 and Mac; `docs/protocol.md` records that `ai-reset` now means "new chat" and that `ai-load` carries `maxSeq`.
```

### `roadmap/14-preflight.md`

```markdown
