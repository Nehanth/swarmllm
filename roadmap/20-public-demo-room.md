# 20 · Public demo room: ask-only guests, question queue, quotas

**Phase:** now (after 15, 17) · **Status:** planned · _now (after 15, 17) · days · high_

## Why

Merges: *Always-on public demo room with status and "lend a GPU"*; *Question queue for shared rooms, never lose what someone typed*; the demo-room mode of *Abuse limits and frame validation*; the copy/transcript half of *Stop, copy, and a transcript*.
Master plan NOW #4 is a stated launch gate with no roadmap file. It cannot ship without three things the code lacks: a queue (today the second asker gets `ai-busy` after the client has already cleared the box at 1174, so their question is lost), per-tab quotas and a prompt cap, and a host flag where only operator devices hold layers and strangers are chat-only guests with the "public experiment, unaudited compute" banner GOVERNANCE requires. "Lend a GPU" is deferred to 03/08 because a volunteer-held spare can corrupt output. If this slips, item 15's status page says "demo room: not yet" honestly.

## Design

See the merged proposals in [docs/roadmap-review.md](../docs/roadmap-review.md) under item 20; turn them into a design note before building.

## Done when

- Acceptance criteria to be written with the design note.
