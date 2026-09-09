# 03 · Spare layer copies and swarm recovery

**Phase:** next · **Status:** planned

## Why
A device leaving mid-answer stalls the room. Rooms often have more devices than the model needs; the surplus should buy resilience.

## Design
- **Prerequisite landed:** versioned plans + re-deal on leave/join/manual (roadmap 12, `room/plan.js`, `docs/protocol.md` § Layer plan); this item adds cached-layer reports from workers and spare copies on top.
- **Spares fall out of re-dealing (roadmap 12):** a device that gives up a range during a re-deal keeps its bytes in the Cache API, so it is a warm spare for exactly that range. Explicit spares on top: assign the same range to more than one device when memory allows, preloaded in the background so promotion is a flip at the next question, never a download.
- **Failover with replay:** the host keeps the full token history. When a peer drops, the spare for that range is promoted and the host re-prefills that range's layers on it (batched prefill, 16 tokens per round). A 500-token conversation recovers in a few seconds of "reconnecting".
- **Re-seat on return:** a device that comes back rejoins with its cache intact and can take its range back or become the spare.
- Later: live state mirroring (stream KV/recurrent state deltas to the spare) for instant failover, and hedged execution (both replicas compute, host takes the first) to also mask slow links.

## Done when
- A room of three keeps answering, after a short pause, when one device closes its tab mid-generation.
- `docs/protocol.md` gains the messages for promotion and re-seating.
