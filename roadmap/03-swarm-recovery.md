# 03 · Spare layer copies and swarm recovery

**Phase:** next · **Status:** planned

## Why
A device leaving mid-answer stalls the room. Rooms often have more devices than the model needs; the surplus should buy resilience.

## Design
- **Redundant layer copies:** assign the same layer range to more than one device when memory allows. Weights are already cached, so a spare costs nothing until it is used.
- **Failover with replay:** the host keeps the full token history. When a peer drops, the spare for that range is promoted and the host re-prefills that range's layers on it (batched prefill, 16 tokens per round). A 500-token conversation recovers in a few seconds of "reconnecting".
- **Re-seat on return:** a device that comes back rejoins with its cache intact and can take its range back or become the spare.
- Later: live state mirroring (stream KV/recurrent state deltas to the spare) for instant failover, and hedged execution (both replicas compute, host takes the first) to also mask slow links.

## Done when
- A room of three keeps answering, after a short pause, when one device closes its tab mid-generation.
- `docs/protocol.md` gains the messages for promotion and re-seating.
