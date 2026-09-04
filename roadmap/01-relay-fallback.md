# 01 · Relay fallback for strict NATs

**Phase:** now · **Status:** planned

## Why
WebRTC hole-punching (STUN) fails between some pairs of home routers (symmetric NAT). Today that room silently never connects. On a launch day that is the worst possible failure: "I joined and nothing happened."

## Design
- Run a TURN server (coturn on a small VPS) and add it to the ICE configuration with credentials issued per room by the signaling broker.
- The relay only ever carries activation frames (~10 KB per hop, f16), never weights: bandwidth cost per room is tens of KB/s.
- Prefer direct paths; use the relay only when ICE reports no direct candidate pair. Show "relayed" in the room UI so users understand the extra latency.
- Later: the signaling broker can double as a WebTransport relay; it cannot replace the mesh (client-server only) but is a better relay than TURN.

## Done when
- A room between two devices behind symmetric NATs connects and completes a generation.
- The room UI shows which links are direct vs relayed.
- `docs/protocol.md` documents the ICE configuration and what the relay can see (encrypted frames only).
