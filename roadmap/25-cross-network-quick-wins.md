# 25 · Cross-network quick wins: draft cache, frame chunking, prefill acks, per-hop telemetry

**Phase:** now · **Status:** planned · _P1 · design: [docs/research/network-scheduler.md](../docs/research/network-scheduler.md) §2, [decode-overhead-and-wire.md](../docs/research/decode-overhead-and-wire.md) §2_

## Why

Cross-internet rooms run at 3.5–6 tok/s, and the modelled ceiling says most of that is recoverable in **days**, not weeks, without touching the lap-overlap machinery (roadmap 09). Four items, each independently shippable:

- **The room's prefill never fills the MTP draft cache.** Solo prefill fills it; the split path does not, so the first drafts after a prompt are cold. Worth +18–45% tokens per lap.
- **Frames are sent as one 10 KB message.** SCTP's initial burst limit means a message above ~4.6 KB costs an extra round trip on the first send of each hop. Pre-slicing to ≤4.6 KB is modelled at −200 ms per lap at 100 ms RTT, −500 ms at 250 ms. A one-hour A/B decides whether to build it.
- **Prefill returns a full 160 KB hidden-state frame per round that the host discards.** An ack plus rounds-in-flight is modelled at ~2.5× better time-to-first-token.
- **There is no per-hop telemetry**, so we cannot say which device or link is slow. Every other network item is blocked on this.

## Design

See the linked designs for message shapes. Order: telemetry (unblocks everything), draft-cache fill, prefill acks, then frame chunking if the A/B supports it.

## Done when

- A generation reports per-hop time and bytes; `docs/bench-log.md` gains a cross-network row with the breakdown.
- Prefill in a room fills the draft cache (`tests/test_mtp_split.js` gate).
- Measured cross-internet decode improves against the current 3.5–6 tok/s baseline.
