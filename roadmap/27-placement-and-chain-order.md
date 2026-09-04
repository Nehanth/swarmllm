# 27 · Placement, chain order and host election

**Phase:** next · **Status:** planned · _P2 · design: [docs/research/network-scheduler.md](../docs/research/network-scheduler.md) §5_

## Why

Layers are dealt in proportion to pledged memory, and the chain order is effectively arbitrary. Neither accounts for how fast a device actually computes or how far it is from its neighbours, so one slow phone or one distant peer sets the pace for every token. On a LAN the network is under a millisecond and the whole lap is device compute; across sites the chain order can double or halve the lap. Prior art agrees on the shape of the fix: exo weights a ring by memory and link, prima.cpp drops peers that would slow the pipe, and both order the ring by measured cost.

## Design

- Measure per-device throughput (the autotuner already does part of this) and pairwise RTT at join.
- Deal layers in proportion to measured throughput, not pledged memory, subject to the memory cap.
- Order the chain to minimise total lap time; drop a peer whose share would cost more in hops than it saves in compute (a phone holding one layer is usually a net loss).
- Prefer peers whose cache already holds the range they are being dealt.
- Elect the host by capability rather than by who clicked first.

## Done when

- A room with one deliberately slow device is measurably faster than the same room with today's memory-proportional split, and the chain order is shown in the room UI.
