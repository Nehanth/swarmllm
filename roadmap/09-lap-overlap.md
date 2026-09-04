# 09 · Overlapping laps across the network

**Phase:** later · **Status:** research

## Why
Cross-network decode pays one round trip per lap while every device waits. Speculation already carries up to 8 tokens per lap; the next step is keeping the pipeline full: start verifying the next draft chain before the current verdict returns, and cancel it if the verdict rejects (PipeInfer reports up to 2.15× on slow interconnects).

## Design sketch
- The host launches lap N+1 on the assumption that lap N accepts fully; peers need two laps' worth of rollback slots and a versioned cancel message.
- Gate on data: the room already reports measured tokens per second per draft depth; build this only if real cross-network rooms show the lap is round-trip-bound.

## Done when
- Cross-internet decode improves measurably on a real two-network room without changing output.
