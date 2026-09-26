# Tabby: the two big next steps (designs, not built)

Both need real devices to tune, so they are written down here instead of built blind. Numbers
below are arithmetic from the model's shapes and the research notes, not measurements.

## 1. Pipelined speculative windows across devices

**Why.** In a split room one speculative step is one lap: host drafts K tokens, the verify frame
goes device 1 → 2 → … → host, the host samples, and only then the next step starts. Every
device idles for the rest of the lap. With D devices each busy for 1/D of a lap, utilisation is
~1/D. Mesh-LLM keeps several verify windows in flight across its pipeline stages
(research §2); PipeInfer does the same for prefill-like work.

**Idea.** Start step s+1 before step s's verify returns, betting that step s is accepted in full:
its drafts are the MTP head's continuation after step s's drafts (the host already has them:
draft K + K' tokens in one go). Device 1 finishes its layers of step s and immediately starts its
layers of step s+1 while device 2 works on step s.

**Exactness.** Step s+1 was computed assuming all of step s's drafts are accepted. When the host
sees step s rejected at column k:
- every device must roll back its recurrent state to "after column k of step s" (the replay
  rollback already does that for one step: `rb` on the next frame),
- and throw away whatever it did for step s+1, which sits on top of it: the DeltaNet state after
  step s+1 is not recoverable from "after step s" without replaying. Replay makes it cheap: each
  device keeps S_pre (state before step s) and step s's per-column inputs, so it restores S_pre,
  replays columns 0..k of step s, and discards step s+1's inputs.
- The KV rows step s+1 wrote are simply overwritten later (positions are absolute).
So a new control word, `rb2 = (step, k)`, instead of `rb = k`, and two sets of replay buffers
(S_pre and the column inputs for 2 steps in flight). Frames carry a step number so a device
drops frames of a cancelled step that arrive after the rollback.

**Cost / win (27B, 3 devices, K = 3, ~85% acceptance).** P(step fully accepted) ≈ 0.85³ ≈ 0.61.
With 2 windows in flight the expected lap throughput rises from 1 step per lap to ~1.6 steps
per lap; wasted work is ~39% of the second window. Memory: one more S_pre per DeltaNet layer
(3 MB each on the 27B, the device's share) and one more set of column inputs.

**Order of work.** (1) step ids in the frame header (spare bits: protocol 5); (2) engine: two
replay slots, `restoreDN(step, k)`; (3) host scheduler: keep ≤ 2 steps in flight, cancel on
reject; (4) measure lap time vs. tok/s on 2-3 real devices; (5) only then consider 3+ windows.

## 2. Several sessions in one pass (batched decode)

**Why.** Today `harness/sessions.js` time-shares: one session computes, the others wait in slots.
Decode is memory-bound (every weight is read once per token), so decoding B sessions in one pass
reads the weights once for B tokens: ~B× throughput for the room, until compute binds (the
batched GEMV / GEMM kernels already exist for 4-16 columns).

**What differs from a verify pass.** A verify's columns are consecutive positions of one
sequence sharing one KV cache and one recurrent state. Session columns are independent:
- attention: each column has its own KV cache and its own length. `attn_flash` already takes
  per-column bind data; add a per-column table { kv buffer offset, seqLen } (one big KV arena,
  paged in 256-position blocks like vLLM's block table, so sessions grow without copies),
- DeltaNet: each column has its own recurrent state S and conv window. `dn_delta_mc` loops
  columns over one S; the batched-session version indexes S by column (one S per session in an
  arena) and runs columns in parallel instead of in sequence (they do not depend on each other),
- positions: rope and the frame's pos become per-column (the kernels already take pos + col;
  make it a per-column array).
Everything else (norms, GEMVs, the head) is already per-column.

**Memory (27B, per session, whole model):** DeltaNet state ~150 MiB + KV 65.5 KB/token (f16) or
36 KB/token (int8). Four 8K-token sessions: ~0.6 GB of states + ~2.1 GB of f16 KV, split over
the room's devices.

**Exactness.** Per-column arithmetic is unchanged, so each session's tokens equal what it would
produce alone (the same property the verify path already has): test = N sessions decoded
together vs. each alone, bit-identical.

**Order of work.** (1) arena + block table for KV and states in the engine, with the current
single-session path as the 1-column case; (2) per-column pos/seqLen table in the frame;
(3) `Sessions` switches from time-sharing to batching when several agents are waiting;
(4) measure tok/s per session and total on real devices.
