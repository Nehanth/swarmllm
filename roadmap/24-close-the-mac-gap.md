# 24 · Close the Mac gap

**Phase:** now · **Status:** planned · _P1 · design: [docs/research/mac-metal-plan.md](../docs/research/mac-metal-plan.md)_

## Why

The Mac is the demo device and the one most users have, and it sits far below its bandwidth roofline: 6.7 tok/s plain against a weight-stream floor of 56–101 ms/token for a Pro/Max-class chip. The research round found that the headline Mac and GB10 numbers were **never measuring the same thing**: the GB10 figure comes from `forwardToken` (one submit per token), while the room path (`aiPipeToken` → `embedRun` + `headFromHidden`) does **three submits, two `mapAsync` round trips and a full pipeline drain** between the last layer and the LM head — and the Mac number predates the `_acc` and `dn_pre` fusions. So part of the "Mac gap" may be harness, not hardware.

## Design

Gates first, optimizations second.

1. **Re-measure at HEAD** and A/B `aiPipeToken` against `forwardToken` in the same tab. Route solo decode through `forwardToken` (half a day, expected 5–15%).
2. **Measure achievable bandwidth through WebGPU on the Mac** with the streaming-read probe. If `15 GB / BW_measured` ≈ the observed token time, there is no gap and the rest of this item is void.
3. **A browser profiling page** (`bench.html`): the Deno harnesses cannot run in Chrome, so the skip-family profiler, the bandwidth probe and a dispatch/encoder-cost sweep need a page version.
4. Then, in order of measured payoff: Dawn toggle ablations and a widened autotune sweep (WG=32 for Apple's SIMD width, ROWS=2/16, `batchCols`); K/V append without blit encoders (−32 copies and −16 pass splits per token, −256 per 8-column verify); column-parallel attention glue (verify pass 1232 → 896 dispatches); encode-ahead double buffering; and the conditional kernel work (`vec4<u32>` weight loads, `attn_softmax`/`head_norm` reshape).

Per-token structure today, measured from the encoder: **898 dispatches, 145 compute passes, 32 blit copies, 1 submit** for decode; **1232 dispatches, 144 passes, 256 copies** per 8-column verify.

## Done when

- The Mac's achievable bandwidth is measured and recorded, so "gap" or "no gap" is a number rather than a guess.
- `bench.html` runs the skip-family profile and the bandwidth probe in Chrome and Safari.
- Whatever the profile indicts is fixed, bit-exact, with a bench-log row per change.
