# 26 · Host-side overhead: encode-ahead, one-submit speculation, GPU sampling, fused glue

**Phase:** next · **Status:** planned · _P2 · design: [docs/research/decode-overhead-and-wire.md](../docs/research/decode-overhead-and-wire.md) §1_

## Why

A GB10 decode token is 82 ms of weight streaming plus ~30 ms of everything else: encoding ~900 dispatches in JavaScript, submitting, and a 1 MB logits readback with CPU sampling. A speculative step at K=3 makes **five blocking GPU round trips and nine submits** for 3.55 tokens. None of that is bandwidth; it is host-side latency, and it is a larger share on Metal than on Vulkan.

## Design

Ranked by measured gain per week:

1. **Encode-ahead**: build token N+1's command buffer while N is in flight (measured −7.45 ms/token on the GB10).
2. **One-submit draft chain**: gather the draft token's embedding on the GPU so the K drafts become one submit instead of K submits with a blocking 16-byte readback each (measured ceiling −52 ms/step at K=3, −105 ms at K=7).
3. **GPU top-k for the batched head**, removing a 4 MB `mapAsync` per verify.
4. **Fused QKV glue** (`qsplit` + two `head_norm` + two `rope_part` + `sigmoid_mul` in one dispatch): measured ceiling 2.84 ms/token.
5. K/V append via dynamic offsets instead of blit copies: measured neutral on Vulkan, unknown on Metal (see roadmap 24).

## Done when

- Non-GEMV time per token is measured before and after on both a GB10 and a Mac, with rows in `docs/bench-log.md`, and output stays bit-exact.
