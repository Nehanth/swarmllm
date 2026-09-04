# Architecture

SwarmLLM has two halves: an inference **engine** that runs a model (or a slice of one) on a device's GPU through WebGPU, and a **room runtime** that connects browsers over WebRTC and threads one generation through all of them.

```
                 ┌──────────────────────── host browser ────────────────────────┐
 text ──tokenize──► embed ──► layers 0..k ──┐                      ┌──► final norm ──► LM head ──► sample ──► text
                                             │ hidden state (f16)   │
                                             ▼                      │
                                       peer A: layers k+1..m ──► peer B: layers m+1..N
```

## One token, end to end

1. **Tokenize** (host CPU). Text → token ids using the tokenizer stored in the GGUF header.
2. **Embed** (host). The id selects a row of the embedding table: a `dim`-float hidden state (5,120 for Qwen 3.8).
3. **Layers** (every device, in order). Each device runs its contiguous range of transformer blocks on its GPU. Qwen 3.8 has 64: 48 Gated-DeltaNet blocks (a recurrent state matrix per head, constant memory in context length) and 16 full-attention blocks (with a KV cache). All ops of a device's range are recorded into one command buffer and submitted once.
4. **Hop.** The hidden state is packed to f16 (10 KB) and sent to the next device over a WebRTC data channel as a binary frame.
5. **Head** (host). Final RMSNorm, then the LM head matvec over the vocabulary (248,320 rows) → logits → readback → sampling on the CPU.
6. **Draft** (host). The model's built-in multi-token-prediction layer proposes the next token(s) from the last hidden state; see below.

Hosts that also hold layers (the usual case) run steps 2–3 for their own range before the first hop.

## Prefill

The prompt is known upfront, so it is processed in batches: up to 8 tokens per GPU pass (column-batched kernels read each weight block once for all columns) and 16 tokens per network round. During prefill nothing runs the LM head; the pass only fills the KV caches and recurrent states. Causality inside a batch is preserved by strictly ordered column processing in the recurrent kernels and by appending K/V before attending.

## Speculative decoding

Qwen 3.5/3.8 ship a `nextn` draft block. Given the trunk's last hidden state and the just-sampled token, it predicts the next token; chaining it predicts several. The host then verifies `1 + K` tokens in **one** batched trunk pass (up to 8 columns, one network lap in a room). Accepted drafts are those the trunk's own sampled token agrees with; the first mismatch ends acceptance, the trunk's token is used, and every recurrent state is restored from a snapshot taken between columns. Because the trunk always decides, the output stream is identical to plain decoding for any sampler. Draft depth (3/5/7) is chosen per room by measured tokens per second.

## Memory and caching

Each device range-fetches only the byte spans of its tensors from the public model repository and streams them into GPU buffers, repacking Q4_0/Q8_0 blocks into separate nibble and f16-scale arrays for coalesced reads. Entries are stored in the browser Cache API with a size stamp, so rejoining a room reloads from disk in seconds.

## Where time goes (GB10, 27B)

Measured with `benchmarks/bench_breakdown.js` (skips kernel families and re-times):

| | per token |
|---|---|
| all matvecs (weights streamed at ~183 GB/s; roofline 184) | 82 ms |
| everything else (small kernels, encode, submit, readback) | ~30 ms |

Decode is at the memory roofline on this GPU; speculation is what raises tokens per second. Prefill is bound by the batched matvec path and is the subject of the GEMM work in `benchmarks/bench_gemm.js`.

## Files

- `engine/engine.js`: base WGSL, the cooperative-GEMV generator (`coopWGSL`), the dense `DenseEngine` (Qwen3, SmolLM), autotune, self-tests.
- `engine/qwen35.js`: `Qwen35Engine` for the hybrid DeltaNet/attention family, batched paths, MTP speculation.
- `engine/gguf.js`: GGUF header/tensor parsing, tokenizer extraction, quantization/repacking, streaming upload.
- `p2p.html`: the room: signaling, WebRTC mesh, layer assignment, download orchestration, the generation loop.
