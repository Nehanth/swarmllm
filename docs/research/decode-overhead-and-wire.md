# decode-overhead-and-wire

I have measured evidence. Writing the design now.

# SwarmLLM: the 30 ms and the wire — implementation-ready design

All numbers below marked **[M]** were measured this session on the GB10 (DGX Spark, Deno 2.9.5 / wgpu / Vulkan, GPU otherwise idle, Qwen 3.8 27B Q4_0, full 64 layers). Probe scripts are at a scratch directory (not committed) and should be committed under `benchmarks/` before any of these numbers go in a paper. Numbers marked **[E]** are estimates. Numbers marked **UNCERTAIN** are not derivable from the repo or this session.

---

## 0. Corrections to the current docs (do these first, they cost minutes)

| Doc | Says | Measured **[M]** |
|---|---|---|
| `docs/kernels.md` trick 9 | "~1,100 dispatches of a token" | **898 dispatches, 145 compute passes, 32 blit copies** per decode token (`probe33.js`, instrumented encoder). 1,100 is the pre-`_acc` count. |
| `docs/kernels.md` trick 11 | "192 fewer dispatches per token" | 128 (`add_res` is 2 sites per layer × 64, not 3). |
| `docs/architecture.md` | "~30 ms everything else" | **28.3 ms at pos 8, ~40 ms at pos 500** — the residual grows with context because attention is O(seqLen) in three serial kernels. |
| `docs/bench-log.md` | GB10 plain decode 9.0 tok/s | Correct as a *Deno* number, but **11.5 ms of every token is Deno's `mapAsync` poll sleep**, not GPU or dispatch cost. See §1.1. |

---

## 1. Part A — the 30 ms of non-GEMV per token

### 1.0 The budget, measured

`probe30.js`, corrected skip-family lists (the shipped `benchmarks/bench_breakdown.js` skip lists are **stale**: they omit every `matvec_*_coop_acc` pipeline and still name `dn_gates`/`dn_l2`, which `_encodeLayerR` no longer dispatches — it dispatches `dn_pre`. Un-skipped `_acc` matvecs are ~30% of streamed weight bytes, so the shipped tool currently dumps ~25 ms/token of pure weight streaming into "everything else").

Plain decode token at pos ≈ 8: **108.4 ms total [M]**

| Bucket | ms **[M]** | What it is |
|---|---|---|
| all matvecs (incl. `_acc`, `gu`, LM head) | **80.06** | at the 184 GB/s roofline; not addressable |
| **Deno `mapAsync` poll floor** | **11.56** | `ext/webgpu/buffer.rs`: `device_poll` then `tokio::time::sleep(10ms)` per await. Chrome polls Dawn every 1–2 ms. |
| CPU encode (`createCommandEncoder`…`finish`) | **4.11** | 898 dispatches × (setPipeline + 2× setBindGroup + dispatch) |
| `queue.submit` | **0.30** | |
| embed row write (CPU dequant + `writeBuffer`) | **0.13** | `_embedRowF32` alone is **0.007 ms [M]** — the CPU dequant is *not* a cost |
| `dn_delta` (48 WGs × 128 threads, 2× serial 128-loop over 64 KB state) | **5.22** | |
| `rmsnorm` (129 dispatches/token, one 256-thread WG each) | **2.99** | ≈ 23 µs each |
| `dn_pre` + `dn_gatenorm` | **2.84** | |
| `qsplit` + 2× `head_norm` + 2× `rope_part` + `sigmoid_mul` | **2.84** | |
| `attn_scores`/`softmax`/`out` | **~0 @pos8 → 8.66 @pos500** | `attn_softmax` is `@workgroup_size(1)`, 24 threads, 3 serial passes over seqLen |
| `dn_conv` | **0.61** | |

Skip-delta attribution over-counts by ~8% (removing a family also removes its dispatch slots); treat the glue rows as ±10%.

Position dependence **[M]** (`probe33.js`): token = 109.1 ms @pos8, 114.4 @pos128, 116.5 @pos256, **120.5 @pos500**; the attention family alone goes 0 → 2.6 → 3.8 → **8.7 ms**, of which `attn_softmax` alone is **2.60 ms** at pos 500.

**The single most important structural fact:** on GB10, moving *bytes* across the readback boundary is nearly free — a 16-byte readback and a 0.99 MB readback both cost **11.6 ms [M]**, and inside a single-submit step the difference between a 16-byte readback and a full 7.95 MB 8-column logits readback is **1.1 ms [M]**. The cost is the *sync point*, not the payload. Every "reduce the readback" idea should be priced against that.

### 1.1 GPU sampling — deprioritize, and here is the number

**Do not build a GPU top-k for speed.** Measured ceiling (`probe32.js`, one-submit K=3 step with `n=8` verify): replacing the full 7.95 MB logits readback with a 16-byte argmax readback saves **1.1 ms** out of a 267 ms step (0.4%). CPU `aiSample(top-k=40)` over 248,320 logits costs **0.28 ms [M]**; `Float32Array.from` of the 1 MB buffer costs **0.03 ms [M]**.

The `argmax` kernel already in `engine/wgsl/qwen35.js:332` is what the draft chain needs, and it already exists. The only reason to build an exact GPU top-40 is:

- discrete-GPU hosts over PCIe (a Mac/GB10 both have unified memory) — **UNCERTAIN**, unmeasured;
- Safari, where `mapAsync` cost is unmeasured — **UNCERTAIN**.

**If built anyway**, the design that preserves `room/sampling.js` semantics exactly:

1. 4 histogram passes of 256 `atomic<u32>` bins over the orderable-u32 bitcast of the logits (`k = v ^ (v>>31 ? 0xFFFFFFFF : 0x80000000)`), pass 1 over all 248,320 keys, passes 2–4 over the surviving 8-bit bucket only (~10³ keys). Yields the exact 40th-largest key.
2. One compaction pass writing `(value, index)` pairs for keys ≥ threshold into a 64-entry buffer with a clamped atomic counter (ties can exceed 40).
3. Host truncates to 40 keeping **lowest index** — `aiSample`'s scan uses strict `v > min`, so the incumbent (lower index) wins a tie. Reproduce that or the sampler is not equivalent.
4. **NaN**: the orderable-u32 transform maps NaN above `+Inf`, so a NaN logit becomes the top candidate. `aiSample` can never select a NaN (`v > min` is false). Mask NaN in the key transform. Note `badF32` (`room.js:874`) only guards the non-batched `headFromHidden` path — the batched verify path has no NaN check today.
5. Batch the histogram across all 8 columns in one dispatch grid (`dispatch(nWG, nCols)`), 4–5 dispatches per verify, not per column.

Effort ≈ 4 days. Expected gain on GB10: **1.1 ms/step [M]**. Rank it last.

### 1.2 Readback pipelining — two separate wins, both real

#### 1.2a Encode-ahead (double-buffered command encoding) — **MEASURED, ship it**

Today `forwardToken` (`engine/qwen35.js:946`) is strictly serial: `writeBuffer → encode 64 layers → submit → mapAsync → sample → next`. The GPU is drained by the `mapAsync` await before the next token's JS encode starts, so 4.1 ms of encode + 0.3 ms submit sit on the critical path every token.

Almost nothing in the command buffer depends on the sampled token: only `writeBuffer(x, embedding)` and the frame uniform. `pos` is known one token ahead in plain decode, and the two things baked into the recording at `pos` — the K/V `copyBufferToBuffer` offsets (`qwen35.js:383-384`) and the `attn_scores` dispatch size `nH*(pos+1)` (`qwen35.js:388`) — are both known ahead.

**A/B measured [M]** (`benchmarks/bench_pipe_ab2.js`, interleaved 6-token blocks × 8, 48 samples each):

| | ms/token | CPU before await |
|---|---|---|
| serial (today) | 113.35 (min 111.66) | 8.88 ms |
| pipelined | **105.90** (min 104.22) | **0.88 ms** |
| delta | **−7.45 ms/token (−6.6%)** | |

8.83 → 9.44 tok/s. Bit-identical by construction (same commands, same order).

Implementation, `engine/qwen35.js`:

```
// state
this._pre = null;          // prebuilt GPUCommandBuffer for this.pos
_buildToken(pos) { ...same body as forwardToken's encode... return enc.finish(); }

async forwardToken(id) {
  const cb = this._pre ?? this._buildToken(this.pos);
  this._setFrame(this.pos, this.pos + 1);
  this.device.queue.writeBuffer(this.x, 0, this._embedRowF32(id));
  this.device.queue.submit([cb]);
  this._pre = null;
  const rbEnc = ...copy logits→stage...; submit;
  this.pos++;
  this._pre = this._buildToken(this.pos);   // <-- record N+1 while N runs
  const logits = await this._readback(...);
  return logits;
}
```

**Invalidation is mandatory and is the only correctness hazard**: `_pre` bakes in `pos`. Discard it (set `this._pre = null`) in `reset()`, in `_restoreDN()`, at the end of `specStep` (accepted count `a` is not known at record time), and on any cancelled generation. An unsubmitted `GPUCommandBuffer` is simply dropped, so discarding is free. `GPUCommandBuffer` is single-submit per spec (`queue.submit` invalidates it), so this is re-recording ahead, not reuse.

Same change applies to the room worker path: `runHidden(xIn, pos)` (`qwen35.js:919`) takes `pos` from the wire, so a worker can record for `pos+1` while the current frame is in flight — but only after the per-peer executor of §2.2 lands, or two in-flight frames will interleave.

Effort: **2–3 days** including the invalidation audit and a golden run.

#### 1.2b One submit + one sync per speculative step — **the biggest single lever, with a caveat**

Today `specStep` (`qwen35.js:838`) has **K+2 sync points**: K × `mtpRun(..., "argmax")` each ending in `stageArg.mapAsync` (`:792`), plus `embedRunBatch → _runBatchAndRead`'s `stageXB.mapAsync` (`:704`), plus `headBatch`'s `stageLogitsN.mapAsync` (`:755`).

Measured ceiling **[M]** (`probe32.js` — everything encoded into one command buffer, one submit, one readback; token ids are stale so timing only):

| config | today | one submit + 16 B | one submit + full logits | sync saving | logits-byte saving |
|---|---|---|---|---|---|
| K=3, n=4 verify | 215.4 ms | **161.9** | 163.2 | **−52.1 ms (−24%)** | 1.3 ms |
| K=3, n=8 verify | 319.2 ms | **267.2** | 268.3 | **−50.9 ms (−16%)** | 1.1 ms |
| K=7, n=8 verify | 404.4 ms | **299.6** | 299.1 | **−105.4 ms (−26%)** | −0.6 ms |

Sync arithmetic checks out: K=3/n=4 removes 4 syncs × 11.5 ms = 46 ms (measured 52, the rest is de-serialized encode); K=7 removes 8 × 11.5 = 92 ms (measured 105).

**The caveat you must state in the paper:** ~11.5 ms of each removed sync is Deno's poll sleep. Chrome's Dawn poll is 1–2 ms, so on a browser this change is worth roughly **6 ms at K=3 and 12 ms at K=7 [E]**, plus the de-serialization of ~5 ms of CPU encode per step. Still worth it, but it is not a 26% browser win. **Measure the Chrome `mapAsync` floor before committing the week** — `probe30.js`'s SYNCFLOOR block ported to a page is a 20-line job.

Corollary worth its own bench-log row: a patched Deno (poll sleep 10 ms → 0–1 ms) would move the published GB10 numbers from 9.0 → **~10.0 tok/s** plain and 16.1 → **~21 tok/s** at K=3, with zero code change. Either patch a local Deno build or report GB10 numbers floor-subtracted with the harness stated.

**What blocks the one-submit step:** exactly one thing. `mtpRun` (`qwen35.js:770`) feeds draft *k+1* by reading back draft *k*'s argmax to the CPU, dequantizing an embedding row on the CPU (`_embedRowF32`), and `writeBuffer`-ing it. The chain is GPU → CPU → GPU per draft.

Note **the CPU dequant is not the problem — it costs 0.007 ms [M]**. The problem is the readback. So the fix is a GPU-side embedding gather indexed by the existing `argBuf`:

**Design: `embed_gather` kernel + GPU-resident embedding table**

- Upload `token_embd.weight` to the GPU as it already is on disk: Q4_0, `248320 × 5120 × 18/32 = 715 MB`. (Not Q8 — the tensor is Q4_0 in the GGUF; `weights.embed` at `qwen35.js:269` currently keeps it CPU-side deliberately. `output.weight` is a *separate* Q6_K tensor requantized to Q8 at load, so there is no tied-embedding shortcut.) Gate the upload on available VRAM; keep `cpuEmbed` as the fallback.
- New kernel, ~20 lines, reuses the existing `q4_qs`/`q4_sc` layout and `q4s()` helper from `engine/wgsl/base.js`:

```wgsl
@group(1) @binding(0) var<storage, read> eg_qs: array<u32>;
@group(1) @binding(1) var<storage, read> eg_sc: array<u32>;
@group(1) @binding(2) var<storage, read> eg_id: array<u32>;   // argBuf: [index, bits]
@group(1) @binding(3) var<storage, read_write> eg_y: array<f32>;
@compute @workgroup_size(64)
fn embed_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= cfg.dim) { return; }
  let row = eg_id[0];                     // GPU-resident token id
  let nb  = cfg.dim / 32u;
  let b   = i / 32u; let j = i % 32u;
  let si  = row * nb + b;
  let s   = unpack2x16float(eg_sc[si >> 1u])[si & 1u];
  let w   = eg_qs[(row * nb + b) * 4u + (j % 16u) / 4u];
  let nib = (w >> (((j % 16u) % 4u) * 8u + select(0u, 4u, j >= 16u))) & 0xFu;
  eg_y[i] = s * (f32(nib) - 8.0);
}
```
  (Exact bit layout must match `q4Repack` in `engine/gguf.js:388`; the reference is `_embedRowF32` at `qwen35.js:426`, and the golden is `dequant(gather(id)) == _embedRowF32(id)` element-for-element.)

- `mtpRun` gains a `srcId: "gpu"` mode: instead of `writeBuffer(M2.emb, …)` it dispatches `embed_gather` reading `argBuf`, into `M2.emb`. Draft *k+1*'s pass then depends only on draft *k*'s `argBuf`, and WebGPU's in-pass/in-encoder ordering makes that a plain data dependency inside one command buffer.
- `specStep` becomes: **write** the K+1 verify column embeddings and frame uniforms → **one encoder** containing [K draft blocks with `embed_gather` + `argmax` each] + [64-layer batched verify with snapshots] + [batched final norm + head] → **one submit** → **one `mapAsync`** of `n*vocab*4` (or 2.5 KB if §1.1 ever lands) → CPU sample + accept.
- After the sample, `_restoreDN(a)` and the `a` refill `mtpRun` calls stay as they are — they already issue no syncs (`mtpRun(..., false)` costs **0.27 ms [M]** of CPU and returns immediately).
- The verify columns still need CPU-side embedding writes because the drafts are only known after the readback of the *previous* step. That is one write of `(K+1) × 5120` floats: `(K+1) × 0.007 ms` **[M]**. Negligible.

**Bit-exactness:** unchanged. Drafts only affect *which* tokens are proposed; the trunk still decides, and the trunk's math is untouched. `tests/test_mtp.js` (spec output == plain output) is the gate and must still pass. The one thing to verify is that `embed_gather` is bit-identical to `_embedRowF32` — if it is not, drafts change, acceptance changes, and `test_mtp.js` still passes but the tok/s row moves. Gate on the element-wise dequant test, not on tok/s.

Effort: **1 week** (kernel + gather golden + `specStep` restructure + VRAM gating).

Second-order benefit: with drafts GPU-resident, the K draft passes and the verify pass are one command buffer, which is also the prerequisite for encode-ahead across spec steps.

### 1.3 Fusions — measured, and mostly small on GB10

Prior evidence says small dispatches are ~free on Vulkan (trick 11: −192 dispatches, neutral). This session's numbers agree: the addressable GPU-side glue is **~14.5 ms/token at pos 8, ~23 ms at pos 500 [M]**, and a fusion recovers only the *launch + re-read* portion of that, not the arithmetic.

Ranked by measured ceiling:

**F1. Fused attention (`attn_scores` + `attn_softmax` + `attn_out` → one workgroup per head).** Ceiling **8.66 ms/token at pos 500, ~0 at pos 8 [M]**. `attn_softmax` is `@workgroup_size(1)` — 24 threads doing three serial passes over `seqLen`; alone it is **2.60 ms at pos 500 [M]**. Design: one WG of 256 threads per head; load K rows coalesced; scores into `var<workgroup> array<f32, maxSeq>` (2 KB at 512); block-reduce max; then weighted V sum; fold `sigmoid_mul` into the epilogue. **Bit-exactness hazard:** max is order-free but the exp-sum is not, and a block reduction reorders f32 adds. To stay bit-identical, keep thread 0 doing the 512-term sum serially from workgroup memory — you still recover the load parallelism and lose only the reduction. Same for the per-head dot. GQA note: 6 q-heads share each kv head, so 24 WGs re-read each K/V row 6×; one WG per *kv* head with 6 resident q-heads removes that.
  **Priority: low today.** `MAX_SEQ = 512` (`room/models.js:26`) and `room.js:884` resets on every send, so a single answer never exceeds ~400 positions. Averaged over an answer this is **~2.5–3 ms/token [E]**. It becomes the top fusion the moment roadmap 13 raises `maxSeq`.

**F2. KV append via dynamic offsets.** Removes **32 blit copies and 32 compute-pass splits** per token [M]. `kvDim*4 = 4096 B`, a multiple of the 256 B `minStorageBufferOffsetAlignment`, so `hasDynamicOffset` bind groups with offset `pos*4096` work directly; `maxSeq 512 → 2 MB` per cache. Do **not** add `hasDynamicOffset` to the shared `matvec_*_coop` layout (it would force `setBindGroup(1, bg, [0])` on every GEMV) — create a second pipeline object for the K/V GEMVs with a dynamic-offset layout and leave the frozen layout alone. Same fix applies to `_encodeLayerBatch`, which blits `2 × nCols` per attention layer (up to **256 blit copies per 8-column verify**) — `B.k.stride = 4096 = kvDim*4` already, so `basePos*4096` lands columns in consecutive slots.
  Expected on GB10: **~0 ms** (this class measured neutral). On Metal: **UNCERTAIN** — Dawn opens one `MTLComputeCommandEncoder` per pass and a `MTLBlitCommandEncoder` per copy run, so this removes 32 pass boundaries + 16 blit encoders. Bit-exact. Effort 2 days.

**F3. Fused QKV glue** (`qsplit` + 2× `head_norm` + 2× `rope_part` + `sigmoid_mul` → 1 kernel). Family ceiling **2.84 ms/token [M]**, of which maybe half is recoverable. `head_norm` accumulates `ss` serially per head — keep that serial in the fused kernel and it stays bit-identical. Effort 3 days.

**F4. Fused `dn_pre` + `dn_conv`.** Combined ceiling **3.45 ms/token [M]**, mostly `dn_pre` (2.84 with `dn_gatenorm`). Marginal.

**F5. Concatenate the four DeltaNet GEMVs that all read `xn`** (wqkv 10240 rows, wz 6144, wBeta 48, wAlpha 48) into one dispatch. Same total workgroup count, so occupancy is unchanged; needs a load-time repack and identical quant kind. Removes 3 dispatches × 48 layers = 144/token. On Vulkan ≈ 0; on Metal **UNCERTAIN**.

**What NOT to fuse:** `rmsnorm` into the GEMV (already measured 0.91× on Metal, `docs/kernels.md`). The 129 `rmsnorm` dispatches cost **2.99 ms [M]** = 23 µs each and are a genuine full-vector reduction; there is no cheap fusion.

**Post-fusion dispatch count:** 898 → ~740 with F1–F3, → ~590 with F5. `docs/kernels.md`'s claim of a 192-dispatch saving from `_acc` should read 128.

---

## 2. Part B — the wire

### 2.1 Fix the f16 packer — **hours of work, the highest-value item in this document**

`engine/gguf.js:16` allocates **two typed arrays per element**:

```js
export function f32ToF16(v) {
  const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);   // ← per call
  ...
  return sign | (e << 10) | ((m + 0x1000) >> 13);                        // ← `|`, not `+`
}
```

Two independent defects.

**(a) Cost.** `packF16` is called once per hop on every frame (`room.js:867, 932, 970` host; `1102, 1124` worker). Measured **[M]** (this machine, Node/V8):

| payload | current | hoisted scratch | speedup |
|---|---|---|---|
| 5,120 floats (1 hidden) | **1.89 ms** | 0.016 ms | 119× |
| 8 × 5,120 (verify frame) | **15.02 ms** | 0.117 ms | 128× |
| 16 × 5,120 (prefill round) | **30.30 ms** | 0.234 ms | 129× |

A 3-hop 8-column verify lap burns **~45 ms of main-thread CPU** in `packF16` alone; a 3-hop 16-token prefill round burns **~91 ms**. On a modeled 300 ms cross-internet lap that is **15%**, for free.

**(b) Correctness.** When the rounded mantissa carries out (`(m + 0x1000) >> 13 == 0x400`), OR-ing into `e << 10` cannot carry when `e` is odd — the exponent is not incremented and **the value is halved**. Measured over 4M random values in [−8, 8] **[M]**: mismatch rate **3.30e-4, all gross** (`f32ToF16(1.9999) → 1.0`, `0.49999 → 0.25`, `7.9999 → 4.0`). That is **~1.7 elements of every 5,120-dim hidden state silently halved on every hop, today.**

Measured on **real hidden states at the real layer-32 split boundary** (`wire_err.js`, 26 prompt positions, `dim=5120`, `max|x| = 278.7`, median channel max 1.394, 3 massive channels):

| f16 packer | rel L2 | rel L2 (ordinary channels) | max abs error |
|---|---|---|---|
| current (buggy) | **2.93e-3** | 5.55e-3 | **1.00** |
| fixed (`+` instead of `|`, scratch hoisted) | **1.98e-4** | 2.06e-4 | 5.97e-2 |

**15× less wire error and 129× faster, in about five lines.**

```js
const _f16buf = new Float32Array(1), _u32buf = new Uint32Array(_f16buf.buffer);  // module scope
export function f32ToF16(v) {
  _f16buf[0] = v;
  const x = _u32buf[0];
  const sign = (x >>> 16) & 0x8000;
  let e = (x >>> 23) & 0xff, m = x & 0x7fffff;
  if (e === 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
  e = e - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) { if (e < -10) return sign; m = (m | 0x800000) >> (1 - e); return sign | ((m + 0x1000) >> 13); }
  return sign | ((e << 10) + ((m + 0x1000) >> 13));   // '+' lets the mantissa carry into the exponent
}
```

**Sequencing hazard — read this before landing it.** `f32ToF16` has three call sites: `room/wire.js:20` (the wire), `engine/gguf.js:414` (`quantizeQ8`, used by `requantQ8Streaming` for every K-quant tensor at load — including `output.weight`, which is Q6_K in this GGUF and becomes the 1.35 GB Q8 LM head), and `engine/quant.js:14` (selftest only). In `quantizeQ8`, `d = f16ToF32(h)` — a halved scale means every element of that block gets clamped at ±127, i.e. **the block's largest weights are clipped to half magnitude**. At 3.3e-4 that is ~13,100 of the head's 39.7M blocks. So:

1. **Land the scoped fix first**: a corrected local `f32ToF16` inside `room/wire.js` (or an exported `f32ToF16Exact`). Zero golden risk — nothing in `tests/` imports `room/wire.js`, so the wire path is currently untested. Add a unit test alongside `tests/unit/generator.test.js` covering 1.9999, 0.49999, 7.9999, ±0, ±Inf, NaN, 65504, 65520, subnormals.
2. **Then** fix `engine/gguf.js` globally as a separate commit with a **planned golden re-baseline** — it changes the shipped LM head weights and can move `tests/test_q38_full.js`.

**Also fix `unpackF16` with a 64K lookup table** (256 KB `Float32Array`, one line): **0.206 → 0.013 ms** at 1 hidden, **0.775 → 0.093 ms** at 8 columns, **1.586 → 0.146 ms** at 16 tokens **[M]**. Bit-identical by construction.

Effort: **~4 hours** total including tests.

### 2.2 Multi-token frames and prefill pipelining

**Today** (`room.js:915-938`): the host builds a 16-token round (2 × 8-column GPU passes at the room's `batchCols: 8`), sends `ai-hidden-b`, and **`await returned;` — discarding the returned value**. Nothing downstream reads it. `ai.pos` is the only thing that advances. So the whole chain idles while one device works, and the last worker packs and ships a **160 KB frame that the host throws away** (plus, today, **30 ms of `packF16`**).

Three changes, one protocol bump:

**(a) Replace the discarded prefill return with an ack.** In the worker's `ai-hidden-b` handler (`room.js:1089-1105`), when `!d.spec && ai.next === "host"`, send `{ t: "ai-prefill-ack", basePos, n }` instead of `packWire(hb)`. Needs its own case in the host handler — `room.js:1112` calls `unpackWire(d)`, which throws "unrecognized hidden-state payload" on a payload-less frame. Saves **160 KB of wire and 30 ms of CPU per round on the last hop** (0.23 ms after §2.1; keep the change anyway for the bytes). Note the `spec` path at `room.js:970` genuinely consumes `h`, so the gate must be on `!d.spec`.

**(b) Stream the rounds.** Positions are strictly increasing and waiters are already keyed `"b" + basePos` (`room.js:929`), so rounds cannot collide. Replace the per-round `await` with a **window of 2–3 in-flight rounds** (an explicit cap, not the 90 s timeout at `room.js:930`) and `Promise.all` at the end. Cost model with `R` rounds over `N` stages: `T_sum + (R−1)·T_max` instead of `R·T_sum`. For a 100-token prompt on a 3-device room at 50 ms RTT that is roughly **5.9 s → 1.9 s [E]** — modeled, not measured; room prefill has never been benchmarked separately.

**(c) The prerequisite that makes (b) safe — the per-peer executor.** `room.js:227` dispatches `aiOnData(from, d)` **fire-and-forget**, and `engine/qwen35.js:507/704` share one `stageXB` staging buffer and one `B.x` activation buffer per engine. Two in-flight rounds would (i) reject with `OperationError` on the double `mapAsync` and (ii) submit columns out of position order, advancing the DeltaNet recurrence wrongly. This is `docs/kernel-plan-3.md` §1 prerequisite #1. Build it once as a per-peer promise chain covering `ai-hidden`, `ai-hidden-b` **and** `ai-rollback` (rollback must be *sequenced*, not applied on receipt as it is at `room.js:1107`), with `ai-reset` clearing the chain. `roadmap/09` (lap overlap) consumes the same seam.

**(d) Fold the prefill tail.** The batched loop runs `while (ids.length - 1 - i >= NC)` with `NC = 8`, so a 100-token prompt leaves 3 leftover tokens plus the final token as **4 serial single-token laps** at RTT + 2 hops each. `embedRunBatch`/`runHiddenBatch` already accept `n < NC` and the worker handler already chunks with `Math.min(NC, nTok - c)` — loosen the loop bound to `>= 1` and only the final token's lap (which needs logits and `ai.lastHidden` for the draft head, `room.js:869`) stays outside the stream.

Effort: **1 week** including the executor, the ack case, and a protocol version constant (there is none today — `docs/protocol.md` promises a message version bump but `ai-load` at `room.js:777` carries no version field; building it is part of this work).

**Also fix the solo decode path while you are here.** `aiPipeToken` (`room.js:850-877`) does `embedRun` (submit + `mapAsync` of the 20 KB hidden) **then** `headFromHidden` (`writeBuffer` + second submit + `mapAsync` of 993 KB) — **two renderer↔GPU-process round trips per token and a full drain between the last layer and the head**, which is exactly the cost `forwardToken`'s own comment says it removed. `forwardToken` is only called from benchmarks and tests. One-line fix: when `!ai.chain.length && needLogits`, call `ai.engine.forwardToken(id)` directly. Expected **~11 ms/token on Deno, ~1.5 ms on Chrome [E]**. This also means the published GB10 9.0 (from `bench.js`/`forwardToken`) and the Mac 6.7 (from the room) **measure different code paths** — worth a bench-log note.

### 2.3 Activation quantization, with measured error bounds

Measured on real hidden states at the layer-32 split boundary, 26 prompt positions, `dim = 5120`, `max|x| = 278.7`, 3 "massive" channels (>20× the median channel max = 1.394). SCTP packets computed at **1,160 B of DATA payload per packet** (dcSCTP `mtu = 1191`, minus 12 B common header and 16 B DATA chunk header).

| codec | bytes | SCTP pkts | rel L2 (all) | rel L2 (ordinary ch.) | max abs err | elements zeroed/tok |
|---|---|---|---|---|---|---|
| f32 | 20,480 | 18 | 0 | 0 | 0 | 0 |
| **f16 (current, buggy packer)** | 10,240 | 9 | **2.93e-3** | 5.55e-3 | **1.00** | 0 |
| **f16 (fixed packer)** | 10,240 | 9 | **1.98e-4** | 2.06e-4 | 5.97e-2 | 0 |
| int8 block-32 (Q8_0 layout) | 5,440 | 5 | 1.06e-2 | 2.01e-2 | 1.04 | 73.2 |
| int8 block-64 | 5,280 | 5 | 1.44e-2 | 2.74e-2 | 1.04 | 102.1 |
| int8 per-token (1 scale) | 5,122 | 5 | **1.20e-1** | 2.29e-1 | 1.10 | **1898.1** |
| **int8 per-token + 16 f16 outliers** | 5,186 | 5 | **5.95e-3** | 1.13e-2 | 5.97e-2 | 100.2 |
| int8 per-token + 32 f16 outliers | 5,250 | 5 | 4.75e-3 | 9.02e-3 | 5.97e-2 | 78.3 |
| int6 block-32 | 4,160 | 4 | 3.51e-2 | 6.67e-2 | 2.90 | 275.8 |
| int4 block-32 | 2,880 | 3 | 9.46e-2 | 1.80e-1 | 6.22 | 1084.2 |
| fp8 e4m3 + per-token scale | 5,122 | 5 | 1.58e-2 | 2.66e-2 | 3.42 | 0.8 |

Three conclusions the data forces:

1. **Per-token/per-tensor int8 is unusable on this residual stream.** One scale for 5,120 values with three channels at |278| flushes **1,898 of 5,120 elements to zero per token**. Block granularity — *or* an explicit outlier list — is what saves it.
2. **The best 5-packet format is int8 per-token + a 16-entry f16 outlier list, not Q8_0 block-32.** 5,186 B vs 5,440 B and **1.8× lower error** (5.95e-3 vs 1.06e-2). The outlier list is exactly what removes the massive channels from the scale. Q8_0 block-32's advantage is that `engine/gguf.js:405` `quantizeQ8` and the existing dequant path already exist; the outlier variant is ~30 lines more.
3. **int6 (the only 4-packet format) is 177× worse than fixed f16** and should not ship for a wire whose packet count is not the binding constraint at 8 columns (see §2.4).

**Does it change the output?** `wire_gate2.js`: two-shard split at layer 33, codec applied at the hop, 3 open-ended prompts × 64 greedy tokens = 192 steps.

| codec | steps identical to f32 | first divergence |
|---|---|---|
| f16 (buggy) | 192/192 | none |
| f16 (fixed) | 192/192 | none |
| int8 blk-32 | 146/192 | **prompt 1, token 18** |
| int8 pt + 16 outliers | 192/192 | none |
| int6 blk-32 | 192/192 | none (luck — see below) |
| int4 blk-32 | 157/192 | prompt 1, token 48 |

Read this carefully. Once a greedy stream diverges it never re-converges, so "146/192" means *one* divergence at step 18 of one prompt. And int6 scoring 192/192 despite 3.3× int8's error is **luck at n=3 prompts**, not evidence. What actually predicts divergence is the codec's rel L2 against the top1–top2 logit margin distribution **[M]**: p5 = 0.183, p25 = 0.72, median = 1.77; **4 of 192 steps have a margin below 0.1**. Any codec at ~1e-2 rel L2 will flip near-ties at a rate of order one per hundred tokens. Fixed f16 at 2e-4 is ~50× under that threshold. int8+outliers at 6e-3 is borderline.

**Recommendation:**

- **Do §2.1 first.** It makes the shipping wire 15× more accurate at zero cost and creates the headroom to reason about anything lossier. Today's wire is *already* at 2.93e-3 — an int8+outliers wire at 5.95e-3 would be only 2× worse than what is shipping and passing.
- **Do not ship any activation quantization until the link speed is measured.** `room.js:296-308` already has a bandwidth probe (`bw-start`/`bw-end`, 64 KiB chunks) whose result is only rendered on a peer card. Log it. At 10 Mbps an 8-column frame's 38.4 KB saving is 30.7 ms/hop; at 50 Mbps it is 6.1 ms/hop; on same-Wi-Fi it is under 1 ms. `docs/bench-log.md` contains no measured cross-network link speed.
- If it ships: **int8 per-token + 16 f16 outliers**, `enc: "q8o"` alongside the existing `f16` flag in `room/wire.js` so older peers keep working, negotiated once per mesh (mixed-precision peers diverge from solo and break `tests/test_*_split.js`). Quantize in a WGSL epilogue (`pack4x8snorm`/`pack4xI8` are core WGSL) so the *readback itself* shrinks from 20 KB to 5.2 KB, not just the wire.
- **The gate is not the goldens** — the wire is already lossy and `WIRE_F16` is already on. The gates are: (i) run `tests/test_split.js`, `test_qwen_split.js`, `test_q38_split.js`, `test_mtp_split.js` with the codec enabled; (ii) **MTP acceptance rate** at K=3 and K=7 — wire noise perturbs both the draft head's hidden and the trunk's, so acceptance can drop, and acceptance is what buys 9 → 16 tok/s. **UNCERTAIN: unmeasured.** That measurement (spec decode over a two-shard split with the codec injected) is the go/no-go and should run before the kernel is written.

### 2.4 DataChannel settings — the real constraint is cwnd, not `max_burst`

Verified from Chrome's dcSCTP (`net/dcsctp`, used by Chrome and Safari 26): `mtu = 1191` (1,160 B of DATA payload), `cwnd_mtus_initial = 10`, `max_burst = 4` packets per `SendBufferedPackets` call, that function invoked once per `Send()` and once per received SACK, receiver SACKs every 2nd packet, slow start adds `min(bytes_acked, mtu)` per SACK **only when fully utilized**, and **cwnd is never decayed on idle** (`OnApplicationLimited` has an empty body).

Verified from the PeerJS 1.5.4 bundle actually loaded at `p2p.html:13`: `_send` is `if (!t && n.byteLength > this.chunker.chunkedMTU) { this._sendChunks(n); return }` with `chunkedMTU = 16300`. **`chunkedBrowsers = {Chrome:1, chrome:1}` is assigned in the constructor and never read by the send path** — it is dead legacy state, so chunking happens on Safari too.

This changes the standard conclusion. Per frame:

| frame | bytes | PeerJS chunks (`Send()` calls) | burst allowance (4/chunk) | packets | binding constraint |
|---|---|---|---|---|---|
| `ai-hidden` (1 hidden, f16) | 10,240 | **1** | **4** | 9 | **`max_burst`** |
| `ai-hidden-b` 8-col verify | 81,920 | 6 | 24 | 71 | **cwnd (10)** |
| `ai-hidden-b` 16-tok prefill | 163,840 | 11 | 44 | 142 | **cwnd (10)** |

- **The 10 KB single-hidden frame is `max_burst`-limited and pays ~1 extra RTT per hop, forever** — it is one `Send()`, so only 4 packets go out, the remaining 5 wait for SACKs. Fix: pre-slice the activation payload yourself into 3 messages of ≤4,600 B with an 8-byte header `{basePos, idx, total}` and call `conn.send()` per slice (each is under `chunkedMTU`, so PeerJS passes it straight to `_bufferedSend`). 3 calls = 12-packet allowance ≥ 9 packets, and cwnd 10 ≈ 9, so the frame goes in one tick. **Saves ~1 RTT per hop [E]** on every non-spec decode lap and on every prefill-tail lap. Effort: half a day.
- **The 80 KB and 160 KB frames are cwnd-limited, and finer chunking buys nothing** — 6 `Send()` calls already authorize 24 packets while cwnd authorizes 10. This is the correction to the usual "chunk everything" advice. Cold delivery from cwnd=10 with +1 MTU per SACK: 10 → 15 → 22 → 33, i.e. **~3.5 RTT for 71 packets**. But **cwnd never decays**, so it reaches 71+ within ~2 laps and thereafter the frame is **0.5 RTT + serialization**. The SCTP cost on multi-column frames is a **2–3 lap warm-up, not a steady-state tax.**
- Therefore the lever on the big frames is **fewer bytes** (§2.3: 80 KB → 43.5 KB warms in one lap instead of two) or nothing at all. Do not spend a week on framing for them.
- **Parallel DataChannels do not help** — they share one SCTP association, one TCB, one cwnd, one burst budget. A second `peer.connect()` creates a separate `RTCPeerConnection` and would work, but chunking makes it unnecessary.
- Add `dataChannel.bufferedAmount` backpressure on the new slice loop (the pattern already exists in `bwTest`, `room.js:304`).
- **Log the link speed.** `bw-result` (`room.js:308`) already computes Mbps and only writes it to a peer card. Put it in the room crumb and in the bench report; every serialization number in §2.3 is conditional on it.

---

## 3. Ranked by expected gain per engineering week

Gain columns: **solo** = ms/token on a GB10 decode token (108 ms baseline, measured); **room** = per 3-hop 8-column verify lap. Room figures are modeled from measured components and are marked as such.

| # | Item | Solo gain | Room gain | Effort | **Gain / eng-week** | Confidence |
|---|---|---|---|---|---|---|
| **1** | **f16 packer: hoist scratch + `+` carry fix (scoped to `room/wire.js`)** | 0 | **−45 ms/lap [M]**, wire error 2.93e-3 → 1.98e-4 | **0.1 wk** | **~450 ms/wk** | measured |
| **2** | **f16 unpack via 64K LUT** | 0 | **−2.0 ms/lap [M]** (3 hops × 0.68) | 0.05 wk | ~40 ms/wk | measured |
| **3** | **`aiPipeToken` → `forwardToken` in solo mode** | **−11 ms (Deno) / −1.5 ms (Chrome) [E]** | 0 | 0.05 wk | ~30–220 ms/wk | high |
| **4** | **Pre-slice 10 KB frames to ≤4.6 KB (`max_burst`)** | 0 | **−1 RTT/hop on single-token laps [E]** | 0.1 wk | large, cross-net only | source-verified |
| **5** | **Encode-ahead (double-buffered encoding)** | **−7.45 ms/token [M]** | −7.45 ms/device/lap | 0.5 wk | **~15 ms/wk** | measured |
| **6** | **One-submit spec step (GPU `embed_gather`)** | **−52 ms/step K=3, −105 ms K=7 (Deno) [M]**; **−6/−12 ms (Chrome) [E]** | same, host-side | 1.0 wk | 52 ms/wk Deno, 6 ms/wk Chrome | measured ceiling; Chrome UNCERTAIN |
| **7** | **Prefill: ack instead of 160 KB return + round pipelining + tail fold + per-peer executor** | 0 | **TTFT ~2.5–3× [E]**; −160 KB/round | 1.0 wk | large on TTFT | modeled |
| **8** | **KV append via dynamic offsets (−32 blits, −32 pass splits; −256 blits per 8-col verify)** | ~0 [M] | ~0 | 0.4 wk | **0 on GB10**; Metal UNCERTAIN | measured neutral class |
| **9** | **Fused QKV glue (`qsplit`+2×`head_norm`+2×`rope`+`sigmoid_mul`)** | ceiling **2.84 ms [M]**, ~1.4 recoverable [E] | same | 0.6 wk | ~2 ms/wk | measured ceiling |
| **10** | **int8 per-token + 16 f16 outliers on the wire** | 0 | −38 KB/hop → **−31 ms/hop @10 Mbps, −6 ms @50 Mbps [E]** | 0.8 wk + acceptance gate | conditional on link speed | error measured; link speed UNCERTAIN |
| **11** | **Fused attention (one WG/head, serial in-WG sum)** | **0 @pos8, −5.5 ms @pos500 [M]**; ~2.5 ms averaged [E] | same | 1.0 wk | ~2.5 ms/wk today | measured ceiling |
| **12** | **Concatenate the 4 DeltaNet GEMVs (−144 dispatches)** | ~0 [M] | ~0 | 0.6 wk | 0 on GB10; Metal UNCERTAIN | inferred |
| **13** | **GPU exact top-40 sampling** | **−1.1 ms/step [M]** | −1.1 ms | 0.8 wk | **~1.4 ms/wk** | measured |

**Suggested order for one maintainer:** 1 → 2 → 3 → 4 (all in under a week, all measured, all low-risk) → 5 → 7 → 6 → 10 (gated on a link-speed measurement) → 9 → 11 → 8/12 (gated on a Mac dispatch measurement) → 13.

### Two things to measure before spending the weeks

1. **Chrome's `mapAsync` floor**, on the Mac and the GB10. Port `probe30.js`'s SYNCFLOOR block to a page (20 lines). It re-prices item 6 by a factor of ~8 and it re-prices every GB10 bench-log row. If it comes back at 1–2 ms as expected, say so in the paper and report GB10 numbers floor-subtracted.
2. **The cross-network link speed**, from the probe `room.js` already runs. Every byte-reduction number in §2.3/§2.4 is conditional on it, and it is not recorded anywhere in the repo.

### One thing that is not in this document because it is not addressable

**80 of the 108 ms is matvec at 183 GB/s against a measured 184 GB/s roofline [M].** Nothing in Part A touches it. The full addressable non-GEMV budget on GB10 is 28 ms at pos 8, of which 11.6 ms is a Deno artifact and ~4.5 ms is CPU encode. That leaves **~12 ms of real GPU glue** to fuse. Items 5, 6, 7 and 1–4 are worth more than every fusion in this document combined, and none of them is a kernel.