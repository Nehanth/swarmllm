# Bello next-steps plan — ranked by gain/effort on the Mac demo device

## 0. Gate: run the existing benchmark + bit-exact suite on M-series Safari 26 AND Chrome first (0.5–1 day)
- **WHAT**: Mac numbers are pending. Run the golden suite + per-kernel CPU-side timing in both browsers on one M-series machine. Bisect any kernel anomalously slow in one browser. Verified expectation: Safari lands at ~83–90% of Chrome on the same Mac; its real failure mode is tab memory (OOM kill), not a 50x cliff.
- **GAIN**: none direct; every estimate below is unvalidated on the demo device until this exists.
- **RISK**: timestamp-query undocumented in Safari — use CPU timestamps around submits.
- **VALIDATE**: existing golden.json must pass bit-exact in both browsers.

## 1. Workgroup-size sweep for the coop GEMV (1–2 days)
- **WHAT**: parameterize threads×rows-per-WG via string substitution at pipeline build; sweep {64,128,256}×{2,4,8,16} per browser/device. Apple SIMD width is 32; small WGs + more per-thread work raise occupancy on M-series. Keep shared mem <16KB.
- **GAIN**: (a) Mac 5–25% decode; (b) GB10 0–10% (may confirm 256/4).
- **RISK**: optimal differs Chrome-Tint-MSL vs Safari-wgslc — tune per browser. None API-wise.
- **VALIDATE**: reduction order fixed per config ⇒ run golden suite per winning config; bit-exactness holds within a config.

## 2. f16 scale storage, u32-packed (no feature gate) (~2 days)
- **WHAT**: stop widening GGUF f16 block scales to f32 in gguf.js q4Repack/q8Repack. Pack two f16 scales per u32; `unpack2x16float` in the GEMV hot loop (core WGSL — works everywhere unconditionally). Cuts Q4_0 streamed bytes ~10% (20B→18B/block) and ~1.5GB resident.
- **GAIN**: (a) Mac ~3–10% decode + memory headroom under Safari tab limits; (b) GB10 ~3–10% decode.
- **RISK**: none — portable Chrome/Safari/wgpu.
- **VALIDATE**: f16→f32 conversion is exact, so output stays bit-identical to today — golden suite must pass unchanged. Best validation-cost/gain item on the list.

## 3. Dispatch-count reduction / kernel fusion (5–8 days; measure first)
- **WHAT**: count dispatches/token, then drive it down: (1) fuse gate/up projections into one dual-output GEMV; (2) fuse the DeltaNet state-update chain; (3) 7→3-style block tiling. Do NOT lead with residual+RMSNorm-into-GEMV fusion — measured 0.91x on Safari Metal; measure-first only.
- **GAIN**: (a) Mac 15–40% decode (Metal dispatch cost 31.7–71µs is first-order at 64 layers × N dispatches; verified 2.01x on a 7→3 MLP tiling on M2); (b) GB10 single-digit–15% (Vulkan ~1.17x-class).
- **RISK**: Metal-specific payoff — validate on a Mac, not GB10. Fusions that merely merge cheap elementwise ops regress on Metal.
- **VALIDATE**: preserve op order inside fused kernels; golden suite bit-exact per fusion, land one fusion at a time.

## 4. MTP self-speculation (the flagship; ~2–3 weeks, staged)
**WHAT** (staged, each stage gated by the bit-exact suite):
1. **Profile/optimize the batched pass** (2–3d). Batch-4 currently costs ~2.2x a decode step (15.9 vs 8.6); the ideal is ~1.1x. Net speculation speedup ≈ (1+α)/verify_cost — this ratio decides everything. Add a 2-wide variant.
2. **DeltaNet state checkpoint/rollback** (2–3d). Verified prerequisite: dn_delta mutates S in place. Ping-pong `copyBufferToBuffer` snapshots of the 48 layers' S + conv4 state inside the same submit (~150MB, ~0.5ms); full-attention layers roll back by not advancing the KV write pointer. NodeNestor's net LOSS happened even with GPU checkpointing — verify cost is the lever, not the copies.
3. **MTP layer plumbing** (3–4d). blk.64.nextn.*: hnorm(hidden₆₃)‖enorm(embed(t_sampled)) with (1+w)·rmsnorm convention, concat [embed,hidden]→10240, eh_proj (Q8_0 — needs a Q8 GEMV variant; verified in our GGUF, head not lobotomized) → one standard attention block (own tiny KV) → shared_head. Prefill MTP-KV: feed hidden(i)+embed(prompt i+1), batches like layers-only prefill (inferred, not documented — validate).
4. **Verify + head at both positions** (2d): 2-token verify via the batched pass with lm_head at both positions (~2MB readback), accept-if-host-sample-equals-draft.
5. **One-submit greedy draft** (1–2d): GPU argmax over draft logits, argmax buffer used directly as embedding index in the next kernel — preserves one-submit-per-token; readback = trunk logits + 4 bytes.
6. **Lossless temperature sampling** (1d): Leviathan/Chen min(1,p/q) + residual resample; keep q rows GPU-resident; same temperature/top-k transform on p and q; ~1MB extra readback only on rejection.
- **GAIN**: (a) Mac and (b) GB10: **1.3–1.7x decode** at α≈0.7–0.85 IF verify ≤1.2x (GB10 8.6 → ~11–14 tok/s). At NodeNestor's α=0.475 or verify 1.75x: breakeven-to-regression — measure α on our pipeline before trusting projections. K=1, extend to K=2 only if α₁ ≥0.8.
- **RISK**: DeltaNet rollback is the item llama.cpp still lists unfinished; acceptance is architecture/pipeline-dependent (GLM got 51–67% with an intact Q8 head). Domain-dependent (prose worse) — don't promise one number.
- **VALIDATE**: (i) MTP prefill-KV path bit-exact vs sequential MTP decode; (ii) speculate-reject-retry stream bit-identical to never-speculating (the golden test that proves rollback); (iii) greedy-draft mode changes no accepted-token output at T=0.

## 5. shader-f16 activations/intermediates/logits, f32 accumulation (3–4 days)
- **WHAT**: feature-detect `shader-f16` (Safari 26: all Apple GPUs; Chrome 120+; wgpu 25+ — 5-line smoke test on the Deno rig first), dual-variant codegen: vec4<f16> activation loads, f16 shared-mem/intermediates, f16 logits (halves 1MB readback), **all accumulation in f32** (verified: f16 accum made Qwen incoherent on M-series).
- **GAIN**: (a) Mac: prefill 5–15%, decode ~3%; (b) GB10: similar prefill, small decode.
- **RISK**: breaks cross-precision bit-exactness — f16 peers can't mix with f32 peers in a mesh (negotiate one precision per mesh or keep wire f32). Rounding differs per vendor.
- **VALIDATE**: switch the split-vs-solo oracle to the tolerance harness for f16 configs; keep f32 path as the golden reference.

## 6. Distributed-path bundle (mesh demo; ~3–4 days total)
- **WHAT**: (1) binary WebRTC frames — drop f32ToB64, send TypedArrays via PeerJS BinaryPack; second DataConnection for control/chat/bandwidth-test (~1d). (2) f16 wire via core `pack2x16float` (clamp to ±65504) — 20KB→10KB, one SCTP chunk/hop (~1d). (3) prefill 16 tokens/round by looping 4-col GPU passes per network message (~1d; chunk >256KB messages). (4) negotiated data channels + live-RTT placement policy: bypass a weak device at decode when RTT > its compute share; keep it for prefill/memory.
- **GAIN**: mesh-mode only: 33–50% hop bytes, prefill RTT rounds ÷4, per-bypassed-device ~1 RTT/token (4→7 tok/s class on cellular). Solo Mac: zero. MTP (#4) then compounds: 2-token verify halves chain rounds/token; draft lives on the tail tab, zero extra hops.
- **RISK**: keep activation channel ordered+reliable; ≤16KiB/message for Safari interop; f16 wire breaks split-vs-solo bit-exactness (keep f32 debug path).
- **VALIDATE**: split-vs-solo golden with f32 wire; tolerance harness for f16 wire.

## 7. Small hygiene items (each ≤1–2 days, do opportunistically)
- **f16 KV** for the 16 attention layers: 0–3% decode short-ctx, halves KV memory (real for Safari tabs at long context). Never q4 KV (verified −92.5% prefill trap). Keep DeltaNet state f32. Validate: tolerance harness.
- **createComputePipelineAsync** for all pipelines in parallel during weight download: kills multi-second cold-start TTFT, no tok/s change.
- **GPU logits pre-reduction** (per-block max+top-k → few-KB readback): 1–3% Chrome, de-risks Safari mapAsync, 250x less peer traffic. Keep full-readback path for validation.
- **Bind-group dynamic offsets** (one layout, per-layer offsets): 0–5%, encoder-CPU/Metal-object hygiene, helps Safari memory accounting.

---

## REJECT
- **Subgroup GEMV fast path now** — Safari 26 has no subgroups (lands Safari 27), Apple HW gains ~2% anyway, Deno/wgpu can't request it (native-only), and it breaks bit-exactness across mixed meshes. Do the ~1-day dual-epilogue templating as groundwork, default OFF.
- **Separate 0.6B draft model** — Qwen3-0.6B vocab is 151936 ≠ 248320; MTP layer beats any external draft on this topology anyway.
- **Tree/multi-candidate drafting (Medusa/EAGLE-style)** — no trained heads for our 27B, and each tree branch needs its own DeltaNet state snapshot; chain-only is correct for a hybrid-linear model.
- **MTP draft depth K≥4** — acceptance decays to ~0.5 by depth 4; marginal tokens ~α^K; K=1–2 (at most 3).
- **q4 KV cache** — measured −92.5% prefill / −35% decode AND more RAM than f16 on GB10-class HW.
- **int8 KV** — only if 32K+ context memory pressure ever binds; f16 first.
- **f16 accumulation anywhere** — verified incoherent output on Apple M-series.
- **RMSNorm/elementwise fusion into neighbors** — measured 0.91x Safari / 0.95x wgpu on Metal.
- **More command batching, buffer pooling, bind-group caching** — all measured ~0%; one-submit-per-token already saturates this.
- **Pre-sending partial activations** — boundary activation doesn't exist until the last layer's residual add; exo's KV-overlap trick has no decode analog here. Only residue: send boundary readback inside the mapAsync callback (~1ms hygiene).
- **Duplicated boundary layers as latency-hiding** — can't remove a hop while every device stays in the chain; keep only as the decode-time bypass placement policy (item 6.4).

**Suggested order**: 0 → 1 → 2 → 3 (Metal fusion) in the pre-demo window; 4 (MTP) staged behind its verify-cost gate; 6 alongside for the mesh story; 5 and 7 as fill. Items 1+2+3 alone plausibly compound to ~1.3–1.8x Mac decode before speculation lands.
## Results: MTP speculative decoding (2026-09-01)

Implemented. The model's own `nextn` block (blk.64: a full-attention layer plus
`eh_proj`, `enorm`, `hnorm`, `shared_head_norm`; math per llama.cpp's MTP graph)
drafts K=3 tokens per step; one 4-column batched trunk pass verifies them.
DeltaNet state is snapshotted per column inside the recurrent kernels so a
rejected suffix rolls back exactly. Output is bit-identical to plain decoding
(`test_mtp_deno.js`, `test_mtp_split_deno.js`).

GB10, 27B Q4_0:

| path | before | after |
|---|---|---|
| solo decode | 9.1 tok/s | **16.0 tok/s** (85% acceptance) |
| two-device chain decode | 6.5 tok/s | **16.5 tok/s** |
| prefill | 15 tok/s | **24 tok/s** |
| batched 4-column pass | 203 ms | 131 ms (single pass: 110 ms) |

What made the batched pass cheap: every per-column op became one multi-column
dispatch (`*_mc` kernels), and the batched matvecs (`_coop_b`, `_gu_b`) now load
and decode each weight word once for all 4 columns with workgroup 64 (their own
size, `WGB`). The old fused gate/up batched kernel re-read the weights per column.

Measurement lesson: GPUs ramp clocks under sustained load. Micro-benchmarks
need >=250 ms of warm-up or they measure the ramp (the autotuner now warms up).

Parked: 16-byte weight loads in the single-column kernel (+12% on the large
matrices, neutral on square ones) — worth a look after launch.
