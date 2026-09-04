# Closing the Mac gap — implementation-ready plan

**Scope:** Chrome/Metal on the maintainer's MacBook, Qwen 3.8 27B Q4_0. Target: explain and close the difference between the Mac's 6.7 tok/s plain / 10.8 tok/s speculative and its bandwidth roofline, without breaking the bit-exact gate, WGSL-only, Chrome + Safari 26.

Everything below is either **verified** (file:line from the repo as read, or arithmetic re-derived from it) or explicitly marked **UNCERTAIN**. Nothing here assumes a number that is not in the tree.

---

## 0. What is actually established

### 0.1 The per-token command structure (verified, read from `engine/qwen35.js:366-423`, `946-961`)

`forwardToken` (the benchmark path) encodes, per token:

| Layer kind | Count | Dispatches each | Compute passes each | Blit copies each |
|---|---|---|---|---|
| Gated-DeltaNet (`else` branch, 397-411) | 48 | 13 | 2 | 0 |
| Full attention (`if L.isFull`, 369-395) | 16 | 17 | 3 | 2 (`383-384`) |
| Final norm + LM head (`952-957`) | 1 | 2 | 1 | 0 |

**Totals per decode token: 898 dispatches, 145 compute passes, 32 `copyBufferToBuffer` (K/V append), 1 submit** — plus a second submit + `mapAsync` of 993,280 B inside `_readback` (`454-461`).

The batched path `_encodeLayerBatch` (`641-696`) at the room's `nCols = 8`:

| | dispatches | passes | copies |
|---|---|---|---|
| DeltaNet layer | 13 | 2 | 0 |
| Attention layer | 9 + (3×8 + 2) + 3 = **38** | 3 | **16** |
| **Per 8-column verify pass** | **1232** | **144** | **256** |

At `nCols = 4` (the twin-kernel path, K=3): 1040 dispatches, 144 passes, 128 copies.

The 3·nCols term is the per-column serial attention loop at `659-668` — all columns share one `scores` scratch, so they cannot run concurrently.

`docs/kernels.md` trick 9 says "~1,100 dispatches"; that is the pre-`_acc` decode number. Trick 11 claims `_acc` removed "192 dispatches/token", but there are exactly **two** `add_res` sites per layer in both branches (`392`/`407` and `421`), so the real figure is **128**. Fix both lines while you are in there. (`dn_pre` removed a further ~48–96; UNCERTAIN which, since the pre-fusion tree is gone.)

### 0.2 The two headline numbers do not measure the same thing (verified — this is the single most important finding)

| | GB10 9.0 tok/s plain | Mac 6.7 tok/s plain |
|---|---|---|
| Harness | `benchmarks/bench.js:71` → `eng.forwardToken` | `room.js:850` `aiPipeToken` → `embedRun` + `headFromHidden` |
| Runtime | Deno / wgpu / Vulkan | Chrome / Dawn / Metal |
| Submits per token | 1 (+1 readback copy) | **3** (`embedRun` layers, `_readback` copy, `headFromHidden` head+copy) |
| `mapAsync` per token | 1 (993 KB) | **2** (20 KB hidden, then 993 KB logits) |
| GPU drain between last layer and head | none | **yes** — `embedRun` awaits the hidden before the head is even encoded |
| Batch config | `BCOLS=4`, WG=256 ROWS=4 fixed | `batchCols: 8, coopRowsB: 2` (`room.js:664`), WG/ROWS from autotune |
| Commit | current | **Aug 31**, `71b7b85..9f8b852` |

`forwardToken`'s own comment says the hidden-state round trip "cost a full pipeline drain" — and the room still pays it. `room.js` never calls `forwardToken`.

The Mac row also predates `_acc` (−128 dispatches) and `dn_pre` (−48 to −96), at **identical pass count** (145 then, 145 now). That is a free natural experiment on per-dispatch cost, described in Step 1 below.

**UNCERTAIN:** whether the Mac 6.7 was solo or the host of a split room. `NEED_GB["qwen3.8-27b"] = 16.5` (`room/models.js:3`) means a solo Mac must have ≥24 GB unified memory, i.e. a Pro/Max-class part. Record the chip before anything else.

### 0.3 Roofline arithmetic

≈15 GB of weights stream per token (`docs/kernels.md`, "Where the time goes"). Observed Mac plain = 149 ms/token.

| Plausible chip | Spec BW | Weight-stream floor | Residual vs 149 ms |
|---|---|---|---|
| M4 (base, 120 GB/s) | 120 | 127 ms | 22 ms |
| M2/M3 Pro (150–200) | 150–200 | 101–76 ms | 48–73 ms |
| M4 Pro (273) | 273 | 56 ms | 93 ms |
| M1–M4 Max (400) | 400 | 38 ms | 111 ms |

Combined with the ≥24 GB constraint, the Mac is almost certainly Pro/Max-class, so the residual is **73–111 ms/token** — roughly half to three quarters of the token. That is a real gap, not a rounding error. But the achievable WebGPU streaming rate is what matters, not spec BW (on GB10 it is 184 of 273 GB/s = 67%), so **Step 1 measures it rather than assuming it**.

Speculative path, which is what the product actually ships: Mac 10.8 tok/s at K=3 and 85% acceptance = 3.55 tok/step = **329 ms per verify step**, versus 221 ms on GB10 (16.07 tok/s). Each step is 3 draft submits + 3 blocking 16-byte `mapAsync` (`qwen35.js:768-797`), one `embedRunBatch` submit + 80 KB `mapAsync`, one `headBatch` submit + **3.97 MB** `mapAsync` (`743-761`), plus up to 3 refill submits — about **5 blocking renderer↔GPU-process round trips and 9 submits per 3.55 tokens**.

---

## 1. Hypotheses, ranked by prior probability × cost-to-test

| # | Hypothesis | Why it is plausible | Discriminating measurement |
|---|---|---|---|
| **H0** | The 6.7 number is a harness artifact: 2 extra submits, 2 extra `mapAsync`, one extra full drain, and a stale commit (§0.2) | Verified structural difference; costs nothing to exclude | Step 1: re-run at HEAD, then A/B `aiPipeToken` vs `forwardToken` on the same tab |
| **H1** | The Mac is near its achievable roofline and there is no gap | Achievable BW on Apple through WebGPU is unmeasured here; GB10 achieves 67% of spec | Step 2 bandwidth probe → compare `15 GB / BW_measured` against 149 ms |
| **H2** | Per-dispatch / per-encoder cost on Metal: 145 compute encoders + 16 blit encoders + 898 dispatches per token; Dawn records compute passes as `MTLDispatchTypeSerial` | `docs/kernel-plan-2.md:23` already estimated 31.7–71 µs/dispatch on Metal → 28–64 ms of 149; trick 11 measured these free on Vulkan and explicitly "kept for Metal", never verified there | Step 1 (dispatch-count delta at constant pass count), Step 4 (skip-family), Step 3 (synthetic dispatch/encoder floor sweep) |
| **H3** | Chrome CPU cost: ~3,600 wire commands/token (setPipeline + 2×setBindGroup + dispatch × 898) + 145 pass begin/end, serialized renderer→GPU process, never overlapped with GPU work | Absent on Deno/wgpu, so invisible in every GB10 number | Step 5: `performance.now()` split of writes / encode / submit / wait |
| **H4** | Tint MSL codegen on the GEMV: robustness clamps on ~10 runtime-sized-array accesses per inner iteration per row, `unpack4xU8` polyfilled to shifts (so trick 6 buys nothing on Metal), workgroup zero-init of `mvc_part` | `coop.js:306-341`; Metal has no hardware robust buffer access | Step 6: Dawn toggle ablations + dumped MSL |
| **H5** | Wrong GEMV shape for Apple: `autotune.js:6` candidates are `[[256,4],[128,4],[256,8],[128,8],[64,4]]` — no WG=32 (Apple SIMD width), no ROWS=2, no ROWS=16, and a 3% bias toward (256,4) | Apple SIMD is 32-wide; `kernel-plan.md:35` flags this | Step 7: widened sweep in the browser |
| **H6** | The latency-bound small kernels scale worse on Apple: `attn_softmax` is `@workgroup_size(1)` with 24 threads total and three serial `seqLen` loops (`base.js:185-198`); `head_norm` is 32-thread WGs with serial 256-loops; `dn_delta` is 48 WGs × 128 threads doing two serial 128-iteration read-modify-write sweeps of a 3 MB state per layer (`wgsl/qwen35.js:76-105`) — ≈576 MB/token of state traffic, 5.6 ms on GB10 | These are occupancy-starved everywhere; on a lower-BW device the state traffic alone costs more | Step 4 skip-family per kernel family |

H0 and H1 are **gates**, not hypotheses to optimize against. Run them first; if H1 holds, most of the plan below is void.

---

## 2. Profiling protocol (Chrome, on the Mac)

Two new artifacts are required, because **every file in `benchmarks/` is Deno-only** (`Deno.open`, `Deno.env`, `../models/q38/model.gguf`) **and `benchmarks/` is listed in `.vercelignore`** — it is not deployed. So:

- **`/bench.html` at the repo root** — synthetic micro-benchmarks, no model, imports from `engine/` (which *is* deployed). Runs in under a minute.
- **`#probe` mode in `p2p.html`/`room.js`** — model-resident probes reusing the weights already in the Cache API (`swarmllm-weights-v1`, same origin). `room.js` has no URL-param handling today (`#debug` at `room.js:585` is the only hash branch); add `location.hash.includes("probe")`.

### Step 0 — environment (10 min, no code)

```bash
system_profiler SPHardwareDataType SPDisplaysDataType | sed -n '1,40p'
sysctl -n hw.memsize
pmset -g | grep -i lowpower
```
Record: chip, GPU core count, unified memory, macOS version, Chrome version, and the `Dawn Info` block from `chrome://gpu` (backend must read Metal). Plug in the laptop, close every other tab and GPU-using app.

Launch a clean Chrome for every run:

```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --user-data-dir=$HOME/chrome-bench --no-first-run --disable-extensions \
  --disable-background-timer-throttling --disable-backgrounding-occluded-windows \
  --disable-renderer-backgrounding \
  "https://<staging-host>/room#debug"
```

Thermal guard, in a second terminal during every timed run:

```bash
sudo powermetrics --samplers gpu_power -i 500 -n 60
```

Report GPU active residency and frequency. **Residency ≥95% at steady frequency ⇒ the GPU is the bottleneck (H1/H2/H4/H5/H6). Residency <70% with idle gaps ⇒ CPU/IPC/dispatch (H0/H2/H3).** A >15% frequency decline over a 300-token run means you are reporting a thermal ramp, not a steady state.

### Step 1 — re-measure at HEAD (30 min, no code)

Run the room solo on the 27B, one fixed prompt, 64 decode tokens. Record: plain tok/s (non-MTP model or MTP disabled), spec tok/s, prefill tok/s, and the `autotune: WG=.. ROWS=..` crumb (`room.js:599`).

**Decision rule.** The Aug 31 Mac build ran ~1024–1072 dispatches per token at 145 passes; HEAD runs 898 at 145. If HEAD returns ≈149 ms/token unchanged, per-dispatch cost on Metal is ≲2 µs and H2 is largely dead. If HEAD returns ≈130–140 ms, per-dispatch cost is ~50–120 µs and H2 is the story. This costs one page load and is the cheapest discriminator in the whole plan.

### Step 2 — achievable bandwidth (`/bench.html`, 30 min)

Allocate one buffer of `min(adapter.limits.maxStorageBufferBindingSize, 512 MB)` and time three kernels after ≥300 ms of sustained warm-up (the `benchmarks/bench_gemm.js:92` pattern), min of 5 interleaved rounds:

- (a) streaming read as `array<vec4<u32>>`, one 16 B load per thread per iteration;
- (b) the same as scalar `array<u32>` (the shape `matvec_q4_coop` actually uses, `coop.js:307`);
- (c) `copyBufferToBuffer` of the same buffer.

Report GB/s for each. **(a)/(b) is how much Apple charges for 4-byte weight loads** and directly sizes fix F7.

**Gate.** Let `BW` = the best of (a)/(b)/(c). Roofline `R = 15 GB / BW`. If `149 ms − R < 25 ms`, the Mac is already at roofline: stop this plan, and the only remaining levers are fewer bytes (f16 KV, smaller model) and more speculation. Otherwise continue with the residual `149 − R` as the budget to attribute.

### Step 3 — dispatch and encoder floor (`/bench.html`, 1 h)

Two sweeps, both synthetic, both timed submit → `onSubmittedWorkDone`:

1. **Dispatch floor.** One encoder, one pass, N ∈ {1, 10, 100, 1000} dispatches of a kernel; slope = µs/dispatch. Run it at three workgroup counts: 1 WG (launch cost only), 48 WGs (`dn_delta` shape), and 4352 WGs (`ceil(17408/4)`, the FFN GEMV shape). **Do not use only the 1-WG no-op** — `docs/kernel-plan-3.md:18` records a ~50 µs floor for *real small GEMVs* on GB10, which is memory latency and occupancy, not launch cost; a no-op measures the wrong thing and will under-predict by an order of magnitude.
2. **Encoder floor.** Same total dispatch count, but N passes of 1 dispatch each (`beginComputePass`/`end` per dispatch) vs 1 pass of N. The delta is the `MTLComputeCommandEncoder` cost. Repeat with a `copyBufferToBuffer` between passes to price the blit-encoder switch (32 of these per decode token, 256 per 8-column verify).

**Decision rule.** Predicted overhead = `898 × floor_µs + 145 × pass_µs + 16 × blit_µs`. Compare against the Step-2 residual. Within ~30% ⇒ H2 confirmed, go to F3/F4/F5. Under a third of the residual ⇒ H2 is secondary.

### Step 4 — kernel-family attribution (`#probe` in the room, 1–2 h)

Port `benchmarks/bench_breakdown.js` into the probe page. `eng.skip` is a plain `Set` checked at `qwen35.js:346, 353, 478, 485`, so no engine change is needed — but **the skip lists in that file are stale and will silently under-report**:

- The matvec families list `matvec_q4_coop`, `matvec_q8_coop`, `matvec_coop`, `matvec_*_gu` but **not the `_acc` variants**, which is what `mvO`/`mvOut`/`mvDown` actually dispatch under coop (`qwen35.js:186`, built with `acc=true` at 214/223/247). `ffn_down` alone (5120×17408) plus `wOut` is roughly a third of streamed weight bytes — so the "all matvecs" row today misses ~30% of the weight traffic and dumps it into "everything else". The same defect exists on the batched list (`_coop_b_acc` missing).
- `dn_gates` / `dn_l2` are no longer dispatched at all; the kernel is `dn_pre` (`qwen35.js:403`) / `dn_pre_mc` (`677`).

Fix both lists, or better, derive families by recording `op.pipe` / `name` into `this.dispatched` on one warm pass and grouping by prefix, so they can never go stale again. **Then re-run on GB10 too**: the 82 ms / 30 ms split in `docs/architecture.md` and `docs/kernels.md` was produced by this tool after those two commits landed, so it is suspect on both machines.

Add `eng.tiny`: dispatch every matvec with 1 workgroup. Then per token:
`t_full − t_tiny` = weight streaming; `t_tiny − t_skipped` = matvec dispatch/encoder overhead; remainder = small kernels.

Run at 1 column (decode) and at 8 columns (the verify pass the product actually uses).

### Step 5 — CPU/GPU split (`#probe`, 1 h)

Port `benchmarks/bench_enc_probe.js` (it is uncommitted; commit it) to the browser. Per token, medians over ≥25 tokens of: `writes` (frame uniform + embedding row), `encode` (createCommandEncoder → finish), `submit`, `wait+readback`. Add an **empty round-trip** measurement: one 1-WG dispatch + a 16-byte `mapAsync`, 100 reps — that is Chrome's fixed IPC floor per readback, and it multiplies by 5 in every speculative step.

**Decision rules.** `writes + encode + submit > 20 ms` ⇒ F2 (encode-ahead) pays. Empty round trip > 1 ms ⇒ F8/F9 (fewer readbacks) pays.

### Step 6 — Dawn toggle ablations (30 min, zero code)

Three separate launches, each re-running Steps 4 and 5:

```bash
--enable-dawn-features=disable_robustness
--enable-dawn-features=disable_workgroup_init
--enable-dawn-features=dump_shaders,disable_symbol_renaming
```

Confirm each is listed as enabled in `chrome://gpu` → Dawn Info. Dumped MSL is emitted via Dawn's `EmitLog`, so **check the page's DevTools console first**; `--enable-logging=stderr` is a fallback, not a requirement. `disable_symbol_renaming` is mandatory or the dump is `tint_symbol_N` soup and you will not find `matvec_q4_coop`.

Read the dumped `matvec_q4_coop` for three things: (i) are `q4_qs[...]`/`q4_sc[...]` wrapped in `min(i, len-1)`; (ii) did `acc0..acc3` stay in registers (trick 2's spill bug is compiler-specific and can recur under Tint); (iii) did the block `for` loop unroll.

Also run the goldens under `disable_workgroup_init` — a mismatch means some kernel reads `var<workgroup>` before writing it and is relying on spec-mandated zeroing. That is a portability bug worth finding regardless of timing.

**Production cannot disable robustness.** These toggles only attribute cost; the shippable fix is fewer, wider accesses (F7).

### Step 7 — shape sweep (`/bench.html`, 1 h)

Sweep `(WG, ROWS)` over {32, 64, 128, 256} × {2, 4, 8, 16} on the two real decode shapes (17408×5120 and 5120×5120) plus the LM head (248320×5120, which needs the 2-D dispatch path). Also A/B `UNPACK=true` vs `false` — Tint polyfills `unpack4xU8` into exactly the shift/mask sequence of the fallback on MSL, so **trick 6 is very likely a no-op on Chrome/Metal despite being "kept for Metal"**; this is a one-flag, ten-minute test.

Also sweep `batchCols` × `coopRowsB` — the room hardcodes `8 × 2` (`room.js:664`) and that pair has never been tuned on Metal.

**Caveat for goldens:** changing `WG` changes the reduction partition in `coop.js`'s `stride = WG/2` halving tree and can change last bits and therefore token ids. `ROWS` does not (each row still reduces over the same WG lanes). So pin `(256, 4)` for any bit-exactness run and report the tuned config separately.

### Step 8 — Safari 26, same page (30 min)

Same URL, same prompt, same probes. Safari ≫ Chrome on identical WGSL points at Tint/Dawn codegen or the Chrome wire; parity points at kernel shape or the hardware.

---

## 3. Fixes, ranked by expected gain per engineering week

Effort is in engineering-weeks for one maintainer including goldens and a bench-log row. "Gain" is on the Mac unless stated. Everything is bit-exact unless flagged.

| # | Fix | Effort | Expected Mac gain | Gain/week | Path affected |
|---|---|---|---|---|---|
| **F1** | Route solo decode through `forwardToken` | **0.05** | 5–15% plain (UNCERTAIN) | **very high** | plain decode, dense models |
| **F2** | Fix `bench_breakdown` skip lists + commit the probes | 0.1 | 0 (measurement correctness) | n/a — prerequisite | all |
| **F3** | Toggle ablations, `UNPACK=false`, widened autotune, `batchCols` sweep | 0.2 | 0–25% (UNCERTAIN; `kernel-plan-2:11` estimated 5–25% for the shape sweep alone) | high | all |
| **F4** | K/V append without blits | 0.4 | decode −32 copies/−16 blit encoders/−16 pass splits; verify −256 copies/−16 pass splits | high **if** Step 3 shows encoder cost | both |
| **F5** | Column-parallel `attn_*_mc` | 0.4 | verify pass 1232 → 896 dispatches at 8 cols; GB10 measured 50 → 12–15 ms at basePos 500 | high | verify + prefill |
| **F6** | Encode-ahead (double-buffered command buffers) | 0.6 | hides `writes+encode+submit`; ~7% measured on GB10/Deno (UNCERTAIN — from the uncommitted `bench_pipe_ab2.js`), likely larger on Chrome | medium-high | all |
| **F7** | Apple-shaped GEMV: `vec4<u32>` weight loads | 1.0–1.5 | only if Step 2/4 show the GEMV far from roofline | medium, conditional | all |
| **F8** | Single-submit draft chain (GPU embedding gather) | 1.0–1.5 | removes K blocking round trips per step | medium, conditional | spec |
| **F9** | GPU top-k / argmax for `headBatch` | 0.8 | removes a 4–8 MB `mapAsync` per verify | medium, conditional | spec |
| **F10** | `attn_softmax` / `head_norm` reshape | 0.5 | GB10 decode softmax alone is 1.3–3.5 ms/token over 16 layers | medium | all |
| **F11** | Register-resident `dn_delta` (spec already written) | 0.5 | ~2% decode on GB10; possibly more on a lower-BW device | low-medium | all |

### F1 — solo decode through `forwardToken` (half a day)

**What.** In `aiPipeToken` (`room.js:850`), when `!ai.chain.length && needLogits`, call `ai.engine.forwardToken(id)` directly instead of `embedRun` + `headFromHidden`. Keep `ai.lastHidden` updated for the MTP draft head where the chain path needs it.

**Why.** Removes one submit, one 20 KB `mapAsync`, one 20 KB `writeBuffer`, and a full pipeline drain between the last layer and the head, per token, on Chrome — where each `mapAsync` is a renderer↔GPU-process round trip.

**Bit-exactness.** Identical kernels, identical order. `forwardToken` is already the path every golden test uses.

**Measure.** A/B in the probe page on the same tab, min-of-5 interleaved, plus the Step-5 CPU split before and after. Report both numbers in the bench log; **and correct the bench-log header to say which harness each column used**, because "GB10 9.0 vs Mac 6.7" is currently not a like-for-like comparison.

### F4 — K/V append without blits (2 days)

**What.** `rope_part` already receives `pos` in `frameBuf`. Have it (or a new `kv_append`/`kv_append_mc`) write K and V straight into `L.kCache`/`L.vCache` at `pos * kvDim * 4`, deleting `qwen35.js:383-384` and `655-658`. `kvDim*4 = 4096 B` on this model (nKV=4, hd=256), a multiple of the 256 B `minStorageBufferOffsetAlignment`, so `hasDynamicOffset` bind groups are a valid alternative — but create a **second pipeline object** for the K/V GEMVs rather than adding dynamic offsets to the shared matvec layout, or every GEMV in the engine has to pass `[0]`.

**Effect.** Decode: 145 → 129 compute passes, 161 → 129 encoders, 32 blits gone. 8-column verify: 256 blits gone, attention layers collapse 3 passes → 2 (144 → 128 passes).

**Bit-exactness.** Same values, different destination. Exact.

**Measure.** Step-3 encoder floor predicts the win before you build it; A/B decode and an 8-column verify pass after.

### F5 — column-parallel attention `_mc` (2 days)

**What.** Today `_encodeLayerBatch:659-668` loops columns and dispatches `attn_scores`/`attn_softmax`/`attn_out` per column against one shared `scores` scratch. Replace with three dispatches keyed on `gid.y = column`, per-column causal length `frame.pos + gid.y + 1u`, and a `scores` scratch sized `nCols × nH × maxSeq` floats (8 × 24 × 512 × 4 = 393 KB at `MAX_SEQ = 512`, `room/models.js:26`).

**Effect.** 8-column verify: attention layers go 38 → 17 dispatches, total 1232 → 896. GB10 measurement for the same change at basePos 500: ~50 ms → 12–15 ms per pass.

**Bit-exactness.** Exact — identical math per column, only the dispatch grid changes. (Do **not** bundle the shared-memory softmax reduction into this step; that changes summation order. See F10.)

**Measure.** Step-4 family timing at 8 columns, before/after; plus end-to-end spec tok/s.

### F6 — encode-ahead (3 days)

**What.** Nothing in a token's command buffer depends on the sampled token except `writeBuffer(x, embedding)` and the frame uniform, and both are queue operations issued before `submit`. After submitting token N, immediately build token N+1's encoder (position is known); when the sample lands, do the two writes and submit the prebuilt buffer. In the speculative path, the verify buffer can be pre-encoded during the draft chain, because `specStep` (`qwen35.js:838`) captures `pos` at entry and the column count is fixed.

**Constraints.** Command buffers are single-submit, so this is re-recording ahead, not reuse. Any event that moves `pos` out of band — `_restoreDN`, `reset()`, a cancelled generation — must discard the prebuilt buffer (dropping an unsubmitted `GPUCommandBuffer` is free). Cross-round pre-encoding in spec mode is **not** available: `pos` after a step is `pos + a + 1` and `a` is only known after verification.

**Bit-exactness.** Exact by construction.

**Measure.** Step-5 split before/after; the `writes+encode+submit` median should collapse to the two writes.

### F7 — Apple-shaped GEMV (1–1.5 weeks, conditional)

**Build this only if Step 2 shows the streaming probe well above `15 GB / 149 ms` AND Step 4 attributes most of the residual to the matvec family.**

**What.** Alias `q4_qs` as `array<vec4<u32>>` (trick 7, already proven legal — `coop.js:349-355`) so a lane loads one whole 32-weight block in one 16 B access instead of four scalar `u32` loads. That is one robustness clamp per 16 B instead of four, and four times fewer memory instructions. Hoist the scale pair. Keep the existing `(nibble − 8) × scale` ordering.

**What NOT to do.** llama.cpp's Metal `yl` trick (pre-scaling x by 1, 2⁻⁴, 2⁻⁸, … and folding −8 into `d·(sum − 8·sumx)`) changes floating-point rounding and therefore breaks bit-identity with the batched `matvec_*_coop_b` kernels, which would make `tests/test_mtp.js` (spec stream must equal plain stream) fail on near-tie tokens. If you want it, apply the same algebra to the batched kernels in the same change, and expect to re-baseline.

**Also note the scope limit.** The LM head and `eh_proj` are Q8, ~10% of streamed bytes; a Q4-only kernel leaves them on the old path.

**Measure.** Prototype as a `/bench.html` variant first (one day), against the Step-2 probe in the same process. Only wire it into `coopWGSL` behind autotune if it wins there.

### F8 / F9 — fewer blocking round trips in the speculative step (conditional)

**Only if Step 5's empty round trip is >1 ms.** F8: put the Q4 embedding table on the GPU (715 MB — affordable on a ≥24 GB Mac, but gate on available memory) and add a gather kernel indexed by the argmax buffer, so the K drafts chain into one command buffer instead of K submits with a `mapAsync` each (`qwen35.js:768-797`). F9: replace `headBatch`'s `n × 993 KB` `mapAsync` with an exact GPU top-40 (matching `room/sampling.js`'s top-k=40, temp 0.8, lowest-index tie-break) so the readback is 2.5 KB. Note `docs/bench-log.md` already recorded GPU argmax as **neutral on GB10 (unified memory)** — this is a Chrome-wire fix, not a bandwidth fix, so it must be justified by Step 5, not by intuition.

### F10 — small-kernel reshape (2–3 days)

`attn_softmax` at `@workgroup_size(1)` (`base.js:185`) is 24 threads for the whole dispatch. Reshape to one 256-thread workgroup per (head, column) with the max computed by tree reduction — **but keep the exp-sum accumulated serially by thread 0** from workgroup memory, or the f32 addition order changes and the goldens break. Same rule for `head_norm`'s `ss`. `qsplit` + 2×`head_norm` + 2×`rope_part` can be fused into one kernel (−4 dispatches per attention layer per token).

### F11 — register-resident `dn_delta`

Already fully specified in `docs/deltanet-prefill-spec.md` (~100 lines, RG=1 is bit-identical). GB10 measured 5.6 → 3.3 ms per decode token across 48 layers. On a lower-bandwidth device the ~576 MB/token of DeltaNet state traffic costs proportionally more, so the Mac gain should be larger than GB10's ~2%. UNCERTAIN by how much.

---

## 4. Branching on what the profiling says

| Step-2/3/4 outcome | Do this |
|---|---|
| `149 − 15 GB/BW < 25 ms` | Stop. Report the Mac as at-roofline, fix the bench-log to say so, and spend the remaining time on speculation depth and prefill. |
| Step 1 alone recovers >10 ms | H2 confirmed cheaply. Order: F4 → F5 → F10 → F6. |
| Step 5 shows `writes+encode+submit` > 20 ms | H3 confirmed. Order: F1 → F6 → F8/F9 → F4. |
| Step 4 shows the matvec family ≫ `15 GB/BW` | H4/H5 confirmed. Order: F3 (free) → F7. |
| Step 4 shows "everything else" ≫ 30 ms | H6 confirmed. Order: F5 → F10 → F11 → F4. |
| `disable_robustness` moves >5% | Budget for F7 regardless of the above; it is the only shippable form of that win. |

**Recommended default order** (before any measurement, on expected value): F1, F2 → Step 0–7 → F3 → F4 → F5 → F6 → conditional set.

Note that F1 and F3 are the only items that are cheap enough to be worth doing before the profiling completes, and F2 must precede any attribution because the current tool under-reports the matvec family by roughly 30%.

---

## 5. Measurement hygiene (bake into the probe page, not into discipline)

1. Warm up to sustained load before timing (`autotune.js:41-42` uses 250 ms first / 40 ms after; `bench_gemm.js:92` uses 300 ms). Take the **min of ≥5 separately-timed batches** and report min/median/max, not a mean of one batch.
2. **Interleave configurations A, B, A** and discard the first slot — `docs/kernel-plan-3.md:561` records a ~55% first-slot clock-ramp penalty. This is the single most common way to manufacture a fake result here.
3. Re-measure the baseline alongside every skip configuration. The current `bench_breakdown.js` measures its baseline once at the top and subtracts from it, which is how it produces negative family costs.
4. Never quote GB/s for a kernel whose time is under ~5× the Step-3 floor for its workgroup count.
5. Record with every row: chip, GPU cores, Chrome version, Dawn toggles, `(WG, ROWS, batchCols, coopRowsB)`, plugged/battery, powermetrics residency and frequency, and the commit. Run the token-id hash pass with `(WG, ROWS)` pinned to `(256, 4)` so the hash is comparable across devices, and the throughput pass with the autotune winner.
6. Every fix lands with a bench-log row on **both** machines, including neutral results, per the non-negotiables.

---

## 6. Open UNCERTAINs to resolve or state in the paper

- The Mac's chip, GPU core count and achievable WebGPU bandwidth are **not recorded anywhere in the repo**. Every "far below roofline" statement, including the one framing this work, is currently unfalsifiable. Step 0 + Step 2 fix that.
- Whether the Mac 6.7/10.8 rows were solo or a split-room host.
- Metal per-dispatch and per-encoder cost: the repo carries two estimates two orders of magnitude apart (31.7–71 µs in `kernel-plan-2.md:23` vs "nearly free" on Vulkan in trick 11). Neither has been measured on Metal.
- Whether `unpack4xU8` (trick 6, "kept for Metal/Android where ALU is scarcer") does anything at all on Chrome/Metal. Tint's MSL backend polyfills it; one flag settles it.
- Safari 26's WGSL compiler is entirely unmeasured on this engine.
- `docs/kernels.md` trick 9 ("~1,100 dispatches") and trick 11 ("192 fewer dispatches") are both stale; the current figures are 898 and 128. `docs/architecture.md`'s 82 ms / 30 ms split was produced by a skip tool with broken family lists and should be regenerated on both machines.