# SwarmLLM — the cross-network plan

Implementation-ready design for (A) transport and telemetry, (B) lap overlap / speculative pipelining against the actual protocol, and (C) same-network placement for 2–6 devices. Everything is grounded in the code as it stands at `bello` (`room.js` 1179 lines, `engine/qwen35.js` 963 lines, `docs/protocol.md`, `docs/kernel-plan-3.md` §"Continuous Speculation"). Numbers that are not measured in-repo are marked **UNCERTAIN** and carry the instrumentation that closes them.

**Headline:** the overlap design in `kernel-plan-3` is sound and should ship, but it is *not* the first thing to build. Three days of transport and draft-cache work are modeled at 2.0–2.5× cross-internet; overlap adds a further 1.3–1.7× on top and costs weeks. Do them in that order, and let the telemetry from step 1 decide whether overlap is still worth the weeks.

---

## 0. Ranked plan (expected gain per engineering week)

| # | Item | Effort | Modeled gain @100 ms RTT | @250 ms | Gain/week | Gate |
|---|---|---|---|---|---|---|
| **A1** | `f32ToF16` scratch hoist (+ a real correctness bug) | **hours** | −17 ms/lap; fixes silently halved wire values | ~5× / day | none — ship now | correctness |
| **A2** | Room prefill fills the MTP draft cache | **1–2 days** | +18–45 % tokens/lap | same | very high | `tests/test_mtp_split.js` |
| **A3** | Frame chunking to ≤ 4.6 KB sends (SCTP `max_burst`) | **1 day** (1 h A/B first) | −1 RTT per hop ⇒ −200 ms/lap | −500 ms/lap | very high | A/B before building |
| **A4** | Per-hop telemetry in the return frame | 2–3 days | 0 (unblocks everything) | 0 | high | roadmap 21 |
| **A5** | Prefill: ack instead of return frame + rounds in flight | 3–4 days | TTFT 2.4× | TTFT 2.4× | high | executor (B0) |
| **C** | Placement / partition + chain order + host election | 3–5 days | 0–2× (topology-dependent) | up to 2× multi-site | medium-high | needs A4 |
| **B** | Lap overlap (continuous speculation, K=3, L=2) | **3–5 weeks** | 1.29–1.52× | 1.37–1.66× | medium | needs A4 + B0 |
| **D** | Cheaper drafts (one-submit chain, FR-Spec vocab) | 3–4 weeks | raises the ceiling 18.3 → 23.7 tok/s | same | low-medium | after B |

Cumulative modeled decode for a two-laptop cross-internet room: **4.6 → 11.7–13.8 tok/s at 100 ms**, **2.6 → 9.1–11.0 tok/s at 250 ms** (§4).

---

## 1. Where the lap goes today

`ai.lapMs` (`room.js:974`) is a single EMA over the whole round trip. Decomposed from the code and the measured engine costs:

```
step = K·t_d  +  T_lap  +  t_h  +  a·t_r
T_lap = t_ht + Σ_hops[ pack + transport + unpack ] + Σ_peers t_pi
```

| symbol | what | value | source |
|---|---|---|---|
| `P4` | whole-model 4-column trunk pass | 133 ms GB10 / **~180 ms laptop-class** | `docs/deltanet-prefill-spec.md:8` (132.7 ms), scaled by 149/111 |
| `t_ht`,`t_pi` | per-device share of `P4` | `P4 × n_i/64` | — |
| `t_d` | one chained draft (MTP block + Q8 head + argmax + 8 B readback) | 22 ms GB10/Deno; **~18 ms browser** | measured; ~11 ms of the Deno figure is Deno's fixed 10 ms `mapAsync` poll, absent in Chrome — **UNCERTAIN** |
| `t_h` | `headBatch(4)` + 4 MB readback + top-k | 21 ms Deno, **~12 ms browser** — **UNCERTAIN** | `engine/qwen35.js:743-761` |
| `t_r` | refill (MTP block only, no head, no readback) | 3.6 ms measured; **5 ms browser** | `specStep` refill loop, `qwen35.js:861-864` |
| `pack` | `packF16` of a 4-col frame (20,480 floats) | **8 ms** → 0.05 ms after A1 | `room/wire.js:19`, `engine/gguf.js:16` |
| `transport` | one hop, 40 KB frame | **~1.5·RTT + 16 ms** (20 Mbit/s uplink) | SCTP `max_burst`=4, cwnd 10 MTU; PeerJS chunks at 16,300 B |
| `E` | tokens per settled lap | **2.8 without the MTP fill / 3.55 with** | bench-log 85 % @K=3; room path never fills (§2.2) |

Two-device room, 50/50 split, K=3:

```
T_lap(today) = 229 + 3·RTT ms      step = T_lap + 79 ms
RTT=100 → 529 / 608 ms → 4.6 tok/s (E=2.8)
RTT=250 → 979 / 1058 ms → 2.6 tok/s
```

Measured cross-internet is 3.5–4 tok/s (`docs/bench-log.md`), i.e. the model is within ~25 % and conservative. **Three-quarters of the 250 ms lap is transport, and none of it is bandwidth.**

---

## 2. Track A — transport and draft cache (days, ships first)

### 2.1 A1 · `f32ToF16` scratch hoist — hours, do this today

`engine/gguf.js:16-17` allocates `new Float32Array(1)` **and** `new Uint32Array` per element. `packWire` (`room/wire.js:37`) runs on every hop: `room.js:867, 932, 970` (host) and `1102, 1124` (workers).

* Measured: 5,120 floats = 2.0 ms, 40,960 floats = 32–56 ms. Hoisting the two scratch arrays to module scope: **160× faster** (37.7 → 0.23 ms), bit-identical.
* It is also a **correctness bug**: line 32 is `sign | (e << 10) | ((m + 0x1000) >> 13)`. When the rounded mantissa carries out to `0x400`, the `|` cannot carry into the exponent, so the value is **halved** (`f32ToF16(1.9999) → 1.0`). Rate ≈ 3.96 × 10⁻⁴ per element ⇒ **1–2 elements of every 5,120-dim hidden state are halved on every hop, today.** Fix with `+` instead of `|`, or switch to `new Uint16Array(new Float16Array(x).buffer)` (Chrome 135+, Safari 18.2+) and keep the corrected bit trick as fallback.
* **Scope the change to `room/wire.js` first.** `f32ToF16` is also used by `quantizeQ8` on the model-load path (`gguf.js:414`), so a global fix changes Q8 scales for K-quant tensors and will move the solo goldens. Fix globally *and* re-baseline, or fix locally now and file the loader fix separately.
* Add a unit test: 4 M random bit patterns + specials against `Float16Array`.

Saving: ~17 ms per lap per hop-pair, plus a latent accuracy bug removed.

### 2.2 A2 · Room prefill must fill the MTP draft cache — 1–2 days, largest cheap win

Solo prefill fills it (`qwen35.js:883-891`, `mtpRun(c, ids[i+c+1], …)` per column). The **room path never does**: `aiGenerate`'s split branch (`room.js:925-938`) sends `ai-hidden-b`, does `await returned;` and **discards the returned hiddens**; no `mtpRun` appears anywhere in `room.js`. So in a room the `nextn` block — a full-attention layer with its own KV cache (`gguf.js:479-483`, `forceFull`) — drafts from an empty (or, on the second question, *stale*: `reset()` at `qwen35.js:23-31` clears only DeltaNet state) cache for every prompt position.

Measured effect (GB10, K=7, greedy): accepted drafts/step **4.44 → 1.89** on a grounded rewrite prompt, 3.93 → 3.60 on code, 1.62 → 1.62 on prose. Trace replay: grounded 9.3 → 6.4 tok/s at 150 ms RTT, 7.4 → 4.6 at 300 ms.

Fix, inside the existing prefill loop:
```js
// room.js ~933, replacing `await returned;`
const hb = await returned;                       // stop discarding it
for (let c = 0; c < nChunks * NC; c++) {
  const p = basePos + c;
  if (p + 1 >= ids.length) break;
  ai.engine.setHidden(hb.subarray(c * hdim, (c + 1) * hdim));
  await ai.engine.mtpRun(null, ids[p + 1], p + 1, false);   // no head, no readback
}
```
and the same for the tail `aiPipeToken` loop (`room.js:939`), where `ai.lastHidden` is already kept.

* Cost: ~8 ms of host GPU per prompt token (measured 5,754 → 6,538 ms for 96 tokens, +13.6 % prefill wall time). `mtpRun(..., false)` submits and returns without a readback, so on the cross-network path it overlaps the next round's wire wait for free — and after A5 it is fully hidden.
* Output is unchanged (drafts only affect speed); the greedy goldens gate it. Note `tests/test_mtp_split.js` sets `host.mtpFill = true`, which is inert outside `prefillTokens` — extend that test with explicit fills or it will keep passing without covering this.
* This is the input that makes the throughput model's `E = 3.55` legitimate for rooms.

### 2.3 A3 · Frame chunking — 1 hour to test, 1 day to build

Chrome/Safari SCTP (`net/dcsctp`, verified in source): `mtu = 1191`, usable DATA payload **1,160 B**, `cwnd_mtus_initial = 10`, **`max_burst = 4`**, receiver SACKs every 2nd packet. `TransmissionControlBlock::SendBufferedPackets` is a bare `for (i < max_burst)` loop invoked once per `RTCDataChannel` message and once per SACK; nothing else reschedules the remainder. cwnd never decays while idle.

* A 4-column frame is 40,960 B = **36 packets**; PeerJS 1.5.4 already splits it into 3 chunks of ≤ 16,300 B (`chunkedMTU`; the `chunkedBrowsers` gate is dead code, so this happens on Safari too). Delivery: 10 (cwnd-capped) → 15 → 22 ⇒ **~1.5–2.5 RTT per hop**.
* An 8-column verify frame is 81,920 B = **71 packets** ⇒ **~3.5 RTT per hop**.
* Fix: pre-slice the payload into ≤ **4,632 B** parts (4 × 1,160 minus an 8-byte header) and call `conn.send()` once per part. Each send opens its own 4-packet burst; once cwnd exceeds the frame (2–3 laps), a hop is ~0.5 RTT + serialization. **Do not** use `serialization:'none'` (loses BinaryPack for every control message) and **do not** open a second `DataConnection` (same SCTP association, same cwnd and burst budget).

Wire change (`room/wire.js`), generic so laps, prefill and returns all use it:
```js
export const PART_BYTES = 4632;
export function sendFramed(conn, hdr, payload /* Uint16Array */) {
  const u8 = new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength);
  const m = Math.ceil(u8.byteLength / PART_BYTES) || 1;
  for (let i = 0; i < m; i++)
    conn.send({ ...hdr, enc: "f16", part: i, parts: m,
                data: u8.slice(i * PART_BYTES, (i + 1) * PART_BYTES) });
}
// receiver: key by hdr.t + (hdr.lap ?? hdr.basePos); channel is ordered+reliable,
// so append in arrival order and complete at parts === m. Drop the whole group if
// hdr.epoch < px.epoch (a cancelled lap's tail is discarded chunk by chunk).
```
Every part carries `lap`/`epoch`, so §3.4's cancel semantics work at chunk granularity.

**Test before building (1 hour):** on a real cross-network room, ping-pong 10/40/80/160 KB (a) as one `conn.send()`, (b) split into ≤ 4,632 B sends; log `dataChannel.bufferedAmount`. Prediction: (a) ≈ 1.5–3.5 RTT, (b) ≈ 0.5 RTT + serialization after 2–4 warm-up laps. If (b) does not win, stop — the lap is not SCTP-bound and the remaining transport cost is uplink serialization, which only A-class byte reduction touches.

### 2.4 A4 · Per-hop telemetry — 2–3 days

No per-hop data exists; roadmap 09 is explicitly gated on it. Add to every forwarded/returned frame a `hops: [{peer, recv→encode, compute, sendAt, rttNext}]` array of **durations only** (`performance.now()` deltas; clocks are unsynchronized). Host computes `transport = T_lap − Σ compute − host pack/compute` and cross-checks against `conns.get(next).rtt` — note each peer already pings every other peer every 2.5 s (`room.js:312, 254-257`), so the peer→peer legs are already measured locally and only need reporting. Feed `perHopMs.p50/p90` into roadmap 21's report schema.

### 2.5 A5 · Prefill: ack, not a return frame; rounds in flight — 3–4 days

* `room.js:1102-1104` packs and returns the full 16×10 KB hidden even when `!d.spec`, and the host throws it away. Replace with `{t:"ai-prefill-ack", basePos, n}` when `!d.spec && ai.next === "host"` — saves 160 KB and ~30 ms of pack on the last hop per round. (Needs its own handler: `ai-hiddenret-b` calls `unpackWire`, which throws on a payload-less frame.) **After A2 the host does need the hiddens** — so keep the return frame on the *host's* last hop only if the host cannot recompute them; it can't, so instead: return the ack, and have the host fill the MTP cache from **its own** `embedRunBatch` output? No — the MTP fill needs the *final trunk* hidden. Keep the return frame while A2 is on, and make the ack an option for `mtpFill === false` rooms.
* Rounds in flight: positions are strictly increasing and waiters are already keyed `"b"+basePos`, so a sliding window of 2–3 rounds is safe **once the peer executor (§3.1) exists**. Also loosen the batching loop from `>= NC` to `>= 1` (`embedRunBatch`/`runHiddenBatch` accept `n < NC`) so the ≤ 7-token tail stops running as serial single-token laps.
* Modeled: 100-token prompt, 2-way split, 250 ms RTT: ~6.1 s → ~2.5 s TTFT (2.4×). Speedup is `R·N/(R+N−1)` with `R = ⌈P/16⌉` rounds — 1.8× at N=2/128 tokens, 2.9× at N=4, 3.7× at N=6.

---

## 3. Track B — lap overlap / speculative pipelining

This refines `docs/kernel-plan-3.md` §"Continuous Speculation". **Corrections to that spec are marked ⚠.** All line references below are to the current tree (the spec still cites `p2p.html`; the room logic moved to `room.js`).

### 3.0 Invariant

> A lap is atomic. Its tokens are emitted only at settle. Peers apply state changes in lap order, and every state change is undoable by a byte copy from a snapshot slot. Under greedy sampling the emitted sequence must be bit-identical to today's sequential `specStep`.

### 3.1 B0 · Prerequisites (zero behavior change, ship with Track A)

Three latent hazards go live the moment a second frame is in flight. All three should land in `room/pipeline.js` (roadmap 22's seam), not in the `room.js` switch.

1. **Per-peer serialized executor.** `room.js:227` dispatches `aiOnData(from, d)` *without* `await`; the `ai-hidden-b` handler (`room.js:1089-1105`) awaits `runHiddenBatch` per chunk, and `qwen35.js:507/704` share a single `stageXB` MAP_READ buffer. Two in-flight frames ⇒ DeltaNet state applied out of column order **and** a second `stageXB.mapAsync` rejecting with `OperationError` (unhandled → the host hits its 90 s timeout). Fix: one promise chain per peer covering `ai-hidden`, `ai-hidden-b`, `ai-lap` and `ai-rollback`; `ai-reset` resets the chain.
2. **Waiters keyed by lap id.** `room.js:967` keys the verify waiter `"b"+pos`. After a full-accept/bonus-miss the corrected lap has the *same* `basePos`, so `Map.set` overwrites the stale resolver and a stale return resolves the corrected lap's promise with hiddens for the wrong tokens — `badF32` will not catch it. Key by `"lap"+id`; carry `lap` in the return.
3. **Rollback sequenced, not applied on receipt.** `room.js:1107-1110` calls `restoreDN(d.k)` immediately, submitting a copy at an arbitrary point between chunk submits. Enqueue it into the same executor.

### 3.2 Messages (protocol v2)

Add to `docs/protocol.md`. `ai-hidden`/`ai-hidden-b` stay for prefill and for peers that do not advertise `caps.lap`.

| Message | Direction | Payload |
|---|---|---|
| `ai-lap` | host→chain[0], peer→peer | `{v:2, lap, epoch, parent, basePos, n, slotBase, restore?, part, parts, data, enc}` |
| `ai-lapret` | last peer→host | `{v:2, lap, epoch, basePos, n, hops[], part, parts, data, enc}` |
| `ai-rollback` | host→all (broadcast) | `{v:2, afterLap, slot, epoch}` |
| `ai-ready` | worker→host | **+** `caps:{ lap:1, chunk:1, proto:2 }` |

⚠ **`slot`, not `k`.** The spec has peers recompute `slotOf(afterLap, k)` locally, which requires host and peers to agree on `K`. Send the absolute slot index the host already computed; peers stay ignorant of `K`. `restoreDN(slot)` already takes an absolute index (`qwen35.js:820-826`).

`restore` piggybacks the rollback on the corrected lap so a peer that has not yet seen the broadcast still restores in order. Idempotent by `epoch`.

Capability negotiation: continuous mode runs only if **every** peer advertises `caps.lap`; otherwise fall back to today's `specStep` + K-ladder. Bump the protocol version so older peers fail loudly at `ai-load` (roadmap 16 must supply the version constant — it does not exist yet; `ai-load` carries no version field today).

### 3.3 Rollback slots

State per DeltaNet layer per slot: `S = nVH·dState²·4 = 48·128·128·4 = 3.0 MiB` + `conv = convDim·3·4 = 120 KiB` (`qwen35.js:241-242`). 48 DN layers ⇒ **150 MiB/slot** for the full model, ~37 MiB for a 12-DN-layer peer.

| mode | slots | full model | 16-layer peer (12 DN) |
|---|---|---|---|
| today (`qwen35.js:509-511`, hard-coded 7) | 7 | 1.02 GiB | 262 MiB |
| **continuous K=3, L=2** | **8** | 1.17 GiB | 299 MiB |
| K=3, L=3 | 12 | 1.75 GiB | — (not worth it) |
| K=7, L=2 | 16 | 2.34 GiB | — (rejected) |

Slot map: **`slotOf(lap, k) = (lap & 1) · (K + 1) + k`**, `k ∈ 0..K`.

⚠ **The spec's `slotOf` uses `engine.NC`, and its constructor asserts `K + 1 === engine.NC`. Both are wrong for the room:** `room.js:664` builds the engine with `batchCols: 8`, so `NC = 8` while `K + 1 = 4`. Two consequences:
* The assertion must be **`K + 1 <= NC`**, and `slotOf` must use `K + 1`, not `NC` (otherwise slot bases are 0 and 8, needing 16 slots and colliding).
* A 4-column lap on an `NC = 8` engine is fine and *cheaper*: `_dop` (`qwen35.js:352-355`) automatically selects the `_b4` twin kernels when `nCols <= 4`. No kernel work needed.

`snapSlots` becomes a constructor argument defaulting to 7; room engines pass 8. **Every** column of an in-flight lap is snapshotted (today the final column is skipped because `restoreDN(K)` was never needed; "fully accepted but bonus missed" now restores it). No WGSL change: passing `{ base: slotBase, total: slotBase + n + 1 }` makes the existing kernels write slots `slotBase..slotBase+n-1` — `frame.snap` packs `base+1` in 8 bits, so `slotBase ≤ 254`.

Full-attention KV needs no rollback: it is position-indexed and written at `(basePos+c)·kvDim·4`, and never read beyond the query position (`docs/kernels.md` trick 18).

Restore cases at `settle(N)` with child `N+1` in flight:

| verdict | restore | child |
|---|---|---|
| `a < K` | `slot(N, a)` | cancelled |
| `a == K`, `out[K] != child.tokens[0]` | `slot(N, K)` | cancelled |
| `a == K`, `out[K] == child.tokens[0]` | none | becomes head of the ring |

### 3.4 Cancel semantics

Ordered, reliable channels deliver a stale payload before the rollback that invalidates it, and `send()` cannot be recalled. Cancellation is therefore **versioned, receiver-side, and best-effort**:

```js
const px = { q: Promise.resolve(), epoch: 0, restoredEpoch: 0, lastDone: -1, pending: null };
function pxApplyRestore(r) {                 // r = {afterLap, slot, epoch}
  if (!r || r.epoch <= px.restoredEpoch) return;
  px.epoch = Math.max(px.epoch, r.epoch);    // SYNCHRONOUS: a running lap probes this
  pxEnqueue(async () => {
    if (r.epoch <= px.restoredEpoch) return;
    if (px.lastDone < r.afterLap) { px.pending = r; return; }
    ai.engine.restoreDN(r.slot);
    px.restoredEpoch = r.epoch;
  });
}
// ai-lap:      pxApplyRestore(d.restore); if (d.epoch < px.epoch) break; pxEnqueue(() => pxRunLap(d));
// ai-rollback: pxApplyRestore(d);
```

Mid-lap abort: `_runBatchAndRead` gains `(shouldAbort, groups=4)` and submits the layer range in `groups` chunks, awaiting `onSubmittedWorkDone()` between them so the event loop can deliver a queued rollback. Waste ≤ ¼ of one device's lap; downstream peers drop the lap outright. `groups = 1` disables probing. ⚠ The per-probe idle-gap cost is **UNCERTAIN** — measure with `groups ∈ {1,2,4}` on Chrome and Safari before defaulting to 4.

| guarantee | mechanism |
|---|---|
| a stale lap never corrupts state after the restore | restore queued behind the running lap; slot copy is total for all DN layers |
| a stale return never resolves the wrong promise | waiters keyed by lap id; deleted at discard |
| a rollback applies exactly once per epoch, any arrival order | `restoredEpoch` + `pxApplyRestore` on both paths |
| the canonical column is never cancelled mid-lap | a lap is dropped before start, aborted at a group boundary (then restored), or completes |
| chunked frames | every part carries `lap`/`epoch`; a stale group's tail is discarded chunk by chunk |

⚠ **The spec claims "the canonical column is never rejected (PipeInfer invariant)". That does not hold for gambled children:** a child's column 0 is `d_{K+1}`, a *draft* standing in for the parent's bonus token, and the whole child is cancelled when the gamble loses. SwarmLLM has no uncancellable canonical run; forward progress on a lost gamble is the re-issued sequential lap. This is exactly the `(1−q)(T_seq + δ)` term in §4.

### 3.5 Host state machine

Two phases per lap: `issue(lap)` (draft → host layer share → send) and `settle(lap)` (wait → head → sample → verdict → rollback/refill). Ring holds at most `L = 2`. The host never blocks on a return while it has issue work.

```
IDLE  ── issueFresh(next) ──▶ ONE_IN_FLIGHT [N]
                                    │
      gate(N): draft d_{K+1} from xChain; pp = Π p(N.drafts) · p(d_{K+1})
        pp ≥ θ ──▶ issueChild(N) ──▶ TWO_IN_FLIGHT [N, N+1]
        pp < θ ──▶ settle(N)

TWO_IN_FLIGHT · settle(N):
   a==K && out[K]==child.tokens[0]  → emit K+1, refill 1..K+1, ring=[N+1]  → ONE_IN_FLIGHT
   else                              → restore slot(N,a); epoch++; broadcast rollback;
                                       discard child; emit a+1; refill 1..a;
                                       setHidden(hs[a]); next=out[a]        → IDLE
ONE_IN_FLIGHT · settle(N) with no child:
   a<K  → restore slot(N,a), rollback, next=out[a]  → IDLE
   a==K → next=out[K]                               → IDLE
```

Engine additions (`engine/qwen35.js`), all small:

* `snapSlots` constructor arg; `_initBatch` allocates `snapSlots × L.S.size` / `× L.convState.size`.
* `slotOf(lap, k)`, `saveChain()`, `loadChain()` and an `xChain` buffer (`dim·4`). ✅ **Resolved UNCERTAIN:** `this.x` is created with `STORAGE | COPY_DST | COPY_SRC` (`qwen35.js:143-144`), so `saveChain` works as written. The chain buffer is needed because the refill loop calls `mtpRun`, which clobbers `this.x`, while a child lap must continue from the MTP output after `d_K`.
* `_runBatchAndRead(basePos, n, shouldAbort, groups)` and a matching `runHiddenBatch` signature.
* `argmax_lse` (argmax + log-sum-exp in one workgroup) so `mtpRun(..., "argmaxp")` returns `{id, p}`. ✅ **Resolved UNCERTAIN:** `_bg` uses `pipe.getBindGroupLayout(group)` (`qwen35.js:333-338`), i.e. per-pipeline auto layouts — `argmax_lse` needs its own bind group. Core WGSL only; barriers stay outside conditionals.

Host driver goes in a new `engine/specpipe.js` (`SpecPipeline`), transport glue (`lapIO.sendLap/discard/rollback`) in `room/pipeline.js`. `specStep` and every kernel other than the additive `argmax_lse` are untouched.

### 3.6 Gamble gate

Gamble iff `Π_{k=1..K} p_k(N) · p(d_{K+1}) ≥ θ`, with `p_k` = the MTP head's probability of its own argmax. Two-stage: stage A on `Π p_k(N)` alone (free); stage B after drafting `d_{K+1}` (costs one draft). θ starts at θ₀ = 0.35, +0.10 per consecutive gamble, reset to θ₀ on an accepted completed lap, −0.05 when a gamble is skipped, clamped [0.15, 0.90]. **All four constants are UNCERTAIN placeholders.** Worse, the measured draft-confidence distribution is heavily top-loaded (128 of 190 traced positions sit at `p ≥ 0.9`; only 7 of 67 code positions and 3 of 47 grounded positions below 0.5), so θ₀ is being tuned in the region with almost no data. **Do not set θ₀ from the current traces** — log `calib` pairs `[p_draft, accepted]` in the first real rooms and fit `q̂(pp)` there, then switch to the expected-gain criterion `q̂(pp)·(T_seq − Δ_s) > (1 − q̂(pp))·δ`.

Sampling note: rooms sample at temp 0.8 / top-k 40 (`room/sampling.js`), so the bonus match is a *sampled* event and per-draft acceptance is below the greedy bench figures. The greedy-equivalence test (§6) is argmax-only; add a seeded-sampler variant before trusting equivalence in a real room.

---

## 4. Expected tok/s at 100 and 250 ms RTT

Canonical room: **2 devices** (host + 1 peer), laptop-class, 50/50 layer split, K = 3, 4-column laps, 20 Mbit/s uplinks.

**Renewal model.** Tokens per settled lap are the same in both branches, so the speedup is a pure time ratio:

```
S = T_seq / [ q·Δ_s + (1 − q)·(T_seq + δ) ]
Δ_s = max( T_host , max_i peer_stage_i , T_lap / L )
T_host = (K+1)·t_d + t_ht + t_h + (K+1)·t_r
```

⚠ The spec's `Δ_s = max(T_host, s_max)` omits the **latency bound `T_lap/L`**: with `L = 2` at most two laps are in flight, so the host cannot settle faster than `T_lap/2` however cheap its own stage is. That term dominates at 250 ms.

Inputs (after Track A): `t_ht = t_pi = 90`, `t_d = 18`, `t_h = 12`, `t_r = 5`, hop = `0.5·RTT + 16`, `δ ≈ 30` (4-group probing), `E = 3.55`.

```
T_host = 4·18 + 90 + 12 + 4·5 = 194 ms      peer stage ≈ 106 ms
T_lap  = 180 + RTT + 32                     T_seq = T_lap + 79
```

| RTT | Δ_s | T_seq | q=0.52 (geometric p=0.85) | q=0.72 (0.85 = mean prefix) |
|---|---|---|---|---|
| 100 ms | max(194,106,156)=**194** | 391 ms | S=**1.29** → 11.7 tok/s | S=**1.52** → 13.8 tok/s |
| 250 ms | max(194,106,231)=**231** | 541 ms | S=**1.37** → 9.1 tok/s | S=**1.66** → 11.0 tok/s |

Cumulative, two-device room:

| stage | 100 ms | 250 ms |
|---|---|---|
| today (modeled; measured anchor 3.5–4) | 4.6 | 2.6 |
| + A1/A3 transport | 7.2 (1.6×) | 5.2 (2.0×) |
| + A2 MTP fill | 9.1 | 6.6 |
| + B overlap | **11.7 – 13.8** | **9.1 – 11.0** |
| ceiling (`E/T_host`) | 18.3 | 18.3 |
| ceiling after D (cheaper drafts, `T_host = 150`) | 23.7 | 23.7 |

**Three devices** (33/33/33, 3 hops), same inputs: after A, 7.8 / 5.2 tok/s; with overlap 10.7–13.1 @100 ms and 7.1–8.5 @250 ms.

**Versus the K-ladder.** Sequential K=7 (8-column lap, 80 KB frames, `P8 ≈ 1.6·P4`) models at 9.6 tok/s @100 and 7.7 @250 — i.e. still competitive after Track A, and beaten by continuous K=3 only at the higher RTT. Keep `pickK` (`room.js:993-1009`) and let it choose among `{continuous K=3, sequential K=5, sequential K=7}` by measured tok/s.

**Reading of these numbers.** Track A is worth 1.6–2.0× for ~3 days. Overlap is worth 1.29–1.66× for 3–5 weeks. Overlap's value *grows* with RTT and *shrinks* as transport is fixed — which is why A3's one-hour A/B must run first: if chunking collapses the lap to `RTT + compute`, the modeled gain above is the honest one; if it does not, laps stay at 600–1000 ms and overlap is worth 1.5–1.8× instead.

---

## 5. Same-network placement and partitioning (2–6 devices)

### 5.1 What is wrong today

* `room.js:755-769` splits layers **proportional to pledged GB**, with `layerBytes` from the GGUF index only — no KV, no DeltaNet state, no shadow slots, no per-device speed.
* `room.js:764-765` forces every peer to hold **≥ 1 layer**, so a phone that pledges 0.5 GB adds a full hop to save ~1.7 ms of a fast device's compute.
* `room.js:724` orders the chain by **`[...conns.keys()].sort()`** — PeerJS ids are random for joiners, so a two-site room can cross the WAN 4 times instead of 2.
* Nothing measures per-device throughput. `autotuneCoop` (`engine/autotune.js`) times a GEMV but reports only `{wg, rows}` and runs **after** the host has already computed and sent `ai-load`.

Single-stream decode is a **sum** of stages plus hops, not a max, so evenness buys nothing: every byte moved onto a slower device costs `bytes·(1/B_slow − 1/B_fast)`, and every extra device costs a hop. exo's own table (1/2/3 × M4 Pro: 49.3 / 44.4 / 39.7 tok/s single-request) is the clean public confirmation.

### 5.2 Measured inputs (add at join)

Each device reports, in `hello`/`pledge`:

```js
{ msPerLayer: <ms for one DN layer in a 4-col pass>,   // ~200 ms probe, real coop GEMV, not a raw read
  attnSurcharge: <extra ms for a full-attention layer at maxSeq>,
  budgetBytes,                                          // usable GPU bytes
  hostCapable: <budgetBytes ≥ extras + one layer>,
  rtt: { peerId: ms, ... } }                            // already measured locally, room.js:254-257,312
```
Extract the probe from `autotuneCoop` and run it at **join**, not at load. Use *effective* ms/layer, not spec bandwidth: the Mac's ~100 GB/s effective rate is what predicts stage time, and its cause is unresolved.

### 5.3 Memory model (per device, exact)

```
bytes(range)                                   // qwen35ShardBytes over the real GGUF index
+ nDN  × (S + conv) × (1 + snapSlots)          // 3.0 MiB + 120 KiB, ×9 at snapSlots=8 → 28.1 MiB/DN layer
+ nAttn × 2 × maxSeq × kvDim × 4               // 4 MiB at maxSeq 512; 32 MiB at 4096
+ (host ? headQ8 1.35 GB + MTP block 265 MB + MTP KV : 0)
+ ~60 MB batch scratch (B buffers + stageLogitsN = NC × vocab × 4 = 7.9 MB at NC=8)
```
Attention layers are `i % 4 === 3` (`gguf.js:427`), so a contiguous range of `n` layers holds `⌈…⌉` of them — count them exactly from the range, not from an average. The embedding table (715 MB Q4) stays in host **RAM**, not GPU (`gguf.js:184`), unless item D moves it.

### 5.4 Algorithm

Objective (decode-dominant; prefill blends in once A5 lands):

```
cost(order, counts) = Σ_i cost_i(range_i)                       // sum, not max — one token in flight
                    + Σ_hops ( 0.5·RTT(hop) + frameBytes/bw )
                    + hostSerial                                 // (K+1)t_d + t_h + (K+1)t_r
cost_blend = w_dec · cost_decode + w_pre · cost_prefill,  w_dec = outTokens/(outTokens + promptTokens)
   // decode is a SUM of stages; pipelined prefill (A5) is a MAX of stages — the two pull opposite ways.
   // At 400 output tokens vs an 80-token prompt, w_dec ≈ 0.83: decode wins, fill fastest first.
```

Decomposed search — brute-forceable for n ≤ 6, ~50 ms of JS:

1. **Host election.** `hostCapable && min msPerLayer`. The mechanism already exists: `aiStartAnywhere` (`room.js:698-711`) elects a "boss" by `contribGB` and hands it the host role via `ai-start-req`. Replace `biggestPeerId()` with a score = `hostCapable ? 1/msPerLayer : 0`. The head (`t_h`) and the whole draft chain (`(K+1)·t_d`) are host-serial, so this is the single highest-leverage placement decision.
2. **Solo check.** If the elected host alone can hold the model + extras, **run solo** — zero hops, and its own pass is the whole model rather than a share. Splitting only wins when the split shares reduce per-device stage time by more than the hops cost. Offer "fastest" vs "fair" as a user toggle; default to fair for the product story, but say what it costs.
3. **Subset choice.** For each subset `S ∋ host` (≤ 32 for n = 6), run a DP over cut points: `DP[l][i] = min cost of assigning the first l layers to the first i devices`, `O(L²·|S|) = 64·64·6 ≈ 25k` ops, with the memory model above as a hard feasibility constraint per device. Keep the minimum-cost feasible subset. This subsumes the "drop weak devices" rule (prima.cpp's Halda removes a device whose ILP share rounds to zero); make it explicit: **delete the `assigned[i] === 0 → steal a layer` guard at `room.js:764-765`, and prune zero-layer peers from `ai.chain` (rewiring `next`)** — leaving a zero-layer peer in the chain pays both its hops for no work, strictly worse than today.
4. **Chain order.** With counts fixed, minimize `Σ RTT(chain[i], chain[i+1])` over permutations: `(|S|−1)! ≤ 120` for 6 devices, brute force. Needs the peer→peer RTT matrix from step 5.2. Zero gain at ≤ 3 devices (a symmetric matrix makes both tours equal) — gate the work on a ≥ 4-device room that actually spans sites.
5. **Fallbacks.** If no subset fits, use all devices (today's behavior) and log the shortfall — `room.js:769-772` already computes `needGB > haveGB·1.15`. Pruned peers keep their weights and stay available as prefill/memory helpers and as roadmap-03 spares.

### 5.5 What this is worth

* **Homogeneous LAN room (classroom):** ~0. Already balanced.
* **Heterogeneous room where one device fits the model:** solo beats splitting. Two-device GB10+Mac: ~16 vs ~11.5–14 tok/s ⇒ **1.15–1.4×**.
* **Mixed room that must split, with headroom on the fast device:** ~10 %, bounded by `slack_fast × (1/τ_slow − 1/τ_fast)` — **zero** when total pledge ≈ model size.
* **Multi-site (≥ 4 devices, 2 sites):** chain ordering alone removes up to half the WAN crossings — at 60 ms WAN RTT on a 300 ms lap, ~1.25×.
* **LAN generally:** hops are 5–15 ms, so byte-level and RTT-level work is worth little; the LAN lever is **aggregate** throughput (several askers pipelined through idle stages, roadmap 20), not single-stream latency. Cost of that: ~300–420 MB per in-flight request per 16-layer device (36 MB of DeltaNet `S` + 288 MB of shadow slots + KV), plus a request id on every compute frame. Post-deadline.

**Do not** pursue tensor parallelism on any network class: 2 collectives × 64 layers = 128 syncs/token at ≥ 3 ms each = 425–2600 ms/token versus ~125 ms pipelined. Even a hypothetical 1 ms sync costs 128 ms to save the 41 ms of GEMV it parallelizes. Layer-split is the right shape; that is a documented decision, not a task.

---

## 6. Test plan and gates

1. **Greedy equivalence (blocks everything in B).** Extend `tests/test_mtp_split.js`: run one prompt through (a) `specStep`, (b) `SpecPipeline` with a loopback `io` whose `sendLap` runs `runHiddenBatch` on a second engine holding the remaining layers, plus an injected delay, and whose `rollback` calls `restoreDN(slot)`. Bit-identical under argmax. **Vary the delay so returns land both before and after `issueChild`.** Add a seeded-sampler variant (temp 0.8 / top-k 40) — the shipped rooms sample.
2. **Slot map.** After a lost gamble, compare a peer's live `S` against a CPU copy of slot `(N, a)`. Assert `snapSlots ≥ L·(K+1)` and `K + 1 <= NC` at construction.
3. **Ordering fuzz (3 tabs, LAN).** Randomize delivery delay of `ai-rollback` vs `ai-lap`; assert `restoredEpoch` monotone and output identical to sequential.
4. **Chunking A/B (A3).** Size sweep, one `send()` vs ≤ 4,632 B parts, on a real cross-network room. Log `bufferedAmount`.
5. **Wire pack.** `f32ToF16` vs `Float16Array` over 4 M random bit patterns + specials; assert no factor-2 mismatches (the bug in A1 currently produces ~4×10⁻⁴).
6. **MTP fill (A2).** Acceptance rate room vs solo on the same prompt must converge to within noise.
7. **Safari 26 peer.** No new WGSL runs on peers; verify the executor, chunked reassembly and `onSubmittedWorkDone` probing.
8. **Cancellation waste.** Log the abort group index; assert ≤ 1 group per cancelled lap per device.

---

## 7. UNCERTAIN register

| item | why it matters | how to close |
|---|---|---|
| `t_d`, `t_h`, `t_r`, `t_ht` **in a browser** | every number in §4 | `performance.now()` around `mtpRun` / `headBatch` / `embedRunBatch` in a real room |
| Real per-hop transport split | decides whether B is worth weeks | A4 telemetry + the A3 A/B |
| `q` (bonus-match rate) under temp 0.8 / top-k 40 | the whole renewal model | `stats.fullAccept/laps`, `bonusMatch/gambled` in the first continuous room |
| Gate constants θ₀ / recovery / decay | gamble economics | `calib` pairs; do **not** set from the current traces |
| `onSubmittedWorkDone` probe cost per layer group | cancel granularity | sweep `groups ∈ {1,2,4}` on Chrome + Safari |
| Same-Wi-Fi lap breakdown | Track C's whole case | never measured separately; A4 gives it |
| Mac roofline gap (149 ms/token, cause unknown) | sets `P4` for every laptop room | separate investigation; not on this plan's critical path |
| Uplink bandwidth in real rooms | serialization term | `room.js:296-308` already has a bandwidth probe; log it |
| 8-column lap time in a room | the K-ladder comparison | measure alongside the K=3 continuous run |

---

## 8. File-by-file change map

| file | change | track |
|---|---|---|
| `engine/gguf.js:16-32` | hoist scratch arrays; `+` not `|` in the mantissa carry | A1 |
| `room/wire.js` | `sendFramed`/reassembly at ≤ 4,632 B; `Float16Array` fast path | A1, A3 |
| `room.js:925-938` | keep the returned hiddens; `setHidden` + `mtpRun` per prompt position; loosen `>= NC` to `>= 1` | A2, A5 |
| `room.js:227, 1089-1110` | move `ai-*` compute frames onto a per-peer promise chain | B0 |
| `room.js:967, 1112-1115` | waiters keyed by lap id; `ai-lapret` handler | B0 |
| `room.js:698-711, 724, 755-772` | host election by speed; chain order by RTT; DP placement; drop the ≥1-layer guard | C |
| `room/pipeline.js` (new) | executor, `lapIO`, chunk reassembly | B0, A3 |
| `engine/qwen35.js:143, 352-355, 509-511, 690-707, 820-826` | `snapSlots`, `slotOf`, `xChain`, abortable `_runBatchAndRead`, `argmax_lse` | B |
| `engine/specpipe.js` (new) | `SpecPipeline` host driver | B |
| `engine/autotune.js` | extract a ms/layer probe callable at join | C |
| `docs/protocol.md` | `ai-lap` / `ai-lapret` / `ai-rollback` v2, `caps`, chunk envelope, ordering guarantees | B |

Sequential `specStep`, solo prefill, and every kernel other than the additive `argmax_lse` are unchanged. Output stays bit-identical to plain decoding under any sampler, because the trunk always decides and the rollback is a byte copy.