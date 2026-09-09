# Room protocol

Browsers in a room form a WebRTC mesh (PeerJS signaling for the introduction only). One browser is the **host**: it owns the conversation, the tokenizer, the embedding table, the LM head and the sampler. The others are **workers** holding contiguous layer ranges; together they form a **chain** in layer order, with the last worker sending back to the host.

## Lifecycle

| Message | Direction | Meaning |
|---|---|---|
| `ai-start-req {model, boss, by}` | any → all | someone pressed start; `boss` (the biggest pledge) runs the first plan |
| `ai-load {v, model, range, next, host}` | host → worker | plan `v`: download and load layers `[range[0], range[1])`, forward to `next`. `v` is the plan version, monotonic per room; a worker ignores any `v` not above the one it holds. Same model and range as the engine already holds ⇒ rewire only (no reload) |
| `ai-wait {v}` | host → worker | no layers in plan `v`: free the GPU, keep the bytes in the Cache API (a warm spare) |
| `ai-progress {pct}` / `ai-hostprog` | worker ↔ host | download progress for the room UI |
| `ai-ready {v, ms?}` | worker → host | layers for plan `v` loaded and the link to `next` is up; `ms` = autotune time of one 5120×17408 q4 matvec on this GPU, feeds the speed weight |
| `ai-error {v?, message}` | worker → host | load failed (`v` set: the host excludes the device from the plan right after `v`, and tries it again in later plans; the re-deal button clears the exclusion) or a GPU/NaN error while serving. A worker whose load for `v` was superseded by `v+1` sends nothing |
| `ai-ready-all` | host → all | every device reported ready for the current plan; re-sent after every completed plan, so a late joiner gets it too. A late joiner without WebGPU (no plan is dealt for it) gets it straight from its `hello` while the room is online |
| `ai-layers {v, model, by, state, note}` | host → all | room state broadcast: `state` ∈ loading · online · generating · redealing · waiting, `by` = `{name: "lo–hi"}`, `note` = the status line to show (may be empty). Also sent to a device that joins a running room |
| `ai-reset` | host → all | new conversation: caches and states back to position 0 |
| `ai-genstart` / `ai-token` / `ai-gendone {stats}` | host → all | mirror the question and streamed answer to every screen. `stats` is the tok/s line, `failed: …`, or `stopped: <name> left (layers a–b)` when a chain member left mid-answer |
| `ai-ask` / `ai-busy` | guest → host | anyone in the room can ask; one generation at a time, and only while the room is `online` |

`ai-next` (re-seating a reloaded device into its old slot by name) is gone: a reload is a leave followed by a join, handled by the planner like any other. `hello {name, meta, tab, died?}` carries a per-tab id (`tab`, from `sessionStorage`: a reload keeps it, a second device never shares it); the host treats a new peer as the reload of an old one only when the tab ids match, so two live devices with the same display name are two devices. A `hello` without `tab` (older peer) matches by name only when the old link is closed or silent for 5 s.

## Compute frames

| Message | Payload | Use |
|---|---|---|
| `ai-hidden {pos}` → … → `ai-hiddenret` | one hidden state | single-token decode lap |
| `ai-hidden-b {basePos, n, spec?}` → … → `ai-hiddenret-b` | `n` hidden states (multiple of the batch width; up to 16) | batched prefill (`spec` absent) or speculative verify (`spec: 1`: the recurrent state is snapshotted after every non-final column) |
| `ai-rollback {k}` | — | host rejected drafts after column `k`; workers restore recurrent state to the snapshot after column `k` |

Hidden states travel as binary frames: an f16-packed `Uint16Array` (10 KB for `dim = 5120`) with the wire format flag `WIRE_F16`; decoders accept f32 for older peers. Frames are correlated by position (`pos` / `basePos`), and the host keeps a timeout per outstanding lap. Header bytes 18–19 carry the plan version `v`: a worker drops a frame whose `v` differs from the plan it holds, and the host drops a return whose `v` is not the answer in flight. `v = 0` (a peer older than this protocol) is accepted on both sides.

## Ordering guarantees

- Data channels are ordered and reliable; a worker processes frames in arrival order, so recurrent states advance deterministically.
- Inside a batched frame, columns are processed strictly in order; snapshot slots are indexed by global column (`frame.snap` packs base and total), so an 8-column verify split into two 4-column chunks on an older worker still rolls back correctly.
- Control messages (`ai-load`, `ai-reset`, `ai-rollback`) and compute frames use different SCTP streams, so nothing is assumed about their relative order: a plan is only applied between answers, when no frame is in flight.

## Room states

`idle → loading → online ⇄ generating`, with `redealing` and `waiting` as the two ways the room changes shape, and `over` as the end on a joiner's screen. The host owns the state and mirrors it with `ai-layers`; `body[data-state]` and `body[data-plan]` expose it on every screen (the emulator reads them).

| State | Meaning |
|---|---|
| `idle` | no model started; `start` is enabled when the pledges cover the model |
| `loading` | plan 1 is being loaded by every device |
| `online` | the current plan is served by every chain member; questions are accepted |
| `generating` | an answer is in flight. **The chain serving it is frozen** (`ai.gen = {v, chain}`): a join, a leave or the re-deal button only queue a plan, applied right after `ai-gendone` (the button reads "re-deal after this answer"). A plan already ticking down its 300 ms coalescing window when a question starts is held back the same way |
| `redealing` | a new plan is out (`ai-load`/`ai-wait` sent); devices whose range changed are reloading, unchanged ones re-linked. Back to `online` when every chain member of the new plan reported `ai-ready {v}` |
| `waiting` | the remaining pledges cannot hold the model (`Σ maxLayers < L`); the room says "waiting for a device" and re-plans as soon as someone joins. The last served plan is remembered as `prev` for that re-plan, so survivors keep their order and the host its range when they still fit |
| `over` | joiners only: the link to the host is gone (PeerJS close, ICE `failed`, or 7.5 s without any message from the host — it pings every 2.5 s idle, every 500 ms while answering). The answer on screen ends with "stopped: the host left", Send unlocks, the pane says "the host left — this room is over"; no further transitions |

A **leave** (connection closed, ICE `failed`, ICE `disconnected` for more than 1.5 s, three missed pongs ≈ 7.5 s while idle, or — while an answer is in flight — 2 s without any message from a chain member, the host pinging that chain every 500 ms; a late pong un-marks the peer and re-plans with it; a watchdog tick that itself came late, because the host's own thread stalled, refreshes the budget instead of judging silence it could not observe) of a chain member during `generating` ends that answer at once: the host rejects every outstanding lap with `stopped: <name> left (layers a–b)`, emits `ai-gendone` with that text so every Send box unlocks, keeps its engine, and re-plans over the remaining devices. There is no `ai-stop` message in v1: the host stops locally by rejecting laps and the room learns it through `ai-gendone`. A device whose host left is told "the host left — this room is over".

All of this runs on the AI host, which the code assumes is also the room host (the creator: the leave/join hooks — `close`, ICE, `hello`, pong revive — are wired on `isHost`, the planner and watchdogs on `ai.role === "host"`). `aiStartAnywhere` picks the biggest pledge as boss; when that is a joiner, leaves and joins do not re-plan (the hooks are no-ops there). Making the two coincide, or moving the hooks to the AI host, is open.

A **join** while the model runs gets a fair share at the next plan (the newcomer is appended to the chain; survivors keep their order, so each survivor's new range overlaps its old one and only the delta is loaded — from the Cache API on desktops). Every answer starts from `ai-reset` and position 0 (single turn), so no state migrates across a plan. The manual **re-deal** button runs the same planner. Pledge changes after start do not re-plan in v1.

## Layer plan

`planSplit({ L, layerBytes, embedBytes, host, workers, prev, exclude })` in `room/plan.js` deals the `L` layers. Inputs: `host {id, pledgeBytes, ms?}`, `workers [{id, name, pledgeBytes, webgpu, ms?}]` in any order, `prev` = the previous plan (`{chain, ranges, hostRange}`) or null, `exclude` = ids that failed to load.

a. **eligible** = workers with `webgpu !== false && pledgeBytes > 0 && !exclude.has(id)`. Order: if `prev` is null, ids sorted lexicographically; else `prev.chain` order with departed ids removed, then newcomers appended sorted by id. The stable order keeps every survivor's new range overlapping its old one.

b. **cap** (bytes): host `max(pledge − embedBytes, layerBytes/2)`, worker `max(pledge, layerBytes/2)`. `maxLayers_i = max(1, floor(cap_i / layerBytes))` for workers, `floor(cap_host / layerBytes)` for the host. `needBytes = L·layerBytes + embedBytes`; `haveBytes = Σcap + embedBytes`.

c. **fits** = `Σ maxLayers ≥ L`. If not, the plan is `{fits: false, needBytes, haveBytes, held, L}` and the room goes to `waiting`. This is stricter than the old log-only warning at `need > 1.15·have`: a plan that cannot fit is refused instead of failing at load time.

d. **speed factor**: `msRef` = median of the known `ms`; `s_i = clamp(round(4·msRef/ms_i)/4, 0.5, 2)` when `ms_i` is known, else 1. The quarter-step dead zone means autotune noise on identical GPUs cannot flip a layer; the clamp means one noisy sample cannot starve a device. With 0 or 1 measured devices every `s_i = 1`.

e. **weights** `w_i = cap_i · s_i`; `assigned_i = floor(L · w_i / Σw)`; leftover layers go to the largest fractional parts, ties by index (identical to the old inline split when all `s_i = 1`).

f. **host pin** (only with `prev`): if the host's previous count `h0` satisfies `h0 ≤ maxLayers_host`, `Σ worker maxLayers ≥ L − h0` and `L − h0 ≥ #workers`, the host keeps `h0` and only `L − h0` layers are dealt among the workers by d–e. Otherwise proportional as above. (Keeps the 27B host from reloading its 50-odd-layer shard on every event.)

g. **repair**, deterministic: while some `assigned_i > maxLayers_i` (lowest index first), move one layer to the device with the largest headroom (ties lowest index; a pinned host does not absorb). Then any worker with 0 layers takes one from the current maximum holder (the host may end at 0 layers; the engines accept an empty range with embed + head). If there are more workers than layers, the lowest-weight workers (ties: highest id first) go to `idle` and receive `ai-wait`.

h. **ranges** = prefix sums in order, host first: `hostRange = [0, a0)`, worker k `[acc, acc + a_k)`. The host keeps embed + head + MTP.

Worked example, Qwen 3.8 27B Q4_0 (64 layers of 223 970 464 B; embed + head + MTP 2 023 303 168 B). Host 16 GiB, desktop 1 GiB, phone 0.5 GiB: caps 14.12 / 1 / 0.5 GiB, maxLayers 67 / 4 / 2, chain sorted by id (phone before desktop here) → host `[0,58)`, phone `[58,60)`, desktop `[60,64)`; the log line reads `layer split by pledge×speed: you 58+embed · phone-e2e 2 · worker-e2e 4`. The desktop leaves: the host pin (58) would leave 6 layers for a phone that holds 2, so the deal is proportional → host `[0,62)`, phone `[62,64)`, note `re-dealing: host takes layers 0–61` (the note is broadcast, so it carries the host's name, "host" here, not "you"; the host-local split line keeps "you"). With a 14 GiB host instead, the same leave gives `Σ maxLayers = 58 + 2 = 60 < 64` → `fits: false`, "waiting for a device — room holds 60 of 64 layers". These literals are pinned in `tests/unit/plan_test.js`.

**Cache affinity in v1 is structural:** stable order plus contiguity means survivors reload only the delta of their range, served from the Cache API on desktops (phones and the safetensors model never cache). A worker → host report of cached layers, and spare copies of a range, are roadmap 03.

## Versioning

Protocol changes bump a message version and make older peers fail loudly at `ai-load`. Peers older than this protocol send `ai-ready` without `v`; the host ignores it, so they never come online — the same fail-loudly rule. A host older than this protocol sends `ai-load` without `v`, which a new worker ignores. See GOVERNANCE.md for what counts as a protocol change.
