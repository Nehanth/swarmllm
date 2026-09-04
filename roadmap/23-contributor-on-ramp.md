# 23 · Contributor on-ramp: labels, seeded issues, no-GPU track, hardware-verifier role

**Phase:** now (launch morning) · **Status:** planned · _now (launch morning) · hours-to-days · medium_

## Why

Merges: *Contributor on-ramp*; *Contributor ladder with criteria, CODEOWNERS, verifier role*.
Master plan NOW #6 and the weeks-2–6 metric ("ten merged community PRs") have no roadmap file, no labels and no issues; CONTRIBUTING says GPU tests run on the maintainer's hardware before merge, so every PR is gated on the one person who is also writing the paper. HN converts contributors in ~48 hours or not at all, and most arrivals lack a WebGPU GPU. Seeds already exist in the tree (`docs/models.md` pointing at `p2p.html`, the hand-duplicated `<select>`, empty `mascot()`, two files in `tests/unit/`). A "hardware verifier" rung, someone who owns a device class and runs the item-21 report, is the project-specific role no generic template supplies; the ladder text is a one-PR edit to GOVERNANCE.md and belongs inside this item, not its own.

---

## 2. Roadmap item files (top 6)

### `roadmap/12-room-liveness.md`

```markdown
# 12 · Stop, fail fast, re-deal: the room survives launch day

**Phase:** now · **Status:** planned

## Why
Two ways a room dies on Monday, both permanent. A friend closes their tab mid-answer: `conn.on("close")` (room.js:190–198) only removes the card, so the host waits out the 30 s / 90 s lap timeouts (865, 930, 968), `aiGenerate`'s catch leaves `ai.engine` set (1029–1033), `aiStart` early-returns on `ai.engine` (715), and nothing re-enables the start button (only `updateNeed` when `!ai.engine`, or the load-failure path at 805). Everyone reloads and re-types the code; guests whose host left still read "cluster online". Or a wrong-direction answer: the decode loops run to EOS or a literal 400 tokens (999, 1016) with no abort path, and at 3.5–6 tok/s cross-network that locks every screen behind `ai-busy` (1156) for up to two minutes. Roadmap 03 (spare copies, replay) is the right end state but is weeks away; this is the floor it sits on, and the master plan's NEXT metric ("median room survives one peer departure") is unreachable without it.

## Design
- **Stop.** An `ai.abort` flag checked between `specStep` iterations and between `aiPipeToken` calls; a Stop button replaces Send while busy on every screen; guests send `ai-stop`, honoured from the current asker or the host. The host still emits `ai-gendone` (`stats: "stopped at N tok"`) so every box unlocks, drops any in-flight `ai-hiddenret-b` by position (the `waiters` map already keys by pos) and leaves `ai.pos` consistent, which matters once item 13 keeps state across turns.
- **Fail fast on departure.** On `close`, if the peer is in `ai.chain` or is `ai.hostId`, reject every outstanding `ai.waiters` entry immediately with "X left (layers a–b)" and end the generation cleanly; drop the peer from `ai.chain` so `aiMaybeReady` can still fire when someone leaves during download.
- **Degraded → re-deal.** After a chain failure the room enters a "degraded" state. The host shows "Re-deal layers": the split planner in `aiStart` is factored into `planSplit()` and re-run over the remaining pledges, fresh `ai-load`s go out (cached ranges reload in seconds), and the host keeps the conversation text so it can re-prefill. The start button is re-enabled in the catch path regardless.
- **Guest side.** When the host connection closes, replace the pane with "the host left; this room is over" and a Create-room button.
- **Read `died`.** The joiner already sends `hello.died` from the localStorage crumb (361–362) and the `hello` handler ignores it; surface it as "phone came back; its tab was killed 40 s ago while streaming blk.30" so people learn that backgrounding Safari kills the peer.
- `ai-stop` is a new message and "degraded" is a new room state, so this file is the GOVERNANCE design note; `docs/protocol.md` gets a row for each. Roadmap 03 then adds spares and automatic replay on top of the same degraded → re-deal state and should be reworded to say so.

## Done when
- In a three-device room, closing one tab mid-answer surfaces the failure within 2 s, offers "Re-deal", and the next question is answered without anyone reloading.
- Pressing Stop on any screen ends generation within one lap and every Send box unlocks.
- A guest whose host left sees "this room is over" rather than "cluster online".
- `docs/protocol.md` documents `ai-stop` and the degraded state. Fail-fast, the guest message and the start-button fix are ordinary bug-fix PRs and land first.
```

### `roadmap/13-conversation.md`

```markdown
# 13 · Multi-turn conversation and an honest context limit

**Phase:** now · **Status:** planned

## Why
The room renders a chat thread, but every Send wipes the model: `aiGenerate` calls `engine.reset()`, sets `ai.pos = 0` and broadcasts `ai-reset` (room.js:884–886), then builds a single bare `user` turn with the think block pre-closed (891–896). "What did I just ask?" fails, which reads as "the model is dumb" rather than "the UI is single-turn", and it is the first thing an HN commenter tries. Nothing enforces the 512-token cap (`MAX_SEQ`, room/models.js:26): a pasted paragraph plus a 400-token answer runs positions past the KV cache with no error; the prompt box is a single-line `<input>` (p2p.html:382) that collapses newlines; answers stop dead at 400 tokens with no notice. The master plan markets 256K context, so a silent 512-token overflow is the opposite of non-negotiable 3. Items 03 (replay needs the token history) and 04 (`messages[]`) both assume conversation state that does not exist; this is their prerequisite.

## Design
- **No reset between turns.** The host keeps `ai.history = [{role, text, ids}]` and the running `ai.pos`. A new turn appends `<|im_end|>\n<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n` and prefills only the new ids; workers already hold the KV and DeltaNet state and advance by frame order, so nothing changes on the wire. `ai-reset` moves behind a visible "new chat" button. First fix the EOS asymmetry: the speculative path breaks on EOS without advancing state (1006) while the plain path pipes it through (1018); both must leave state at `pos` identically.
- **Template builder.** `room/template.js` exporting `buildIds(tok, {system?, messages, thinking})`: the one place item 04's `messages[]`, a later system-prompt setting and a thinking budget plug in, with a unit golden for the rendered ids.
- **`maxSeq` per model.** 48 of 64 layers are constant-memory DeltaNet; only 16 attention layers hold f32 KV at `maxSeq * kvDim * 4` per K and V (engine/qwen35.js:218–219), so 4096 costs tens of MB per attention layer. Put `maxSeq` in the `MODELS` entry, count KV bytes for the attention layers in each device's range inside `aiStart`'s pledge math, and send `maxSeq` in `ai-load` so workers size buffers identically.
- **Counter and refusal.** A live "143 / 4096 tokens" line under the box from the host tokenizer; refuse to send when prompt + reply cap would overflow ("start a new chat or shorten the prompt") instead of producing garbage; when the reply cap hits, show "[stopped at N tokens]" with a Continue button that resumes decode from `ai.pos` without re-prefill.
- **Textarea.** Auto-growing; Enter sends, Shift+Enter inserts a newline; phones get a Send button only.
- Bit-exactness gate: a golden asserting a two-turn chat equals a one-shot prompt containing both turns (`tests/test_reset.js` is the template).

## Done when
- A 10-turn chat on a two-device room never re-prefills history and the second answer refers to the first question.
- The room shows "context N / M"; an over-long prompt is refused with a message, never silently truncated.
- `tests/test_q38_multiturn.js` passes bit-exact on the GB10 and Mac; `docs/protocol.md` records that `ai-reset` now means "new chat" and that `ai-load` carries `maxSeq`.
```

### `roadmap/14-preflight.md`

```markdown
# 14 · Pre-flight check, join links, and a model ladder that says what this room can run

**Phase:** now · **Status:** planned

## Why
The first thing an HN reader does is open swarmllm.ai in whatever browser they have. index.html never mentions WebGPU; the only signal is "⚠ no WebGPU" on a peer card after a room already exists (room.js:125–126), followed by a start button stuck on "add 0.8 GB more". The second thing they do is paste a 4-letter code into Discord for a friend to retype: there is no deep link (`location.hash` is read once, for `#debug`, line 585). Third, the dropdown lists 0.6B first and the 27B headline last; the pledge default is half of `maxBufferSize`, a per-buffer limit rather than GPU memory (52–56, 94); `measureBudgetGB` (63) is defined and never called; phones are pinned at 0.5 GB (96) and an iPad is classified as a Mac by UA substring (40–42). A 32 GB Mac therefore presents as a 2 GB device, first-hour users pick 27B because that is the pitch, see a disabled button, and conclude it does not work. The master plan's launch metrics (join success >95%, click-to-first-token <60 s) cannot be measured if a third of visitors bounce here.

## Design
- **Pre-flight on the landing page and join screen.** Run `probeGPU()` before "Create room" is enabled and render one line per outcome with its remedy: "WebGPU works: <vendor · arch>", "Chrome on Linux: enable chrome://flags/#enable-unsafe-webgpu", "Firefox: WebGPU is not enabled on this platform; use Chrome/Edge 113+ or Safari 26+", "iOS: update to Safari 26". Classify devices with `maxTouchPoints` / `userAgentData.mobile` so iPads take the laptop path with a measured budget.
- **Asker role.** A no-WebGPU device joins as a first-class "ask only" guest (the protocol already accepts `ai-ask` from any peer) instead of a confusing 0 GB peer; on phones offer "send this room to your laptop".
- **Share links.** `/r/ABCD` pre-fills the code and auto-joins (the same URL roadmap 07 later extends with model + assignment); a QR of the link in the side panel (the seed for 05's QR signaling); Web Share API on mobile. Item 17 adds the fragment secret to the same link.
- **Model ladder.** Replace the need-bar copy with a ladder from `NEED_GB` and current pledges: "This room (2 devices, 6 GB) runs: SmolLM ✓ · 0.6B ✓ · 1.7B ✓ · 4B ✓ · 27B needs ~2 more laptops like yours". Default the select to the largest model that fits and move it as devices join; label the solo path ("alone you can run 4B now; 27B when two friends join", master plan NOW #5).
- **Measured budget.** Wire `measureBudgetGB` to a "measure my GPU (10 s)" button on desktop and use the result as the pledge default; give phones a bounded probe (128 MB chunks under the out-of-memory error scope, result cached in localStorage keyed by adapter info); one sentence of pledge copy ("GPU memory this device holds for the model; lower it if the browser feels sluggish"). In `aiStart`, count KV, DeltaNet state and snapshot slots per device instead of `pledge - embedBytes` (755–757), so a room never starts a download a device cannot finish.
- **Honest speed.** Expected tok/s per model class next to each entry, quoted from docs/bench-log.md with the hardware named (non-negotiable 3).

## Done when
- Firefox, flagless Linux Chrome and iOS 18 each get a specific remedy before any room is created; none reaches a dead start button.
- Opening `/r/ABCD` on a second device joins the room with no typing; scanning the QR does the same from a phone.
- A 32 GB Mac's default pledge reflects a measurement; the 27B is preselected when the room can run it; an iPad is treated as a tablet.
- `docs/bench-log.md` carries the per-class numbers the ladder quotes.
```

### `roadmap/15-signaling-broker.md`

```markdown
# 15 · Self-hosted signaling, vendored PeerJS, and a status canary

**Phase:** now · **Status:** planned

## Why
Every room is introduced by the free PeerJS cloud (`new Peer(..., { debug: 1, config: ICE })`, room.js:340, no `host`), and the PeerJS library comes from jsdelivr with no integrity attribute (p2p.html:13). On Show HN day a third-party best-effort broker and a CDN are two single points of failure for non-negotiable 1; room ids `swarmllm-room-XXXX` live in a global namespace anyone can squat; and the broker's operator and retention are unknown while SECURITY.md says "no counterparty". Roadmap 01 already assumes a project-run broker ("credentials issued per room by the signaling broker"), which the public cloud cannot provide, so this is 01's unstated prerequisite. Master plan NOW #4 asks for a live status page and names Petals' dead health dashboard as its loudest death signal; no item owns it.

## Design
- **Broker.** `peerjs-server` (or `local-server/server.js`, which already implements in-memory WebSocket room signaling) on the same small VPS as 01's coturn, behind `signal.swarmllm.ai`: in-memory state only, no access logs, room ids expire 60 s after the host disconnects, explicit `alive_timeout` / `concurrent_limit` / per-IP caps, and a `/turn` endpoint minting time-limited HMAC credentials per room so 01's relay is never an open relay. `/health` exposes connected peers and uptime.
- **Client.** `new Peer(id, { host: "signal.swarmllm.ai", secure: true, … })`, falling back to the PeerJS cloud when the self-hosted broker does not answer within ~3 s so the link always works; "signaling: swarmllm / fallback" in the side panel; a `?broker=` override for LAN rooms (05). Host-claims-well-known-id semantics unchanged.
- **Vendor PeerJS.** `vendor/peerjs-1.5.4.min.js` with the upstream URL and sha384 recorded in `vendor/README.md`; no CDN on the critical path.
- **Status canary.** Upptime-style, zero infrastructure: a GitHub Actions cron every 5 min runs checks and writes `status.json` to a `status` branch; a static `/status` page renders it and the landing page shows one line. Checks: `/room` returns 200 with the expected `BUILD` (item 16); broker `/health` plus a WebSocket handshake; a TURN allocation; an HF `Range: bytes=0-1023` request per model URL returning 206 with the pinned size (item 19); "demo room: online · N devices · 27B" from item 20's heartbeat, or an honest "demo room: not yet". Failures open an issue and email the maintainer. The headless-Chrome Qwen 0.6B golden check needs a self-hosted runner and degrades to the HTTP/WebSocket/TURN/HF probes on hosted runners.
- **Docs.** `docs/broker.md`; a SECURITY.md section "What the signaling server and relay can see" (IPs, room codes, connection timing, SDP fingerprints; never text, weights or activations) and "What we log" (nothing beyond aggregate counters); a capacity note in `docs/protocol.md` (one WebSocket per tab, no model traffic, so a single small box handles thousands of tabs).
- Adding a server and vendoring a dependency are "substantial" under GOVERNANCE; this file is the design note.

## Done when
- With `0.peerjs.com` blocked, a room forms via `signal.swarmllm.ai`; with the VPS down, it falls back to the cloud and says so in the panel.
- `p2p.html` loads no third-party script.
- `/status` is live, linked from the Show HN post, and pages the maintainer when the broker, TURN or an HF probe fails.
- SECURITY.md names the broker's operator and retention; roadmap 01 cross-links this item as its prerequisite.
```

### `roadmap/16-fail-loudly.md`

```markdown
# 16 · Fail loudly and legibly: actionable errors, a diagnostic report, build and protocol version

**Phase:** now · **Status:** planned

## Why
Launch-day failures will be signaling, Hugging Face and mixed versions, not kernels, and every one of them is illegible today. `peer.on("error")` falls through to "error: <type>" (room.js:374–382); `rangeFetch` throws "model host refused range requests" for every non-206 including 429 and 503 (444); a joiner waits 15 s in silence before "no room with that code"; on phones the `#chat-log` that carries every warning ("needs ~16.5 GB", "X left", the layer split) is `display: none` (p2p.html:302). The site deploys continuously from `main`, so a tab opened hours ago hosts for a friend who just loaded new code; docs/protocol.md, CONTRIBUTING.md and RELEASE.md all promise a message version that makes older peers fail loudly, but `hello` (291, 362, 370) and `ai-load` (779) carry none, so mixed rooms fail as NaNs and timeouts. Bug reports arrive as screenshots with no commit; CHANGELOG has everything since 0.1.0 under Unreleased while package.json says 0.2.0 and CITATION.cff already claims a 2026-09-07 release. One maintainer cannot debug "my Mac made NaNs" from a screenshot, and there is no written telemetry policy for the HN commenter who will ask.

## Design
- **`explain(err)` table.** Map PeerJS `err.type`, the HTTP status from `rangeFetch` (206 refused vs 429/503 vs CORS/offline), GPU error prefixes and each timeout to one plain sentence plus what to do ("Hugging Face is rate-limiting downloads (429). Wait a minute and press start again; cached layers are kept."). Say "still connecting…" at 5 s; extend the join timeout only while ICE is still `checking`.
- **Mobile.** Route warnings through `toast()` (room.js:15) or a collapsible events sheet instead of the hidden log.
- **Copy diagnostic report.** A button in the room and on every failure state assembling JSON: build, protocol, browser and OS family, `adapter.info` and limits, autotune result, self-test outcome, first GPU error, the last 20 crumbs, role, layer range, lap/RTT/bandwidth and speculation stats. Excluded: prompt text, tokens, peer names, IPs, room code (the same exclusion list 07's stats will use). Copies to the clipboard and opens a prefilled issue; `010-bug-room.yml` and `011-bug-output.yml` gain a report field. No collection endpoint; SECURITY.md gains "What the site collects: nothing", and CI fails on any `fetch`/`sendBeacon` to a host outside Hugging Face and the broker.
- **Versions.** `room/version.js` exports `PROTOCOL = 1` and `BUILD` (git short SHA written to `build.json` by the deploy scripts; no bundler). `{proto, build}` travels in `hello`, `ai-load` and `ai-ready`; the host refuses a mismatched `proto` with "X is on an older version, ask them to reload", and a worker refuses an `ai-load` it does not understand. `hello` also carries a capability map (batch columns, f16 wire, rollback slots) so the host picks the lowest common configuration. `BUILD` shows in the status line and the crumb; the page polls `/version.json` while idle and offers a reload, never during a generation. RELEASE.md: MINOR bump = `PROTOCOL` bump; CHANGELOG gains a "Protocol" heading.
- **Launch release.** Tag `v0.2.0`, move Unreleased, fix CITATION.cff. `scripts/deploy-prod.sh` deploys, runs the item-15 smoke checks against the fresh deployment URL and only then aliases swarmllm.ai; `scripts/rollback.sh` re-aliases the last good deployment and is drilled once against staging before Monday. `vercel.json` headers set `no-cache` on HTML and `room.js` and a short max-age on `engine/*.js` so a rollback takes effect on reload. Launch-week freeze on `main` except hotfixes; staging stays the playground.

## Done when
- Every failure a tester can provoke (blocked broker, HF 429, no WebGPU, peer gone, mixed versions) shows one sentence and a next step, on desktop and on a phone.
- A bug or benchmark report filed from the live site names the exact commit; a `proto` mismatch produces a toast, never a hang.
- `v0.2.0` is tagged and is what swarmllm.ai serves on Monday; `rollback.sh` has been run once.
- SECURITY.md states what the site collects and CI enforces it.
```

### `roadmap/17-security-sweep.md`

```markdown
# 17 · Security sweep before Show HN: host-authoritative protocol, honest SECURITY.md, hardened page

**Phase:** now · **Status:** planned

## Why
Every peer in a room is fully trusted for every message. `ai-next`, `roster` and `ai-start-req` (room.js:237–253) and `ai-load` (1049–1053) are honoured from any sender, so anyone who guesses a 4-character code (30-symbol alphabet, `rand(4)` at 320; the host id `swarmllm-room-<code>` is predictable, 340; the public broker answers "peer-unavailable" instantly) can rewire the chain and receive other people's activations, make a victim download 15 GB, or re-seat into a named peer's slot (`aiRejoin` keys on display name, 811). `ai-hidden-b` allocates `(d.n || 4) * dim` floats from an unchecked `n` (1093–1095); `pledge` trusts `d.gb` (261) and it drives the layer split; once in, there is no lock or kick. Separately, SECURITY.md line 12 says "Other devices never receive your prompt text … or the chosen tokens" while the host broadcasts the prompt (`ai-genstart {text}`, 902) and every token (`ai-token`, 950) to every peer by design, and line 21 lists peer metadata as "length and timing" while `hello`/`roster` fan out GPU vendor + architecture, device class, pledged GB, display name and the crash crumb (`died`, 361–362; `broadcastRoster`, 281). The page sets no security headers (vercel.json has only rewrites) and its one runtime dependency is an unpinned CDN script. All of this is discoverable by anyone reading room.js after launch, and non-negotiable 3 makes the wording gap the worst of it.

## Design
- **Authorization by role.** A `room/validate.js` module (no GPU, unit-tested) that `onData`/`aiOnData` call first: only `ai.hostId` may send `ai-load`, `ai-wait`, `ai-next`, `ai-reset`, `ai-layers`, `ai-hostprog`, `ai-ready-all`, `ai-genstart`, `ai-token`, `ai-gendone`, `ai-rollback`, `roster`, `ai-start-req`; workers accept `ai-hidden`/`ai-hidden-b` only from the chain predecessor named in their `ai-load` (add `prev`); the host accepts `ai-ready`/`ai-progress`/`ai-error`/`ai-hiddenret*` only from ids in `ai.chain`; `ai-start-req` is validated by recomputing `biggestPeerId()` rather than trusting `d.boss`. Everything else is dropped and logged as "ignored <t> from <name>".
- **Field validation and quotas.** `range` within `[0, L)`; `n ∈ {4,…,16}` and a multiple of `NC`; `byteLength === n*dim*2` (or `*4` for f32); `pos`/`basePos` monotonic and below `maxSeq`; `pct` 0–100; `gb ≤ 64`; name and text length caps; binary payloads accepted only while that peer's `bw-start` is open and capped at the test size. A per-peer token bucket (control messages per second, asks per 5 s, one bandwidth test per minute) with a visible "alice is being rate-limited" line and disconnect after repeated violations.
- **Resume token, stronger codes, lock, kick.** `ai-load` hands out a random 16-byte resume token that the worker stores in sessionStorage and presents in `hello.resume`; `aiRejoin` keys on it, not the name. Created rooms use 6-character codes plus a fragment secret (`/r/CODE#SECRET`, carried by item 14's share link and accepted as `CODE-SECRET` in the join box) that the host requires in `hello`. A host-side "lock room" toggle (auto-on when the model starts) rejects new `hello`s with `ai-locked`; an "x" on each peer card (host only) closes the connection, adds the id and its resume token to a session deny set, re-broadcasts the roster, and marks the range vacant for item 12's re-deal.
- **SECURITY.md truth pass.** Split "Text stays on the host" into (a) the compute path: workers receive only activations, and (b) the product: a room is a shared chat and everyone in it sees every question and answer; list the metadata actually exchanged (name, device class, GPU string, pledged memory, layer range, per-token timing); add a `hello.meta` allow-list and keep `died` local (item 12 reads it from the crumb of the rejoining device's own report instead); state what a code and secret protect against and what they do not ("anyone with the link is a member until kicked"). Put the same sentence in the room's first swarm-log line. `tests/unit/wire_privacy.test.js` asserts the field set of every outbound message type against the documented table, so any future change trips GOVERNANCE's "what peers can learn" rule in CI.
- **Page hardening.** Vendored PeerJS (item 15) or SRI; `vercel.json` headers: a CSP with `script-src 'self'`, `connect-src` limited to the Hugging Face hosts, the broker and TURN, `frame-ancestors 'none'`, `base-uri 'none'`, `media-src data:` for the wake-lock video; plus HSTS, `Referrer-Policy: no-referrer` (HF requests otherwise carry the room page as Referer), `X-Content-Type-Options: nosniff` and a restrictive `Permissions-Policy`. Move index.html's inline script (line 330) into `landing.js`; pin GitHub Actions to commit SHAs. Validate the CSP on staging against the real HF LFS hostnames before production so weight fetches keep working.
- The "who may send what" table goes into `docs/protocol.md`; this file is the design note for `ai-locked`, the `prev`/`resume` fields and the lifecycle change. The SECURITY.md rewrite and the swarm-log line are a small PR that lands first.

## Done when
- `tests/unit/room_authz.test.js` feeds forged `ai-load`, `ai-next`, `roster`, an oversized `ai-hidden-b` and a spoofed `hello` through the validator and all are dropped; a peer repeating them is rate-limited, then disconnected.
- SECURITY.md, `docs/protocol.md` and the room's first log line agree on what peers see; the privacy test passes in CI.
- The host can lock a running room and kick a peer; a kicked peer cannot rejoin under its old name.
- `curl -I swarmllm.ai/room` shows the CSP and headers; the page loads no unpinned third-party script.
```

---

## 3. Folded below the cap (kept, not ranked)

| Proposal | Where it lives now |
|---|---|
| Bring-your-own GGUF (×3: *any GGUF URL*, *Community model catalogue*, *custom URL for fine-tunes*) | One "next" item to file as 24 when a slot opens: catalogue JSON + `?model=<url>` restricted to `qwen3`/`qwen35` architectures, absorbing 06's entries. Correct and on-strategy (master plan §6), but 06 ships the launch models and the top 12 are all launch or paper. Note: `examples/custom-model` does not exist; dense entries need cfg/tok URLs. |
| *Sampling controls and deterministic mode*; *System prompt and thinking control* (settings half) | Follow-on to 13's `room/template.js`; greedy (`temperature = 0`) is the advertised cross-device bit-exact demo, seeded sampling is per-host-browser only (`Math.exp` ulp differences). Settings travel in `ai-genstart`, so a protocol note. |
| *Question queue*; copy/transcript from *Stop, copy, transcript* | Inside 20 (queue) and 07 (share card / stats line). |
| *Per-hop lap telemetry and throughput-weighted split*; *Latency-aware chain placement*; *Phone-class peers*; *Heterogeneous device matrix*; *Load time / TTFT columns*; *Mac/Metal profile* | Inside 21. The weighted split and placement are its follow-ons and 09's gate; placement stays master-plan LATER. |
| *Request and stream contract*; *examples/ + engine-api.md*; *@swarmllm/client* | Inside 22 (contract and examples) and as 22's follow-on (SDK), after 04 so attribution and versioning are designed once. |
| *Contributor ladder*; *Hardware report bundle* (schema half) | Inside 23 and 21 respectively; the ladder is a one-PR GOVERNANCE.md edit, not a slot. |
| *Real-browser cross-network testbed (Playwright + netem)* | Follow-on to 22's transport seam and 01; weeks of work, and the netem harness makes 01's done-criterion testable. File after 22. |
| *Weight download parallel streams*; *Phone weight cache*; *Model manager* | Inside 18 and 19. |

## 4. Rejected

- **Constrained decoding / JSON schema / tool calling.** Real engine work that 04's "JSON mode" quietly depends on, but it changes model output under a per-request switch, needs a grammar engine plus snapshot/rollback of grammar state across rejected draft columns, and has no consumer until 04 and 22 exist. Declined for the roadmap now; reopen as a research-template issue when 04 ships, and until then 04 must stop promising "JSON mode".
- **`/v1/embeddings` via an embedding model on the host.** A second model kind, a `finalHidden` readback path, a new endpoint and a golden, all downstream of 04 and 22. A follow-on line in 04, not a roadmap item.
- **Docs site at swarmllm.ai/docs.** `.vercelignore` excluding `docs` is a deliberate no-build choice; GitHub renders the docs, `md()` has no link or table support so "reuse the markdown module" is really a renderer project, and the launch need (a troubleshooting page) is met by 16's `explain()` table and diagnostic report plus a README FAQ. Revisit once 21 produces a benchmarks page worth hosting.
- **Battery and thermal awareness.** No web thermal API and no Battery API on iOS, so most of it is inference from timing that 21's per-hop telemetry already provides; the energy column joins 21's device matrix. The one concrete piece, dropping the silent-video hack when the real Wake Lock succeeded (room.js:387–408 starts both unconditionally), is a small PR, not a slot.
- **Autotune the batched, head and gate/up shapes and persist.** The tuning half belongs under 02, which is actively changing the batched path; the proposed f16 `_h` and DP4a candidates change numerics and must never be auto-selected (non-negotiable 4), which removes most of the claimed upside. Persisting the existing autotune result in localStorage is a small PR that 21's load-time measurement will justify or not.
- **"Lend a GPU" volunteer worker role in the demo room.** Contradicts the proposal's own safeguard (maintainer devices hold layers) and the README exclusion of stranger swarms by default; a volunteer-held spare promoted by 03 can corrupt output before 08 exists. The demo room ships ask-only (item 20); this returns as a follow-on gated on 03 and 08.
- **Community benchmark database harvest action.** The GitHub Action that scrapes issues into `benchmarks/reports/` and regenerates a page is a build step with a moderation problem (unverified rows presented as the project's numbers). 21 keeps the schema, `npm run report`, the copy-report button and template validation; harvesting is manual until there are enough reports to need it.
- **Release cadence (monthly tags, quarterly "state of the swarm" post, "paper mode" banner).** Policy lines for RELEASE.md and the roadmap README, not deliverables; the launch tag, build id and rollback drill are in 16.
- **Proposal-side claims dropped as unsupported by the tree:** SECURITY.md line references 285/294 (file is 34 lines; the quotes are lines 12 and 21); "`measureBudgetGB` runs unbounded until WebKit kills the tab" (it takes a `capGB` argument and is never called; the defect is that no measurement path exists); `performance.memory` sent over the wire (only the localStorage crumb carries it); references to `examples/custom-model`, `engine/generate.js` and "item 17" as existing.

## Design

See the merged proposals in [docs/roadmap-review.md](../docs/roadmap-review.md) under item 23; turn them into a design note before building.

## Done when

- Acceptance criteria to be written with the design note.
