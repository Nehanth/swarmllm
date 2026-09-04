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
