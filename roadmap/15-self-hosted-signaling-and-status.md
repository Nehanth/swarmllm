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
