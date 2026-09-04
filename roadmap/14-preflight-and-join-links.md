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
