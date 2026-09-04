# 11 · Native peer for headless GPUs

**Phase:** later · **Status:** planned

## Why
Servers, Jetsons and gaming PCs without a browser open should be able to join a room. The engine already runs headless in Deno (every benchmark in this repo does); the only missing piece is WebRTC in a non-browser runtime.

## Design
- `swarmllm join <room>`: a Deno-compiled binary using the same engine modules plus a WebRTC implementation (`werift` or `node-datachannel`) speaking the existing room protocol.
- The same process is the natural home for the local endpoint (roadmap 04).
- A native peer never suffers tab throttling and can hold larger layer ranges.

## Done when
- A headless Linux machine joins a browser room and serves layers; the browser peers see no difference.
