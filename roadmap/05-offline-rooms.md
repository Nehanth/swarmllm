# 05 · Offline rooms: hotspot + QR signaling + peer weight sharing

**Phase:** next · **Status:** planned

## Why
SwarmLLM only ever needed a network, not the internet. Classrooms, ships, clinics, and internet shutdowns all have local Wi-Fi (a phone hotspot, a router, mesh routers) but no upstream. Three things currently assume the internet: loading the page, the signaling broker, and downloading weights.

## Design
- **Offline-loadable app:** a PWA with a service worker caching the whole app, and a single-file build that can be passed around.
- **QR-code signaling:** the host shows its WebRTC offer (local IP + port + DTLS fingerprint) as a QR code; the peer scans it and shows its answer; no server at any point.
- **Peer weight sharing:** a device that has a layer range cached serves it to others over the local network, so one cached copy seeds a room with no Hugging Face access.
- Bandwidth reality (measured): local Wi-Fi is the only viable link; Bluetooth is ~1 Mbit/s and browser-to-browser is impossible from a web page; packet radio is kbit/s. See the discussion in `docs/`.

## Done when
- With Wi-Fi on and the upstream unplugged, three devices with cached weights form a room from QR codes and answer a question.
