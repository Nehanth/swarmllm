# 07 · Persistent rooms and contribution stats

**Phase:** next · **Status:** planned

## Why
A room today is a session you lose. Persistent room links that remember the model and layer assignment make rejoining instant (weights are cached), and per-device stats ("your Mac served 4,812 tokens") are the retention mechanic volunteer computing has relied on for twenty years, with no backend needed.

## Design
- Room links like `/r/<name>` storing model + assignment in the URL or the host's localStorage; rejoin re-attaches from cache.
- Per-tab counters (tokens served, layers held, uptime) and lifetime stats in localStorage; a one-click plain-text share card.
- No accounts: identity is the browser.

## Done when
- Rejoining a named room re-establishes the same split from cache in seconds; the room shows live per-device counters.
