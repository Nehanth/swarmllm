// Per-hop timing/bytes for cross-network laps (roadmap 25 · A4,
// docs/research/network-scheduler.md §2.4). Hop entries are durations only
// (performance.now() deltas); peer clocks are not synchronized, so nothing here
// assumes a shared clock or compares timestamps across devices.

const MAX_SAMPLES = 200;   // rolling window per peer / for transport, so a long room
                            // session doesn't grow the sample arrays unboundedly

export const round1 = (x) => Math.round(x * 10) / 10;

// nearest-rank percentile over a sample set; good enough for a live p50/p90 readout,
// not a substitute for docs/bench-log.md's controlled measurements
export function percentile(samples, p) {
  if (!samples.length) return 0;
  const sorted = [...samples].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx];
}

// host: transport = lapMs − Σ peer computeMs − host's own pack/compute overhead.
// Peer encodeMs is left inside the transport residual, matching the formula in
// network-scheduler.md §2.4 — it is serialized before the byte hits the wire, and
// splitting it out precisely needs a synchronized clock, which peers don't have.
export function deriveTransport(lapMs, hops, hostPackMs = 0) {
  const computeMs = hops.reduce((s, h) => s + (h.computeMs || 0), 0);
  return Math.max(0, lapMs - computeMs - hostPackMs);
}

export function makeHopStats() { return { byPeer: new Map(), transportMs: [] }; }

function push(arr, v) { arr.push(v); if (arr.length > MAX_SAMPLES) arr.shift(); }

// Record one lap's telemetry (host only — a lap's `hops[]` is only ever fully
// populated once it reaches back to the host). Returns this lap's derived transport
// time so the caller can use it immediately without waiting for a summary.
export function recordLap(stats, { lapMs, hops, hostPackMs = 0 }) {
  for (const h of hops) {
    const arr = stats.byPeer.get(h.peer) || [];
    push(arr, h.computeMs || 0);
    stats.byPeer.set(h.peer, arr);
  }
  const transportMs = deriveTransport(lapMs, hops, hostPackMs);
  push(stats.transportMs, transportMs);
  return transportMs;
}

// p50/p90 per peer (computeMs) and for the derived transport — feeds the room UI
// and a docs/bench-log.md row.
export function summarizeHopStats(stats) {
  const perHop = {};
  for (const [peer, samples] of stats.byPeer)
    perHop[peer] = { p50: round1(percentile(samples, 50)), p90: round1(percentile(samples, 90)) };
  return {
    perHop,
    transport: { p50: round1(percentile(stats.transportMs, 50)), p90: round1(percentile(stats.transportMs, 90)) },
  };
}
