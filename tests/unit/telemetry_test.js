// room/telemetry.js: per-hop aggregation for roadmap 25 · A4 (per-hop telemetry).
// transport derivation, percentile math, and the rolling per-peer/transport stats.
import { assert, assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { deriveTransport, percentile, makeHopStats, recordLap, summarizeHopStats } from "../../room/telemetry.js";

Deno.test("deriveTransport subtracts peer compute and host pack/compute from the lap", () => {
  const hops = [{ peer: "a", computeMs: 20 }, { peer: "b", computeMs: 15 }];
  assertEquals(deriveTransport(100, hops, 10), 55);   // 100 - (20+15) - 10
});

Deno.test("deriveTransport floors at zero (never a negative network time)", () => {
  const hops = [{ peer: "a", computeMs: 90 }];
  assertEquals(deriveTransport(50, hops, 20), 0);
});

Deno.test("deriveTransport treats a missing computeMs as zero, not NaN", () => {
  const hops = [{ peer: "a" }, { peer: "b", computeMs: 5 }];
  assertEquals(deriveTransport(30, hops), 25);
});

Deno.test("percentile: p50/p90 on a known set (nearest-rank)", () => {
  const samples = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
  assertEquals(percentile(samples, 50), 50);
  assertEquals(percentile(samples, 90), 90);
  assertEquals(percentile([], 50), 0);
});

Deno.test("percentile does not require pre-sorted input and does not mutate it", () => {
  const samples = [30, 10, 20];
  const before = [...samples];
  assertEquals(percentile(samples, 50), 20);
  assertEquals(samples, before);
});

Deno.test("recordLap folds hops into per-peer stats and returns this lap's transport", () => {
  const stats = makeHopStats();
  const transportMs = recordLap(stats, {
    lapMs: 120,
    hops: [{ peer: "worker-1", computeMs: 30 }, { peer: "worker-2", computeMs: 25 }],
    hostPackMs: 5,
  });
  assertEquals(transportMs, 60);   // 120 - (30+25) - 5
  assertEquals(stats.byPeer.get("worker-1"), [30]);
  assertEquals(stats.byPeer.get("worker-2"), [25]);
  assertEquals(stats.transportMs, [60]);
});

Deno.test("summarizeHopStats reports p50/p90 per peer and for the derived transport", () => {
  const stats = makeHopStats();
  for (const computeMs of [10, 20, 30, 40, 50])
    recordLap(stats, { lapMs: 100, hops: [{ peer: "worker-1", computeMs }], hostPackMs: 0 });
  const { perHop, transport } = summarizeHopStats(stats);
  assertEquals(perHop["worker-1"].p50, 30);
  assertEquals(perHop["worker-1"].p90, 50);
  // transport for each lap = 100 - computeMs -> [90, 80, 70, 60, 50]
  assertEquals(transport.p50, 70);
  assertEquals(transport.p90, 90);
});

Deno.test("hop sample arrays stay bounded across a long room session", () => {
  const stats = makeHopStats();
  for (let i = 0; i < 500; i++) recordLap(stats, { lapMs: 100, hops: [{ peer: "worker-1", computeMs: i }] });
  assert(stats.byPeer.get("worker-1").length <= 200, "per-peer samples should be capped");
  assert(stats.transportMs.length <= 200, "transport samples should be capped");
});

Deno.test("multi-hop chain: each peer's compute is attributed to that peer, not blended", () => {
  const stats = makeHopStats();
  // 3-peer chain, host pack/compute 8ms, lap 150ms
  recordLap(stats, {
    lapMs: 150,
    hops: [{ peer: "phone", computeMs: 60 }, { peer: "laptop", computeMs: 20 }, { peer: "desktop", computeMs: 10 }],
    hostPackMs: 8,
  });
  const { perHop, transport } = summarizeHopStats(stats);
  assertEquals(perHop.phone.p50, 60);
  assertEquals(perHop.laptop.p50, 20);
  assertEquals(perHop.desktop.p50, 10);
  assertEquals(transport.p50, 52);   // 150 - (60+20+10) - 8
});
