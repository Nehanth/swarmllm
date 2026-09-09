// room/plan.js: the layer planner. The first deal must reproduce the split aiStart used to
// compute inline (the "legacy oracle" below is that code, verbatim); everything else checks
// the invariants docs/protocol.md § "Layer plan" promises.
import { planSplit, describeSplit, planNote } from "../../room/plan.js";

const GiB = 2 ** 30;
// real 0.6B Q8 / 4B Q8 (approx) / 27B Q4_0 index sizes (bytes, from the GGUF headers)
const DIMS = {
  q06: { L: 28, layerBytes: 16720896, embedBytes: 165306368 },
  q4b: { L: 36, layerBytes: Math.round(0.11 * GiB), embedBytes: Math.round(0.6 * GiB) },
  q27: { L: 64, layerBytes: 223970464, embedBytes: 2023303168 },
};
const assert = (c, m) => { if (!c) throw new Error(m || "assert"); };
const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };

// room.js aiStart 828-842 before the planner moved out (pledges already in bytes)
function legacy(L, layerBytes, embedBytes, hostPledge, workerPledges) {
  const parts = [
    { cap: Math.max(hostPledge - embedBytes, layerBytes / 2) },
    ...workerPledges.map((p) => ({ cap: Math.max(p, layerBytes / 2) })),
  ];
  const totalCap = parts.reduce((s, p) => s + p.cap, 0);
  const assigned = parts.map((p) => Math.floor(L * p.cap / totalCap));
  const fracs = parts.map((p, i) => ({ i, f: L * p.cap / totalCap - assigned[i] })).sort((a, b) => b.f - a.f);
  let rem = L - assigned.reduce((a, b) => a + b, 0);
  for (let k = 0; k < rem; k++) assigned[fracs[k % fracs.length].i]++;
  for (let i = 1; i < assigned.length; i++)
    if (assigned[i] === 0) { const j = assigned.indexOf(Math.max(...assigned)); assigned[j]--; assigned[i]++; }
  return assigned;
}

// small deterministic PRNG
function rng(seed) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 2 ** 32; }; }
const idOf = (i) => "peer-" + String.fromCharCode(97 + (i % 26)) + (i * 7919 % 1000);

function room(seed, dims) {
  const r = rng(seed);
  const n = 2 + Math.floor(r() * 11);   // 2..12 devices
  const pledge = () => Math.round((0.5 + r() * 23.5) * 2) / 2 * GiB;
  const host = { id: "host-" + seed, name: "host", pledgeBytes: pledge() };
  const workers = [];
  for (let i = 1; i < n; i++) workers.push({ id: idOf(seed * 31 + i), name: "w" + i, pledgeBytes: pledge(), webgpu: true });
  return { ...dims, host, workers };
}

function checkInvariants(plan, input) {
  const { L } = input;
  assert(plan.fits, "fits");
  const order = [input.host.id, ...plan.chain.map((c) => c.id)];
  let acc = 0;
  for (const id of order) {
    const r = plan.ranges[id];
    eq(r[0], acc, "contiguous " + id);
    assert(r[1] >= r[0], "range order");
    acc = r[1];
    assert(r[1] - r[0] <= plan.weights[id].maxLayers, `over cap ${id}: ${r[1] - r[0]} > ${plan.weights[id].maxLayers}`);
  }
  eq(acc, L, "covers L");
  for (const c of plan.chain) assert(c.range[1] - c.range[0] >= 1, "worker with 0 layers");
  eq(plan.assigned, order.map((id) => plan.ranges[id][1] - plan.ranges[id][0]), "assigned matches ranges");
  eq(plan.hostRange, plan.ranges[input.host.id], "hostRange");
  const nElig = input.workers.filter((w) => w.webgpu !== false && w.pledgeBytes > 0).length;
  if (plan.idle.length) assert(nElig > L, "idle only when workers > L");
  eq(plan.chain.length + plan.idle.length, nElig - (input.exclude ? [...input.exclude].filter((id) => input.workers.some((w) => w.id === id && w.webgpu !== false && w.pledgeBytes > 0)).length : 0), "every eligible worker is in the chain or idle");
}

Deno.test("plan: first deal equals the legacy inline split (300 seeded rooms)", () => {
  let same = 0, repaired = 0;
  for (let seed = 1; seed <= 300; seed++) {
    const dims = [DIMS.q06, DIMS.q4b, DIMS.q27][seed % 3];
    const input = room(seed, dims);
    const sorted = [...input.workers].sort((a, b) => (a.id < b.id ? -1 : 1));
    const oracle = legacy(dims.L, dims.layerBytes, dims.embedBytes, input.host.pledgeBytes, sorted.map((w) => w.pledgeBytes));
    const plan = planSplit(input);
    if (!plan.fits) { assert(plan.held < dims.L, "fits:false only when held < L"); continue; }
    checkInvariants(plan, input);
    eq(plan.chain.map((c) => c.id), sorted.map((w) => w.id), "lexicographic order");
    const maxL = [plan.weights[input.host.id].maxLayers, ...sorted.map((w) => plan.weights[w.id].maxLayers)];
    const oracleOk = oracle.every((a, i) => a <= maxL[i]) && input.workers.length <= dims.L;
    if (oracleOk) { eq(plan.assigned, oracle, `seed ${seed}`); same++; } else repaired++;
  }
  assert(same > 100, "oracle compared on too few rooms: " + same);
  console.log(`  legacy-identical ${same}, repaired ${repaired}`);
});

Deno.test("plan: deterministic and order-independent", () => {
  for (let seed = 400; seed < 430; seed++) {
    const input = room(seed, DIMS.q27);
    const a = planSplit(input), b = planSplit(input);
    eq(a, b, "same input twice");
    const shuffled = { ...input, workers: [...input.workers].reverse() };
    eq(planSplit(shuffled), a, "shuffled input");
    if (a.fits) checkInvariants(a, input);
  }
});

const three = (ms = {}) => ({
  ...DIMS.q27,
  host: { id: "h", name: "host", pledgeBytes: 16 * GiB, ms: ms.h },
  workers: [
    { id: "a", name: "A", pledgeBytes: 4 * GiB, webgpu: true, ms: ms.a },
    { id: "b", name: "B", pledgeBytes: 4 * GiB, webgpu: true, ms: ms.b },
    { id: "c", name: "C", pledgeBytes: 4 * GiB, webgpu: true, ms: ms.c },
  ],
});
const count = (plan, id) => plan.ranges[id][1] - plan.ranges[id][0];

Deno.test("plan: speed weight moves layers but never past the memory cap", () => {
  const base = planSplit(three({ h: 100, a: 100, b: 100, c: 100 }));
  const flat = planSplit(three());
  eq(base.assigned, flat.assigned, "equal speeds == no ms");
  const slow = planSplit(three({ h: 100, a: 200, b: 100, c: 100 }));   // A at 2x median
  assert(count(slow, "a") <= count(base, "a"), "slower device never gains");
  eq(slow.weights.a.s, 0.5, "s=0.5 at 2x median");
  const fast = planSplit(three({ h: 100, a: 25, b: 100, c: 100 }));    // A at median/4
  eq(fast.weights.a.s, 2, "clamped at 2");
  assert(count(fast, "a") <= 2 * count(base, "a") + 1, "at most ~2x share");
  assert(count(fast, "a") <= fast.weights.a.maxLayers, "memory cap beats speed");
  const noisy = planSplit(three({ h: 100, a: 103, b: 100, c: 100 }));
  eq(noisy.assigned, base.assigned, "100 vs 103 ms: identical plan");
  const one = planSplit(three({ a: 50 }));
  eq(one.assigned, flat.assigned, "a single measured device is s=1");
  eq(one.weights.a.s, 1);
  const missing = planSplit(three({ h: 100, a: 100 }));
  eq(missing.weights.b.s, 1, "missing ms == 1");
  // memory cap beats speed: a fast but tiny device stays tiny
  const tiny = planSplit({ ...three({ h: 100, a: 25, b: 100, c: 100 }), workers: [{ id: "a", name: "A", pledgeBytes: 0.5 * GiB, webgpu: true, ms: 25 }, { id: "b", name: "B", pledgeBytes: 8 * GiB, webgpu: true, ms: 100 }] });
  assert(count(tiny, "a") <= tiny.weights.a.maxLayers, "tiny fast device capped");
  checkInvariants(tiny, { ...three(), workers: [{ id: "a", pledgeBytes: 1, webgpu: true }, { id: "b", pledgeBytes: 1, webgpu: true }] });
});

Deno.test("plan: does not fit, and who is never dealt", () => {
  const small = planSplit({ ...DIMS.q27, host: { id: "h", pledgeBytes: 4 * GiB }, workers: [{ id: "a", name: "A", pledgeBytes: 1 * GiB, webgpu: true }] });
  eq(small.fits, false);
  eq(small.L, 64);
  assert(small.held < 64 && small.held > 0, "held reported");
  assert(small.needBytes > small.haveBytes, "need > have");
  const p = planSplit({ ...three(), exclude: new Set(["b"]), workers: [
    ...three().workers, { id: "g", name: "guest", pledgeBytes: 8 * GiB, webgpu: false }, { id: "z", name: "zero", pledgeBytes: 0, webgpu: true }] });
  eq(p.chain.map((c) => c.id), ["a", "c"], "excluded / no webgpu / zero pledge never appear");
  assert(!("b" in p.ranges) && !("g" in p.ranges) && !("z" in p.ranges));
});

Deno.test("plan: leave keeps survivor order, host pin keeps host range", () => {
  const unit = { L: 64, layerBytes: GiB, embedBytes: 0 };
  const prev = { chain: ["a", "b", "c"], ranges: { h: [0, 8], a: [8, 28], b: [28, 48], c: [48, 64] }, hostRange: [0, 8] };
  const input = { ...unit, host: { id: "h", name: "host", pledgeBytes: 8 * GiB }, prev,
    workers: [{ id: "c", name: "C", pledgeBytes: 40 * GiB, webgpu: true }, { id: "a", name: "A", pledgeBytes: 40 * GiB, webgpu: true }] };
  const p = planSplit(input);
  checkInvariants(p, input);
  eq(p.chain.map((c) => c.id), ["a", "c"], "prev order minus the departed");
  eq(p.hostRange, [0, 8], "host pinned");
  eq(p.pinned, true);
  const overlap = (x, y) => Math.max(0, Math.min(x[1], y[1]) - Math.max(x[0], y[0]));
  assert(overlap(p.ranges.a, prev.ranges.a) > 0, "A overlaps its old range");
  assert(overlap(p.ranges.c, prev.ranges.c) > 0, "C overlaps its old range");
  const note = planNote(prev, p, (id) => ({ h: "you", a: "A", c: "C" })[id]);
  assert(/^re-dealing: /.test(note) && /A takes layers 8–35/.test(note), note);
  eq(planNote(p, p, (id) => id), "", "unchanged plan -> empty note");
});

Deno.test("plan: join appends the newcomer last with at least one layer", () => {
  const input = three();
  const first = planSplit(input);
  const joined = { ...input, prev: first, workers: [...input.workers, { id: "d", name: "D", pledgeBytes: 2 * GiB, webgpu: true }] };
  const p = planSplit(joined);
  checkInvariants(p, joined);
  eq(p.chain.map((c) => c.id), ["a", "b", "c", "d"], "survivors first, newcomer last");
  assert(count(p, "d") >= 1, "newcomer has layers");
  assert(/D takes layers/.test(planNote(first, p, (id) => id.toUpperCase())), "note names the newcomer");
  // the previous plan is passed as planSplit returned it (chain entries are objects): the
  // survivors' order must come from it, not from the id sort (a re-deal in the room shuffled the
  // chain and reloaded every survivor before this was checked)
  const shuffled = { ...first, chain: [first.chain[2], first.chain[0], first.chain[1]] };
  const q = planSplit({ ...joined, prev: shuffled });
  eq(q.chain.map((c) => c.id), ["c", "a", "b", "d"], "previous chain order kept, newcomer last");
});

Deno.test("plan: worked 27B example (docs/protocol.md)", () => {
  // Qwen 3.8 27B Q4_0: 64 layers of 223970464 B, embed+head+MTP 2023303168 B
  const w = (hostGiB, workers) => ({ ...DIMS.q27, host: { id: "h", name: "host", pledgeBytes: hostGiB * GiB }, workers });
  const desk = { id: "w", name: "worker-e2e", pledgeBytes: 1 * GiB, webgpu: true };
  const phone = { id: "p", name: "phone-e2e", pledgeBytes: 0.5 * GiB, webgpu: true };
  const p0 = planSplit(w(16, [phone, desk]));
  eq(p0.ranges, { h: [0, 58], p: [58, 60], w: [60, 64] }, "16 + 1 + 0.5 GiB");
  eq(p0.chain.map((c) => c.id), ["p", "w"]);
  eq(describeSplit(p0, (id) => ({ w: "worker-e2e", p: "phone-e2e" })[id], "Qwen 3.8 27B"), "Qwen 3.8 27B — layer split by pledge×speed: you 58+embed · phone-e2e 2 · worker-e2e 4");
  // the desktop leaves: the phone cannot take its 4 layers (maxLayers 2), so the host grows
  const p1 = planSplit({ ...w(16, [phone]), prev: p0 });
  eq(p1.ranges, { h: [0, 62], p: [62, 64] }, "after the desktop left");
  eq(p1.pinned, false);
  eq(planNote(p0, p1, (id) => ({ h: "you", p: "phone-e2e" })[id]), "re-dealing: you takes layers 0–61");
  // with a 14 GiB host the room no longer holds the model without the desktop
  const q0 = planSplit(w(14, [phone, desk]));
  eq(q0.ranges, { h: [0, 58], p: [58, 60], w: [60, 64] }, "14 + 1 + 0.5 GiB");
  const q1 = planSplit({ ...w(14, [phone]), prev: q0 });
  eq(q1.fits, false);
  eq(q1.held, 60);
  eq(q1.L, 64);
});

Deno.test("plan: more workers than layers -> idle, host may hold 0", () => {
  const input = { L: 3, layerBytes: GiB, embedBytes: 0, host: { id: "h", pledgeBytes: 8 * GiB },
    workers: ["a", "b", "c", "d", "e"].map((id) => ({ id, name: id, pledgeBytes: 4 * GiB, webgpu: true })) };
  const p = planSplit(input);
  checkInvariants(p, input);
  eq(p.idle.length, 2);
  eq(p.chain.length, 3);
  const s = describeSplit(p, (id) => id, "x");
  assert(/layer split/.test(s) && /standing by/.test(s), s);
});
