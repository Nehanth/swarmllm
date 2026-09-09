// Layer plan: which device runs which contiguous layer range. DOM-free so it can be unit
// tested (room.js touches the document at import). The algorithm is documented verbatim in
// docs/protocol.md § "Layer plan"; keep the two in step.
//
// planSplit({ L, layerBytes, embedBytes, host, workers, prev, exclude })
//   host:    { id, name?, pledgeBytes, ms? }
//   workers: [{ id, name, pledgeBytes, webgpu, ms? }] in any order
//   prev:    the previous planSplit result (or { chain: [ids], ranges: {id: [lo, hi]}, hostRange }), or null
//   exclude: Set of ids that must not be dealt layers (failed to load)
// -> { fits: true, needBytes, haveBytes, hostId, hostRange, chain: [{id, name, range}], idle: [ids],
//      ranges: {id: [lo, hi]}, assigned: [host, ...chain], weights: {id: {cap, s, w, maxLayers}}, pinned }
// -> { fits: false, needBytes, haveBytes, held, L } when the room cannot hold the model.

const median = (xs) => { const s = [...xs].sort((a, b) => a - b); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

// floor(n * w_i / Σw) each, leftover layers by largest fractional part, ties by index
function dealBy(weights, n) {
  const total = weights.reduce((a, b) => a + b, 0);
  const assigned = weights.map((w) => total > 0 ? Math.floor(n * w / total) : 0);
  const fracs = weights.map((w, i) => ({ i, f: total > 0 ? n * w / total - assigned[i] : 0 })).sort((a, b) => b.f - a.f);
  const rem = n - assigned.reduce((a, b) => a + b, 0);
  for (let k = 0; k < rem; k++) assigned[fracs[k % fracs.length].i]++;
  return assigned;
}

export function planSplit({ L, layerBytes, embedBytes, host, workers, prev = null, exclude = new Set() }) {
  // a. eligible workers, in a stable order
  const byId = new Map(workers.map((w) => [w.id, w]));
  const eligible = workers.filter((w) => w.webgpu !== false && w.pledgeBytes > 0 && !exclude.has(w.id)).map((w) => w.id);
  const elig = new Set(eligible);
  let order;
  if (!prev) order = [...eligible].sort();
  else {
    // prev.chain: ids, or the {id, name, range} entries of a previous planSplit result
    const kept = prev.chain.map((c) => typeof c === "string" ? c : c.id).filter((id) => elig.has(id));
    const keptSet = new Set(kept);
    order = [...kept, ...eligible.filter((id) => !keptSet.has(id)).sort()];
  }

  // b. memory caps and the layers each device can hold
  const capHost = Math.max(host.pledgeBytes - embedBytes, layerBytes / 2);
  const devs = [{ id: host.id, name: host.name, cap: capHost, ms: host.ms, maxLayers: Math.floor(capHost / layerBytes), host: true }];
  for (const id of order) {
    const w = byId.get(id);
    const cap = Math.max(w.pledgeBytes, layerBytes / 2);
    devs.push({ id, name: w.name, cap, ms: w.ms, maxLayers: Math.max(1, Math.floor(cap / layerBytes)), host: false });
  }
  const needBytes = L * layerBytes + embedBytes;
  const haveBytes = devs.reduce((s, d) => s + d.cap, embedBytes);

  // c. does it fit at all?
  const held = devs.reduce((s, d) => s + d.maxLayers, 0);
  if (held < L) return { fits: false, needBytes, haveBytes, held, L };

  // d. speed factor: quarter steps around the median so autotune noise cannot flip a layer,
  //    clamped so one odd sample cannot starve a device; 0 or 1 measured devices -> all 1
  const known = devs.filter((d) => d.ms > 0).map((d) => d.ms);
  const msRef = known.length >= 2 ? median(known) : null;
  for (const d of devs) d.s = msRef && d.ms > 0 ? Math.min(2, Math.max(0.5, Math.round(4 * msRef / d.ms) / 4)) : 1;
  for (const d of devs) d.w = d.cap * d.s;

  // g (part): more workers than layers: the lowest-weight ones stand by
  const idle = [];
  if (devs.length - 1 > L) {
    const cand = devs.slice(1).sort((a, b) => a.w - b.w || (a.id < b.id ? 1 : -1));
    for (const d of cand.slice(0, devs.length - 1 - L)) idle.push(d.id);
    const idleSet = new Set(idle);
    for (let i = devs.length - 1; i >= 1; i--) if (idleSet.has(devs[i].id)) devs.splice(i, 1);
  }
  const nW = devs.length - 1;

  // e/f. deal: host pinned to its previous count when that still works, else proportional
  let assigned, pinned = false;
  const h0 = prev?.hostRange ? prev.hostRange[1] - prev.hostRange[0] : -1;
  const workerHeld = devs.slice(1).reduce((s, d) => s + d.maxLayers, 0);
  if (h0 >= 0 && h0 <= devs[0].maxLayers && workerHeld >= L - h0 && L - h0 >= nW) {
    pinned = true;
    assigned = [h0, ...dealBy(devs.slice(1).map((d) => d.w), L - h0)];
  } else {
    assigned = dealBy(devs.map((d) => d.w), L);
  }

  // g. repair: nobody above its cap (overflow goes to the largest headroom, ties lowest index;
  //    a pinned host does not absorb), then every worker holds at least one layer
  for (;;) {
    const over = assigned.findIndex((a, i) => a > devs[i].maxLayers);
    if (over < 0) break;
    let to = -1, best = 0;
    for (let i = 0; i < devs.length; i++) {
      if (pinned && i === 0) continue;
      const room = devs[i].maxLayers - assigned[i];
      if (room > best) { best = room; to = i; }
    }
    if (to < 0) break;   // cannot happen when held >= L
    assigned[over]--; assigned[to]++;
  }
  for (let i = 1; i < assigned.length; i++)
    if (assigned[i] === 0) { const j = assigned.indexOf(Math.max(...assigned)); assigned[j]--; assigned[i]++; }

  // h. contiguous ranges in order, host first
  const ranges = {};
  const chain = [];
  let acc = 0;
  devs.forEach((d, i) => {
    const r = [acc, acc + assigned[i]]; acc += assigned[i];
    ranges[d.id] = r;
    if (i > 0) chain.push({ id: d.id, name: d.name, range: r });
  });
  const weights = Object.fromEntries(devs.map((d) => [d.id, { cap: d.cap, s: d.s, w: d.w, maxLayers: d.maxLayers }]));
  return { fits: true, needBytes, haveBytes, hostId: host.id, hostRange: ranges[host.id], chain, idle, ranges, assigned, weights, pinned };
}

// "Qwen3 0.6B — layer split by pledge×speed: you 16+embed · worker-e2e 8 · phone-e2e 4"
export function describeSplit(plan, nameOf, label) {
  const parts = [`you ${plan.assigned[0]}+embed`, ...plan.chain.map((c) => `${nameOf(c.id)} ${c.range[1] - c.range[0]}`)];
  const idle = plan.idle.length ? ` (standing by: ${plan.idle.map(nameOf).join(", ")})` : "";
  return `${label} — layer split by pledge×speed: ${parts.join(" · ")}${idle}`;
}

// "re-dealing: phone-e2e takes layers 40–63 · worker-e2e gives up 8"; "" when nothing moved
export function planNote(prevPlan, plan, nameOf) {
  const notes = [];
  const count = (r) => r ? r[1] - r[0] : 0;
  const ids = [plan.hostId, ...plan.chain.map((c) => c.id)];
  for (const id of ids) {
    const now = plan.ranges[id], was = prevPlan?.ranges?.[id];
    const gained = count(now) - count(was);
    if ((!was && count(now) > 0) || gained > 0) notes.push(`${nameOf(id)} takes layers ${now[0]}–${now[1] - 1}`);
    else if (gained < 0) notes.push(`${nameOf(id)} gives up ${-gained}`);
  }
  for (const id of plan.idle) if (count(prevPlan?.ranges?.[id]) > 0) notes.push(`${nameOf(id)} stands by`);
  return notes.length ? "re-dealing: " + notes.join(" · ") : "";
}
