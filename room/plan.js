// Layer placement and the model ladder. DOM-free so it can be unit tested.

// Deal L layers over devices in proportion to what each can hold. caps: bytes each device offers
// for layers (host first; the host's cap already has the embedding, head and draft block taken
// out). Every device gets at least one layer; rounding leftovers go to the largest remainders.
// Returns { assigned: [count per device], ranges: [[lo, hi) per device] }.
export function planSplit(L, caps) {
  const totalCap = caps.reduce((s, c) => s + c, 0);
  const assigned = caps.map((c) => Math.floor(L * c / totalCap));
  const fracs = caps.map((c, i) => ({ i, f: L * c / totalCap - assigned[i] })).sort((a, b) => b.f - a.f);
  const rem = L - assigned.reduce((a, b) => a + b, 0);
  for (let k = 0; k < rem; k++) assigned[fracs[k % fracs.length].i]++;
  for (let i = 1; i < assigned.length; i++)
    if (assigned[i] === 0) { const j = assigned.indexOf(Math.max(...assigned)); assigned[j]--; assigned[i]++; }
  const ranges = [];
  let acc = 0;
  for (const a of assigned) { ranges.push([acc, acc + a]); acc += a; }
  return { assigned, ranges };
}

// What this room can run, smallest model first: [{ key, need, ok, short }]. short is how many
// more GB the room needs for that model (0 when it fits).
export function ladder(needGB, pledgedGB) {
  return Object.entries(needGB)
    .sort((a, b) => a[1] - b[1])
    .map(([key, need]) => ({ key, need, ok: pledgedGB >= need, short: Math.max(0, +(need - pledgedGB).toFixed(1)) }));
}

// The largest model that fits the room's pledges, or the smallest one when nothing fits yet.
export function bestFit(needGB, pledgedGB) {
  const l = ladder(needGB, pledgedGB);
  const fits = l.filter((x) => x.ok);
  return (fits.length ? fits[fits.length - 1] : l[0]).key;
}

// Room code from a join link: /r/ABCD, /room/ABCD, ?code=ABCD or #ABCD. Codes use the room's
// 30-letter alphabet (no I, L, O, U, 0, 1); anything else is ignored. Returns "" when absent.
export function codeFromLocation(pathname = "", search = "", hash = "") {
  const ok = (c) => /^[A-HJKMNP-TV-Z2-9]{4,6}$/.test(c) ? c : "";
  const m = /\/(?:r|room)\/([A-Za-z0-9]{4,6})\/?$/.exec(pathname);
  if (m) return ok(m[1].toUpperCase());
  const q = new URLSearchParams(search).get("code");
  if (q) return ok(q.trim().toUpperCase());
  const h = hash.replace(/^#/, "");
  if (h && h !== "debug") return ok(h.toUpperCase());
  return "";
}
