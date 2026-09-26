// Top-k / temperature sampling over a logits vector (host CPU).

export function aiSample(logits, temp = 0.8, topk = 40) {
  // single-pass top-k selection: sorting all 248k logit indices cost tens of
  // milliseconds per token; this is O(n) with a tiny candidate table.
  const idx = new Int32Array(topk), val = new Float32Array(topk).fill(-Infinity);
  let min = -Infinity, minAt = 0;
  for (let i = 0; i < logits.length; i++) {
    const v = logits[i];
    if (v > min) {
      idx[minAt] = i; val[minAt] = v;
      min = val[0]; minAt = 0;
      for (let j = 1; j < topk; j++) if (val[j] < min) { min = val[j]; minAt = j; }
    }
  }
  const order = [...idx.keys()].sort((a, b) => val[b] - val[a]);
  const mx = val[order[0]];
  const ps = order.map((j) => Math.exp((val[j] - mx) / temp));
  const sum = ps.reduce((a, b) => a + b, 0);
  let r = Math.random() * sum;
  for (let i = 0; i < order.length; i++) { r -= ps[i]; if (r <= 0) return idx[order[i]]; }
  return idx[order[0]];
}

// Sampling presets the host picks for the room. "exact" is greedy: the same question gives the
// same answer on any room shape, which is also how the bit-exactness claims are checked.
export const SAMPLING = {
  creative: { label: "creative (t 0.8)", temp: 0.8, topk: 40 },
  focused: { label: "focused (t 0.4)", temp: 0.4, topk: 20 },
  exact: { label: "exact (greedy)", temp: 0, topk: 1 },
};

export function greedy(logits) {
  let best = 0, bv = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
  return best;
}

// logits -> token id for a preset key (unknown keys fall back to creative)
export function pickSampler(key) {
  const p = SAMPLING[key] || SAMPLING.creative;
  return p.temp === 0 ? greedy : (logits) => aiSample(logits, p.temp, p.topk);
}
