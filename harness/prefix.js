// Which saved checkpoint to resume from. A hybrid model's state (DeltaNet) cannot be cut back to
// an arbitrary earlier position, so reuse works at checkpoints: the harness saves the state after
// the system prompt + tool list and after every finished assistant turn, and a new request resumes
// from the longest checkpoint whose tokens are a prefix of the new prompt, prefilling only the rest
// (SGLang / vLLM do the same for hybrid models). DOM-free.
export class PrefixIndex {
  constructor(limit = 64) { this.items = []; this.limit = limit; this.clock = 0; }   // [{ ids, key, t }], t = use order
  add(ids, key) {
    this.items = this.items.filter((x) => x.key !== key);
    this.items.push({ ids: Array.from(ids), key, t: ++this.clock });
    if (this.items.length > this.limit) { this.items.sort((a, b) => b.t - a.t); this.items.length = this.limit; }
  }
  remove(key) { this.items = this.items.filter((x) => x.key !== key); }
  // Longest checkpoint that is a strict prefix of ids (at least one token must be left to run,
  // since the last prompt token has to go through the head). -> { key, n } or null
  best(ids) {
    let best = null;
    for (const x of this.items) {
      const n = x.ids.length;
      if (n >= ids.length || (best && n <= best.n)) continue;
      let ok = true;
      for (let i = 0; i < n; i++) if (x.ids[i] !== ids[i]) { ok = false; break; }
      if (ok) best = { key: x.key, n };
    }
    if (best) this.items.find((x) => x.key === best.key).t = ++this.clock;
    return best;
  }
}
