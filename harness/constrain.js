// Lazy tool-call constraint: decode freely, but inside a tool call force the function name and
// the parameter names to be ones the tools declare (llama.cpp's lazy grammars / XGrammar's
// trigger mode, reduced to what a coding harness needs). Values stay free text. Works with the
// XML format (<function=NAME> / <parameter=NAME>) and the JSON format ("name": "NAME").
//
// Usage per generated token: const m = C.allowed(); if (m) mask logits outside m (or call
// C.mask(logits)); sample; C.push(tokenText). With speculative decoding, apply the same check to
// each position's logits in the sample callback, so accepted tokens always satisfy the mask.
//
// tokenText(id) must give the token's decoded text (byte-level BPE: the decoded string of that
// single token). The mask for a state is computed once over the vocabulary and cached, so the
// cost is one vocabulary scan per distinct (state, typed-so-far) pair, then O(1).

export class ToolCallConstraint {
  constructor(tools, { vocabSize, tokenText, style = "xml" }) {
    this.fns = new Map(tools.map((t) => [t.name, Object.keys(t.parameters?.properties || {})]));
    this.vocabSize = vocabSize; this.tokenText = tokenText; this.style = style;
    this.text = "";          // everything generated so far (only the tail matters)
    this.cache = new Map();
  }
  // Where we are: null = free text; else { targets: [strings the slot may be], typed }.
  _slot() {
    const t = this.text;
    const call = t.lastIndexOf("<tool_call>");
    if (call < 0 || t.indexOf("</tool_call>", call) >= 0) return null;
    const body = t.slice(call);
    if (this.style === "xml") {
      const f = /<function=([^>\n]*)$/.exec(body);
      if (f) return { targets: [...this.fns.keys()].map((n) => n + ">"), typed: f[1] };
      const fn = /<function=([^>\n]+)>/.exec(body)?.[1];
      const p = /<parameter=([^>\n]*)$/.exec(body);
      if (p && fn && this.fns.has(fn)) return { targets: this.fns.get(fn).map((n) => n + ">"), typed: p[1] };
      return null;
    }
    const j = /"name"\s*:\s*"([^"]*)$/.exec(body);
    if (j) return { targets: [...this.fns.keys()].map((n) => n + "\""), typed: j[1] };
    return null;
  }
  // Uint8Array(vocabSize) of allowed tokens, or null when anything goes.
  allowed() {
    const s = this._slot();
    if (!s) return null;
    const key = s.targets.join("\u0000") + "\u0001" + s.typed;
    let m = this.cache.get(key);
    if (m) return m;
    m = new Uint8Array(this.vocabSize);
    const rest = s.targets.filter((x) => x.startsWith(s.typed)).map((x) => x.slice(s.typed.length));
    for (let id = 0; id < this.vocabSize; id++) {
      const w = this.tokenText(id);
      if (!w) continue;
      // the token either stays inside a target, or finishes one (anything after the closing
      // character is the next slot's business: a newline, the next tag)
      for (const r of rest) if (r.startsWith(w) || w.startsWith(r)) { m[id] = 1; break; }
    }
    if (!m.some(Boolean)) return null;   // no token fits (odd tokenizer): don't wedge the decoder
    this.cache.set(key, m);
    return m;
  }
  mask(logits) {
    const m = this.allowed();
    if (m) for (let i = 0; i < logits.length; i++) if (!m[i]) logits[i] = -Infinity;
    return logits;
  }
  push(tokenText) { this.text += tokenText; if (this.text.length > 8192) this.text = this.text.slice(-4096); }
}
