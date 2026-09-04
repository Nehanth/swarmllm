// Byte-level BPE tokenizer built from a tokenizer.json (Qwen/Llama family).

export function makeTokenizer(tj) {
  // special tokens (<|im_start|>, <think>, ...) live in added_tokens for
  // Qwen-family tokenizer.json files, not in model.vocab
  const vocab = { ...tj.model.vocab };
  for (const t of tj.added_tokens || []) if (t && t.content !== undefined) vocab[t.content] = t.id;
  const idToTok = {};
  for (const [t, i] of Object.entries(vocab)) idToTok[i] = t;
  const ranks = new Map();
  tj.model.merges.forEach((m, i) => ranks.set(Array.isArray(m) ? m.join(" ") : m, i));
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const byteToChar = {}, charToByte = {};
  bs.forEach((b, i) => { byteToChar[b] = String.fromCharCode(cs[i]); charToByte[String.fromCharCode(cs[i])] = b; });
  const pat = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
  const enc = new TextEncoder(), dec = new TextDecoder();
  function bpe(word) {
    let parts = [...word];
    while (parts.length > 1) {
      let best = null, bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = ranks.get(parts[i] + " " + parts[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best === null) break;
      parts = [...parts.slice(0, best), parts[best] + parts[best + 1], ...parts.slice(best + 2)];
    }
    return parts;
  }
  return {
    vocab,
    encode(text) {
      const ids = [];
      for (const piece of text.match(pat) || []) {
        let word = "";
        for (const b of enc.encode(piece)) word += byteToChar[b];
        for (const tok of bpe(word)) ids.push(vocab[tok]);
      }
      return ids;
    },
    decode(ids) {
      const bytes = [];
      for (const id of ids) {
        const tok = idToTok[id];
        if (tok === undefined) continue;
        for (const ch of tok) { const b = charToByte[ch]; if (b !== undefined) bytes.push(b); }
      }
      return dec.decode(new Uint8Array(bytes));
    },
  };
}

// ---------- sharded weight fetch ----------
