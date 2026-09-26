// Tiny synthetic Qwen3-architecture DENSE model for end-to-end runs without the real weights:
// a GGUF laid out like Qwen/Qwen3-*-GGUF's Q8_0 files (general.architecture = "qwen3", Q8_0
// matrices incl. token_embd, f32 norms, per-head q/k norms, tied output like the real 0.6B/1.7B/4B
// or --untied for a separate output.weight), plus the config.json and tokenizer.json the room
// fetches for the dense models (room/models.js M.cfg / M.tok; engine/dense.js reads cfg, the
// tokenizer comes from tokenizer.json, not from the GGUF).
//
//   node tests/e2e/synth_dense.mjs [outdir] [--seed 1] [--layers 8] [--dim 256] [--untied]
//        writes outdir/qwen3-synth.gguf, outdir/config.json, outdir/tokenizer.json
//
// Kernel constraints respected (engine/dense.js, engine/wgsl/base.js, coop.js): every matrix dIn
// % 32 === 0 (Q8_0 blocks, vec4 loads), dim/qDim/inter multiples of 32, nH % nKV === 0,
// head_dim even (rope pairs i, i+half). qDim != dim on purpose (the real Qwen3 has qDim = 2*dim
// for 0.6B), so a q/o shape mix-up shows.
//
// Output text: special tokens and non-printable byte tokens only have weight in the "silent"
// residual dims [0, NS), which output_norm zeroes, so their logits are exactly 0 while the
// printable rows have random logits well above 0: greedy answers are long, readable nonsense and
// never end on <|im_end|> (the room's max-new cap ends them). Layers are strong enough
// (--layer-scale) that attention, i.e. positions and the KV cache, changes the argmax.
import fs from "fs";
import path from "path";

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const _f = new Float32Array(1), _u = new Uint32Array(_f.buffer);
function f32ToF16(v) {
  _f[0] = v;
  const x = _u[0];
  const sign = (x >>> 16) & 0x8000;
  let e = (x >>> 23) & 0xff, m = x & 0x7fffff;
  if (e === 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
  e = e - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) { if (e < -10) return sign; m = (m | 0x800000) >> (1 - e); return sign | ((m + 0x1000) >> 13); }
  return sign | ((e << 10) + ((m + 0x1000) >> 13));
}
function quantQ8_0(x) {   // quantize_row_q8_0_ref
  const nb = x.length / 32, out = new Uint8Array(nb * 34);
  for (let b = 0; b < nb; b++) {
    let amax = 0;
    for (let j = 0; j < 32; j++) amax = Math.max(amax, Math.abs(x[b * 32 + j]));
    const d = amax / 127, id = d ? 1 / d : 0;
    const h = f32ToF16(d);
    out[b * 34] = h & 0xff; out[b * 34 + 1] = h >> 8;
    for (let j = 0; j < 32; j++) out[b * 34 + 2 + j] = Math.round(x[b * 32 + j] * id) & 0xff;
  }
  return out;
}

// ---- byte-level BPE vocabulary (same scheme as engine/tokenizer.js) ----
function byteChars() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const m = new Array(256);
  bs.forEach((b, i) => { m[b] = String.fromCharCode(cs[i]); });
  return m;
}
const CORPUS = `The ocean is deep and the sea is wide. Write three sentences about the ocean.
Hello, how are you today? I am fine, thank you. What is the capital of France? The capital of France is Paris.
Write the Python code for two sum. def two_sum(nums, target): return the indices of the two numbers.
The quick brown fox jumps over the lazy dog. A swarm of devices runs one model in the browser.
user assistant system the and that this with from there their then than when where which while
Tell me a story about a robot who learns to paint. Explain how a transformer works in simple words.
1 2 3 4 5 6 7 8 9 10 100 2024 2025 2026 the the the of of to to in in is is it it on on`;
function trainMerges(nMerges) {
  const B = byteChars(), enc = new TextEncoder();
  const pat = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
  const words = new Map();
  for (const piece of CORPUS.match(pat)) {
    const k = [...enc.encode(piece)].map((b) => B[b]).join("\u0001");
    words.set(k, (words.get(k) || 0) + 1);
  }
  let W = [...words].map(([k, c]) => ({ s: k.split("\u0001"), c }));
  const merges = [];
  while (merges.length < nMerges) {
    const pairs = new Map();
    for (const { s, c } of W) for (let i = 0; i < s.length - 1; i++) { const p = s[i] + " " + s[i + 1]; pairs.set(p, (pairs.get(p) || 0) + c); }
    if (!pairs.size) break;
    const best = [...pairs].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))[0][0];
    merges.push(best);
    const [a, b] = best.split(" ");
    W = W.map(({ s, c }) => { const o = []; for (let i = 0; i < s.length; i++) { if (i < s.length - 1 && s[i] === a && s[i + 1] === b) { o.push(a + b); i++; } else o.push(s[i]); } return { s: o, c }; });
  }
  return { byteTokens: [...B], merges };
}
export const SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"];

// ---- GGUF writer ----
const GT = { U32: 4, I32: 5, F32: 6, BOOL: 7, STR: 8, ARR: 9 };
const GGML = { F32: 0, Q8_0: 8 };
class Wr {
  constructor() { this.buf = new Uint8Array(1 << 20); this.n = 0; }
  grow(k) { if (this.n + k > this.buf.length) { const b = new Uint8Array(Math.max(this.buf.length * 2, this.n + k)); b.set(this.buf.subarray(0, this.n)); this.buf = b; } }
  dv() { return new DataView(this.buf.buffer); }
  u32(v) { this.grow(4); this.dv().setUint32(this.n, v, true); this.n += 4; }
  i32(v) { this.grow(4); this.dv().setInt32(this.n, v, true); this.n += 4; }
  u64(v) { this.grow(8); this.dv().setBigUint64(this.n, BigInt(v), true); this.n += 8; }
  f32(v) { this.grow(4); this.dv().setFloat32(this.n, v, true); this.n += 4; }
  u8(v) { this.grow(1); this.buf[this.n++] = v; }
  bytes(b) { this.grow(b.length); this.buf.set(b, this.n); this.n += b.length; }
  str(s) { const b = new TextEncoder().encode(s); this.u64(b.length); this.bytes(b); }
  pad(a) { while (this.n % a) this.u8(0); }
  out() { return this.buf.slice(0, this.n); }
}
function writeKV(w, key, type, val) {
  w.str(key); w.u32(type);
  const one = (t, v) => { if (t === GT.U32) w.u32(v); else if (t === GT.I32) w.i32(v); else if (t === GT.F32) w.f32(v); else if (t === GT.BOOL) w.u8(v ? 1 : 0); else if (t === GT.STR) w.str(v); };
  if (type === GT.ARR) { const [et, arr] = val; w.u32(et); w.u64(arr.length); for (const v of arr) one(et, v); } else one(type, val);
}

export const DEFAULTS = { seed: 1, layers: 8, dim: 256, inter: 512, nH: 8, nKV: 4, hd: 64, merges: 123, layerScale: 0.35, headGain: 3, ropeTheta: 1e6, eps: 1e-6, ctx: 40960, untied: false, padVocab: 64 };
const NS = 8;   // "silent" residual dims: output_norm is 0 there

export function buildDense(opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const rnd = mulberry32(o.seed * 2654435761 + 29);
  const U = (a) => (rnd() * 2 - 1) * a;
  const { dim, inter, nH, nKV, hd } = o;
  const qDim = nH * hd, kvDim = nKV * hd;
  for (const d of [dim, inter, qDim, kvDim]) if (d % 32) throw new Error("dims must be multiples of 32");
  if (nH % nKV) throw new Error("nH % nKV");
  const { byteTokens, merges } = trainMerges(o.merges);
  const tokens = [...byteTokens, ...merges.map((m) => m.replace(" ", "")), ...SPECIALS];
  const nTok = tokens.length;
  const vocab = Math.ceil((nTok + 1) / o.padVocab) * o.padVocab;   // padded like the real 151936 > 151669
  const id = Object.fromEntries(tokens.map((t, i) => [t, i]));
  const printable = (t) => [...t].every((ch) => { const c = ch.charCodeAt(0); return (c >= 33 && c <= 126) || c === 0x120 || c === 0x10A; });
  const silent = (r) => r >= nTok || r >= byteTokens.length + merges.length || !printable(tokens[r]);

  const tensors = [];
  const mat = (name, rows, cols, gain, fill) => {
    const a = gain * Math.sqrt(3 / cols);
    const x = new Float32Array(rows * cols);
    for (let i = 0; i < x.length; i++) x[i] = U(a);
    if (fill) fill(x);
    tensors.push({ name, shape: [rows, cols], type: GGML.Q8_0, data: quantQ8_0(x), f32: x });
  };
  const vec = (name, n, f) => {
    const x = new Float32Array(n);
    for (let i = 0; i < n; i++) x[i] = f(i);
    tensors.push({ name, shape: [n], type: GGML.F32, data: new Uint8Array(x.buffer), f32: x });
  };
  const norm = () => 1 + U(0.15);
  // embeddings rms ~1; silent rows live in dims [0, NS) only (their logit through the tied head is 0)
  const embFill = (x) => {
    for (let r = 0; r < vocab; r++) for (let c = 0; c < dim; c++) {
      const s = silent(r);
      x[r * dim + c] = s ? (c < NS ? U(Math.sqrt(3) * 4) : 0) : (c < NS ? 0 : U(Math.sqrt(3)));
    }
  };
  mat("token_embd.weight", vocab, dim, 1, embFill);
  // random signs: with a tied head the current token's own row would otherwise always win (its
  // embedding is the bulk of the residual stream), and greedy answers would repeat one token
  vec("output_norm.weight", dim, (i) => (i < NS ? 0 : o.headGain * norm() * (rnd() < 0.5 ? -1 : 1)));
  if (o.untied) mat("output.weight", vocab, dim, 1, (x) => { for (let r = 0; r < vocab; r++) if (silent(r)) x.fill(0, r * dim, (r + 1) * dim); });
  const ls = o.layerScale;
  for (let i = 0; i < o.layers; i++) {
    const p = `blk.${i}.`;
    // file order like llama.cpp's qwen3 GGUFs
    vec(p + "attn_k_norm.weight", hd, norm);
    mat(p + "attn_k.weight", kvDim, dim, 1);
    vec(p + "attn_norm.weight", dim, norm);
    mat(p + "attn_output.weight", dim, qDim, ls * 2);
    vec(p + "attn_q_norm.weight", hd, norm);
    mat(p + "attn_q.weight", qDim, dim, 1.5);
    mat(p + "attn_v.weight", kvDim, dim, 1);
    mat(p + "ffn_down.weight", dim, inter, ls * 2);
    mat(p + "ffn_gate.weight", inter, dim, 1);
    vec(p + "ffn_norm.weight", dim, norm);
    mat(p + "ffn_up.weight", inter, dim, 1);
  }
  const w = new Wr();
  const kv = [
    ["general.architecture", GT.STR, "qwen3"],
    ["general.name", GT.STR, "SwarmLLM synthetic qwen3 dense (test only)"],
    ["general.alignment", GT.U32, 32],
    ["general.file_type", GT.U32, 7],
    ["qwen3.block_count", GT.U32, o.layers],
    ["qwen3.context_length", GT.U32, o.ctx],
    ["qwen3.embedding_length", GT.U32, dim],
    ["qwen3.feed_forward_length", GT.U32, inter],
    ["qwen3.attention.head_count", GT.U32, nH],
    ["qwen3.attention.head_count_kv", GT.U32, nKV],
    ["qwen3.attention.key_length", GT.U32, hd],
    ["qwen3.attention.value_length", GT.U32, hd],
    ["qwen3.attention.layer_norm_rms_epsilon", GT.F32, o.eps],
    ["qwen3.rope.freq_base", GT.F32, o.ropeTheta],
    ["tokenizer.ggml.model", GT.STR, "gpt2"],
    ["tokenizer.ggml.pre", GT.STR, "qwen2"],
    ["tokenizer.ggml.tokens", GT.ARR, [GT.STR, tokens]],
    ["tokenizer.ggml.merges", GT.ARR, [GT.STR, merges]],
    ["tokenizer.ggml.eos_token_id", GT.U32, id["<|im_end|>"]],
    ["tokenizer.ggml.add_bos_token", GT.BOOL, false],
  ];
  w.u32(0x46554747); w.u32(3); w.u64(tensors.length); w.u64(kv.length);
  for (const [k, t, v] of kv) writeKV(w, k, t, v);
  let off = 0;
  for (const t of tensors) {
    w.str(t.name); w.u32(t.shape.length);
    for (const d of t.shape.slice().reverse()) w.u64(d);
    w.u32(t.type); w.u64(off);
    off += Math.ceil(t.data.length / 32) * 32;
  }
  w.pad(32);
  for (const t of tensors) { w.bytes(t.data); w.pad(32); }

  const config = {
    architectures: ["Qwen3ForCausalLM"], model_type: "qwen3", hidden_size: dim, intermediate_size: inter,
    num_hidden_layers: o.layers, num_attention_heads: nH, num_key_value_heads: nKV, head_dim: hd,
    rms_norm_eps: o.eps, rope_theta: o.ropeTheta, vocab_size: vocab, tie_word_embeddings: !o.untied,
    max_position_embeddings: o.ctx, hidden_act: "silu", torch_dtype: "bfloat16",
    bos_token_id: id["<|endoftext|>"], eos_token_id: id["<|im_end|>"],
  };
  const vocabMap = {};
  for (let i = 0; i < byteTokens.length + merges.length; i++) vocabMap[tokens[i]] = i;
  const tokenizer = {
    version: "1.0", truncation: null, padding: null,
    added_tokens: SPECIALS.map((s) => ({ id: id[s], content: s, single_word: false, lstrip: false, rstrip: false, normalized: false, special: !s.includes("think") })),
    normalizer: { type: "NFC" },
    pre_tokenizer: { type: "Sequence", pretokenizers: [] },
    post_processor: null,
    decoder: { type: "ByteLevel", add_prefix_space: false, trim_offsets: false, use_regex: false },
    model: { type: "BPE", dropout: null, unk_token: null, continuing_subword_prefix: "", end_of_word_suffix: "", fuse_unk: false, byte_fallback: false, ignore_merges: false,
      vocab: vocabMap, merges: merges.map((m) => m.split(" ")) },   // new-style [a, b] pairs like current Qwen3 tokenizer.json
  };
  return { gguf: w.out(), config, tokenizer, tensors, info: { vocab, nTok, dim, layers: o.layers, qDim, kvDim, ids: id } };
}

// The same kind of model as a SmolLM2-style (Llama architecture, kind "safetensors") checkpoint:
// BF16 safetensors with HF names, no q/k norms, tied embeddings, head_dim = dim / nH.
const HF = { "token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight" };
const HFL = { attn_norm: "input_layernorm", attn_q: "self_attn.q_proj", attn_k: "self_attn.k_proj", attn_v: "self_attn.v_proj",
  attn_output: "self_attn.o_proj", ffn_norm: "post_attention_layernorm", ffn_gate: "mlp.gate_proj", ffn_up: "mlp.up_proj", ffn_down: "mlp.down_proj" };
export function writeSmol(dir, opts = {}) {
  const r = buildDense({ nH: 4, nKV: 2, hd: 64, ...opts, untied: false });
  const header = {}, parts = [];
  let off = 0;
  for (const t of r.tensors) {
    let name = HF[t.name];
    const m = /^blk\.(\d+)\.(\w+)\.weight$/.exec(t.name);
    if (m) { if (!HFL[m[2]]) continue; name = `model.layers.${m[1]}.${HFL[m[2]]}.weight`; }
    const u32 = new Uint32Array(t.f32.buffer, t.f32.byteOffset, t.f32.length);
    const bf = new Uint16Array(t.f32.length);
    for (let i = 0; i < bf.length; i++) bf[i] = (u32[i] + 0x7fff + ((u32[i] >>> 16) & 1)) >>> 16;   // round to nearest even
    header[name] = { dtype: "BF16", shape: t.shape, data_offsets: [off, off + bf.byteLength] };
    parts.push(new Uint8Array(bf.buffer)); off += bf.byteLength;
  }
  let hj = new TextEncoder().encode(JSON.stringify(header));
  const padN = (8 - (hj.length % 8)) % 8;
  hj = new TextEncoder().encode(JSON.stringify(header) + " ".repeat(padN));
  const out = new Uint8Array(8 + hj.length + off);
  new DataView(out.buffer).setBigUint64(0, BigInt(hj.length), true);
  out.set(hj, 8);
  let o = 8 + hj.length;
  for (const p of parts) { out.set(p, o); o += p.length; }
  const config = { ...r.config, architectures: ["LlamaForCausalLM"], model_type: "llama", tie_word_embeddings: true };
  delete config.head_dim;
  fs.mkdirSync(dir, { recursive: true });
  const f = { st: path.join(dir, "model.safetensors"), cfg: path.join(dir, "config.json"), tok: path.join(dir, "tokenizer.json") };
  fs.writeFileSync(f.st, out);
  fs.writeFileSync(f.cfg, JSON.stringify(config, null, 1));
  fs.writeFileSync(f.tok, JSON.stringify(r.tokenizer));
  return { files: f, size: out.length, ...r.info };
}

export function writeDense(dir, opts = {}) {
  const r = buildDense(opts);
  fs.mkdirSync(dir, { recursive: true });
  const f = { gguf: path.join(dir, "qwen3-synth.gguf"), cfg: path.join(dir, "config.json"), tok: path.join(dir, "tokenizer.json") };
  fs.writeFileSync(f.gguf, r.gguf);
  fs.writeFileSync(f.cfg, JSON.stringify(r.config, null, 1));
  fs.writeFileSync(f.tok, JSON.stringify(r.tokenizer));
  return { files: f, size: r.gguf.length, ...r.info };
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname)) {
  const argv = process.argv.slice(2);
  const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
  const out = argv[0] && !argv[0].startsWith("--") ? argv[0] : path.resolve(new URL(".", import.meta.url).pathname, "../../models/synth/dense");
  const r = writeDense(out, { ...(arg("shape") ? JSON.parse(arg("shape")) : {}), seed: +arg("seed", 1), layers: +arg("layers", DEFAULTS.layers), ...(arg("dim") ? { dim: +arg("dim") } : {}), untied: argv.includes("--untied"), layerScale: +arg("layer-scale", DEFAULTS.layerScale) });
  console.log(JSON.stringify({ ...r, ids: undefined }, null, 1));
}
