// Tiny synthetic Qwen 3.5/3.8-architecture GGUF for end-to-end runs on machines without the real
// weights (CI boxes, containers, SwiftShader). Same metadata keys and tensor names as the real
// file (general.architecture = "qwen35"), hybrid layout (Gated-DeltaNet layers, full attention on
// i % 4 === 3), a `blk.{N}.nextn.*` multi-token-prediction block, Q4_0 matrices, f32 norms, and a
// real byte-level BPE tokenizer (256 byte tokens, trained merges, Qwen chat specials).
//
//   node tests/e2e/synth.mjs [out.gguf] [--seed 1] [--layers 8] [--dim 256] [--mtp echo|random]
//                            [--layer-scale 0.04] [--vocab-merges 123] [--eos-at 90]
//
// Weights are seeded (same options -> byte-identical file). Default out: models/synth/qwen35-synth.gguf
// (models/ is git-ignored).
//
// --mtp echo (default): the model is made "embedding-dominated": every layer adds a small
// (layer-scale) update to the residual stream, and the draft head's eh_proj copies the token
// embedding, so the draft usually agrees with the trunk but not always. Both the accept path and
// the reject/rollback path of speculative decoding get exercised. --mtp random: random draft head
// (almost every draft rejected).
//
// --eos-at: answers end on <|im_end|> once the context reaches about eos-at * (<|im_start|> count)
// / 2 tokens: ~60 answer tokens for a first chat turn (see "The answer length" below).
//
// Kernel constraints this respects (engine/wgsl/*.js): dState = 128 (dn_gatenorm reduces over 128
// lanes), 2*nKH + nVH <= 128 (dn_pre), nVH <= 64 (gate scratch), every matrix dIn % 32 === 0
// (Q4_0 / Q8_0 blocks). The prefill GEMM only has pinned shapes for the 27B, so it stays off here.
import fs from "fs";
import path from "path";

// ---------------- tiny deterministic RNG ----------------
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

// ---------------- f16 / quantizers (ggml reference semantics) ----------------
const _f = new Float32Array(1), _u = new Uint32Array(_f.buffer);
function f32ToF16(v) {
  _f[0] = v;
  const x = _u[0];
  const sign = (x >>> 16) & 0x8000;
  let e = (x >>> 23) & 0xff, m = x & 0x7fffff;
  if (e === 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
  e = e - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    m = (m | 0x800000) >> (1 - e);
    return sign | ((m + 0x1000) >> 13);
  }
  return sign | ((e << 10) + ((m + 0x1000) >> 13));
}
// quantize_row_q4_0_ref
function quantQ4_0(x) {
  const nb = x.length / 32, out = new Uint8Array(nb * 18);
  for (let b = 0; b < nb; b++) {
    let amax = 0, max = 0;
    for (let j = 0; j < 32; j++) { const v = x[b * 32 + j]; if (Math.abs(v) > amax) { amax = Math.abs(v); max = v; } }
    const d = max / -8, id = d ? 1 / d : 0;
    const h = f32ToF16(d);
    out[b * 18] = h & 0xff; out[b * 18 + 1] = h >> 8;
    for (let j = 0; j < 16; j++) {
      const x0 = x[b * 32 + j] * id, x1 = x[b * 32 + 16 + j] * id;
      const q0 = Math.min(15, Math.trunc(x0 + 8.5)), q1 = Math.min(15, Math.trunc(x1 + 8.5));
      out[b * 18 + 2 + j] = q0 | (q1 << 4);
    }
  }
  return out;
}
// quantize_row_q8_0_ref
function quantQ8_0(x) {
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

// ---------------- byte-level BPE (GPT-2 / Qwen style) ----------------
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
  return m;   // byte -> char
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
  const words = new Map();   // word (array of symbols joined by \u0001) -> count
  for (const piece of CORPUS.match(pat)) {
    const w = [...enc.encode(piece)].map((b) => B[b]);
    const k = w.join("\u0001");
    words.set(k, (words.get(k) || 0) + 1);
  }
  let W = [...words].map(([k, c]) => ({ s: k.split("\u0001"), c }));
  const merges = [];
  while (merges.length < nMerges) {
    const pairs = new Map();
    for (const { s, c } of W) for (let i = 0; i < s.length - 1; i++) {
      const p = s[i] + " " + s[i + 1];
      pairs.set(p, (pairs.get(p) || 0) + c);
    }
    if (!pairs.size) break;
    const best = [...pairs].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))[0][0];
    merges.push(best);
    const [a, b] = best.split(" ");
    W = W.map(({ s, c }) => {
      const o = [];
      for (let i = 0; i < s.length; i++) {
        if (i < s.length - 1 && s[i] === a && s[i + 1] === b) { o.push(a + b); i++; } else o.push(s[i]);
      }
      return { s: o, c };
    });
  }
  return { byteTokens: [...B], merges };
}
export const SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"];

// ---------------- GGUF writer ----------------
const GT = { U32: 4, I32: 5, F32: 6, BOOL: 7, STR: 8, ARR: 9 };
const GGML = { F32: 0, Q4_0: 2, Q8_0: 8 };
class W {
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
  const one = (t, v) => {
    if (t === GT.U32) w.u32(v); else if (t === GT.I32) w.i32(v); else if (t === GT.F32) w.f32(v);
    else if (t === GT.BOOL) w.u8(v ? 1 : 0); else if (t === GT.STR) w.str(v);
  };
  if (type === GT.ARR) { const [et, arr] = val; w.u32(et); w.u64(arr.length); for (const v of arr) one(et, v); }
  else one(type, val);
}

// ---------------- the model ----------------
export const DEFAULTS = {
  seed: 1, layers: 8, dim: 256, inter: 512, nH: 4, nKV: 2, hd: 64, nRot: 32,
  dState: 128, nKH: 2, nVH: 2, merges: 123, mtp: "echo", layerScale: 0.04, headGain: 4,
  ropeTheta: 1e7, eps: 1e-6, ctx: 4096, eosAt: 90, counter: true,
};
// The 27B's shapes (dim 5120, FFN 17408, 24 q / 4 kv heads of 256, 16 / 48 DeltaNet heads): every
// matrix hits a pinned prefill-GEMM shape (engine/wgsl/gemm.js GEMM_S), so the GEMM path runs.
// --shape 27b; fewer layers keep it loadable on SwiftShader (4 layers + draft block ~ 1.1 GB).
export const SHAPE_27B = { dim: 5120, inter: 17408, nH: 24, nKV: 4, hd: 256, nRot: 64, nKH: 16, nVH: 48, counter: false, layers: 4, q8out: true };

// Residual-stream feature dims reserved for the answer-length counter (see buildSynthGGUF).
const F_MARK = 0, F_COUNT = 1, F_BIAS = 2, NF = 3;

// The answer length. Random weights never pick the end-of-turn token by themselves, and a fixed
// token budget would cap every answer (which hides the "answer ended on <|im_end|>" paths). So the
// model counts: residual dims 0..2 are kept free of the random weights, and
//   * every <|im_start|> embedding carries a marker in dim 0, every token a constant in dim 2;
//   * in layer 3 (the first full-attention layer), head 0 has q = 0, i.e. uniform attention over
//     the whole context, and reads the marker: its output is (markers so far) / (context length),
//     written into dim 1;
//   * the <|im_end|> row of the head is  beta * dim2 - gamma * dim1:  it rises as the context grows
//     past the markers and wins at about  context = eosAt * markers / 2  (eosAt: 90 -> a first
//     answer of ~60 tokens after a ~30-token prompt; later turns scale with their marker count).
// The draft head never sees the counter (its eh_proj drops dims 0..2), so it never drafts the end.
export function buildSynthGGUF(opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const rnd = mulberry32(o.seed * 2654435761 + 17);
  const U = (a) => (rnd() * 2 - 1) * a;                  // uniform [-a, a], std a/sqrt(3)
  const { dim, inter, nH, nKV, hd, dState, nKH, nVH } = o;
  const qDim = nH * hd, kvDim = nKV * hd, keyDim = dState * nKH, dInner = dState * nVH, convDim = 2 * keyDim + dInner;
  if (dState !== 128) throw new Error("dState must be 128 (dn_gatenorm / dn_delta workgroups are 128 wide)");
  if (2 * nKH + nVH > 128 || nVH > 64) throw new Error("too many DeltaNet heads for dn_pre");
  for (const d of [dim, inter, qDim, dInner]) if (d % 32) throw new Error("dims must be multiples of 32");
  if (o.counter && nH / nKV !== 2) throw new Error("the counter head assumes 2 query heads per kv head (or pass counter: false)");
  const L = o.layers, N = L;                              // trunk layers 0..L-1, nextn block = blk.L
  const { byteTokens, merges } = trainMerges(o.merges);
  const tokens = [...byteTokens, ...merges.map((m) => m.replace(" ", "")), ...SPECIALS];
  const vocab = tokens.length;
  const tokenType = tokens.map((_, i) => (i >= byteTokens.length + merges.length ? 3 : 1));
  const id = Object.fromEntries(tokens.map((t, i) => [t, i]));

  const tensors = [];   // { name, shape (torch order), type, data: Uint8Array }
  // matrix with uniform entries scaled so y = W x has rms `gain` for a unit-rms x.
  // role "in": reads the residual stream (feature columns zeroed); "out": writes it (feature rows zeroed)
  const mat = (name, rows, cols, gain, { type = GGML.Q4_0, fill = null, role = null, edit = null } = {}) => {
    const a = gain * Math.sqrt(3 / cols);
    const x = new Float32Array(rows * cols);
    if (fill) fill(x, a); else for (let i = 0; i < x.length; i++) x[i] = U(a);
    if (role === "in") for (let r = 0; r < rows; r++) for (let c = 0; c < NF; c++) x[r * cols + c] = 0;
    if (role === "out") x.fill(0, 0, NF * cols);
    if (edit) edit(x);
    tensors.push({ name, shape: [rows, cols], type, data: type === GGML.Q4_0 ? quantQ4_0(x) : type === GGML.Q8_0 ? quantQ8_0(x) : new Uint8Array(x.buffer) });
  };
  const vec = (name, n, f) => {
    const x = new Float32Array(n);
    for (let i = 0; i < n; i++) x[i] = f(i);
    tensors.push({ name, shape: [n], type: GGML.F32, data: new Uint8Array(x.buffer) });
  };
  const f32mat = (name, rows, cols, f) => {
    const x = new Float32Array(rows * cols);
    for (let i = 0; i < x.length; i++) x[i] = f(i);
    tensors.push({ name, shape: [rows, cols], type: GGML.F32, data: new Uint8Array(x.buffer) });
  };
  const norm = () => 1 + U(0.1);
  const normF = (i) => (i < NF ? 1 : norm());              // feature dims pass norms unscaled
  const ls = o.layerScale;

  // ---- counter constants ----
  const MARK = 16, BIAS = 3;                               // embedding values in dims 0 and 2
  const markXn = MARK / Math.sqrt((dim + MARK * MARK + BIAS * BIAS) / dim);   // rmsnorm'd marker
  const C1 = 90;                                           // dim1 = C1 * (markers / 2) / context
  const G = C1 / markXn;                                   // wo gain: 0.5 (sigmoid(0) gate) * 2 markers * markXn * G = C1
  const GAMMA = 12, T = 11.5;                              // T: typical best logit of the other rows
  const BETA = (T + GAMMA * C1 / o.eosAt) / (BIAS / 1.02); // final rms ~1.02

  // embeddings (rms ~1) and head
  const imStart = id["<|im_start|>"];
  mat("token_embd.weight", vocab, dim, 1, { fill: (x) => {
    for (let i = 0; i < x.length; i++) x[i] = U(Math.sqrt(3));
    for (let r = 0; r < vocab; r++) { x[r * dim + F_MARK] = r === imStart ? MARK : 0; x[r * dim + F_COUNT] = 0; x[r * dim + F_BIAS] = BIAS; }
  } });
  const outNorm = Array.from({ length: dim }, (_, i) => normF(i));
  vec("output_norm.weight", dim, (i) => outNorm[i]);
  const eos = [id["<|im_end|>"], id["<|endoftext|>"]];
  // special rows and byte tokens that are not printable ASCII get an all-zero row (logit 0, far
  // below the best), so greedy answers decode to readable (nonsense) text; <|im_end|> is the counter
  const printable = (t) => [...t].every((ch) => { const c = ch.charCodeAt(0); return (c >= 33 && c <= 126) || c === 0x120 /* Ġ = space */ || c === 0x10A /* Ċ = \n */; });
  const silent = new Set([...eos, imStart, id["<think>"], id["</think>"], ...tokens.map((t, i) => (i < byteTokens.length + merges.length && !printable(t) ? i : -1)).filter((i) => i >= 0)]);
  mat("output.weight", vocab, dim, o.headGain, { role: "in", fill: (x, a) => {
    for (let r = 0; r < vocab; r++) for (let c = 0; c < dim; c++) x[r * dim + c] = silent.has(r) ? 0 : U(a);
  }, edit: o.counter ? (x) => { const r = id["<|im_end|>"]; x[r * dim + F_COUNT] = -GAMMA; x[r * dim + F_BIAS] = BETA; } : null });

  const layer = (i, full, counter = false) => {
    const p = `blk.${i}.`;
    vec(p + "attn_norm.weight", dim, normF);
    vec(p + "post_attention_norm.weight", dim, normF);
    mat(p + "ffn_gate.weight", inter, dim, 1, { role: "in" });
    mat(p + "ffn_up.weight", inter, dim, 1, { role: "in" });
    // the real 27B file keeps ffn_down, ssm_out and attn_output in Q8_0 (q8out mirrors that)
    const OT = o.q8out ? GGML.Q8_0 : GGML.Q4_0;
    mat(p + "ffn_down.weight", dim, inter, ls * 2, { role: "out", type: OT });
    if (full) {
      // [q | gate] per head; the counter head (0) has q = 0 and gate = 0 (sigmoid 0.5)
      mat(p + "attn_q.weight", 2 * qDim, dim, 1, { role: "in", edit: counter ? (x) => x.fill(0, 0, 2 * hd * dim) : null });
      mat(p + "attn_k.weight", kvDim, dim, 1, { role: "in" });
      // kv head 0, component 0 = the marker (normed dim 0)
      mat(p + "attn_v.weight", kvDim, dim, 1, { role: "in", edit: counter ? (x) => { x.fill(0, 0, dim); x[F_MARK] = 1; } : null });
      mat(p + "attn_output.weight", dim, qDim, ls * 2, { role: "out", type: OT, edit: counter ? (x) => {
        for (let r = 0; r < dim; r++) { x[r * qDim + 0] = 0; x[r * qDim + hd] = 0; }   // heads 0 and 1 share kv head 0
        x[F_COUNT * qDim + 0] = G;
      } : null });
      vec(p + "attn_q_norm.weight", hd, norm);
      vec(p + "attn_k_norm.weight", hd, norm);
    } else {
      mat(p + "attn_qkv.weight", convDim, dim, 1, { role: "in" });
      mat(p + "attn_gate.weight", dInner, dim, 1, { role: "in" });
      mat(p + "ssm_beta.weight", nVH, dim, 1, { type: GGML.Q8_0, role: "in" });
      mat(p + "ssm_alpha.weight", nVH, dim, 1, { type: GGML.Q8_0, role: "in" });
      vec(p + "ssm_dt.bias", nVH, () => U(0.5));
      vec(p + "ssm_a", nVH, () => -(0.3 + rnd() * 1.5));   // A = -exp(A_log) < 0 -> decay in (0, 1)
      f32mat(p + "ssm_conv1d.weight", convDim, 4, () => U(0.6));
      vec(p + "ssm_norm.weight", dState, norm);
      mat(p + "ssm_out.weight", dim, dInner, ls * 3, { role: "out", type: OT });
    }
  };
  for (let i = 0; i < L; i++) layer(i, i % 4 === 3, o.counter && i === 3);
  // multi-token-prediction block: a full-attention layer + eh_proj / enorm / hnorm / shared_head_norm
  layer(N, true);
  const p = `blk.${N}.nextn.`;
  const echo = o.mtp === "echo";
  // x = eh_proj [enorm(e) | hnorm(h)]. echo: copy the embedding half (not the feature dims), a
  // faint trace of h
  mat(p + "eh_proj.weight", dim, 2 * dim, 1, { fill: echo ? (x) => {
    for (let r = 0; r < dim; r++) for (let c = 0; c < 2 * dim; c++)
      x[r * 2 * dim + c] = c === r ? 1 : c >= dim ? U(0.02) : 0;
  } : null, edit: (x) => {
    x.fill(0, 0, NF * 2 * dim);
    for (let r = 0; r < dim; r++) for (let c = 0; c < NF; c++) { x[r * 2 * dim + c] = 0; x[r * 2 * dim + dim + c] = 0; }
  } });
  // echo: the draft sees the same norm weights as the trunk's final norm, so its prediction for
  // "the token after t" matches the trunk's up to the (small) layer updates
  vec(p + "enorm.weight", dim, echo ? () => 1 : norm);
  vec(p + "hnorm.weight", dim, norm);
  vec(p + "shared_head_norm.weight", dim, echo ? (i) => outNorm[i] : norm);

  // ---- serialize ----
  const w = new W();
  const kv = [
    ["general.architecture", GT.STR, "qwen35"],
    ["general.name", GT.STR, "SwarmLLM synthetic qwen35 (test only)"],
    ["general.alignment", GT.U32, 32],
    ["general.file_type", GT.U32, 2],
    ["qwen35.block_count", GT.U32, L + 1],
    ["qwen35.nextn_predict_layers", GT.U32, 1],
    ["qwen35.context_length", GT.U32, o.ctx],
    ["qwen35.embedding_length", GT.U32, dim],
    ["qwen35.feed_forward_length", GT.U32, inter],
    ["qwen35.attention.head_count", GT.U32, nH],
    ["qwen35.attention.head_count_kv", GT.U32, nKV],
    ["qwen35.attention.key_length", GT.U32, hd],
    ["qwen35.attention.value_length", GT.U32, hd],
    ["qwen35.attention.layer_norm_rms_epsilon", GT.F32, o.eps],
    ["qwen35.rope.dimension_count", GT.U32, o.nRot],
    ["qwen35.rope.freq_base", GT.F32, o.ropeTheta],
    ["qwen35.full_attention_interval", GT.U32, 4],
    ["qwen35.ssm.conv_kernel", GT.U32, 4],
    ["qwen35.ssm.state_size", GT.U32, dState],
    ["qwen35.ssm.group_count", GT.U32, nKH],
    ["qwen35.ssm.time_step_rank", GT.U32, nVH],
    ["qwen35.ssm.inner_size", GT.U32, dInner],
    ["qwen35.vocab_size", GT.U32, vocab],
    ["tokenizer.ggml.model", GT.STR, "gpt2"],
    ["tokenizer.ggml.pre", GT.STR, "qwen2"],
    ["tokenizer.ggml.tokens", GT.ARR, [GT.STR, tokens]],
    ["tokenizer.ggml.token_type", GT.ARR, [GT.I32, tokenType]],
    ["tokenizer.ggml.merges", GT.ARR, [GT.STR, merges]],
    ["tokenizer.ggml.eos_token_id", GT.U32, id["<|im_end|>"]],
    ["tokenizer.ggml.padding_token_id", GT.U32, id["<|endoftext|>"]],
    ["tokenizer.ggml.add_bos_token", GT.BOOL, false],
  ];
  w.u32(0x46554747); w.u32(3); w.u64(tensors.length); w.u64(kv.length);
  for (const [k, t, v] of kv) writeKV(w, k, t, v);
  let off = 0;
  for (const t of tensors) {
    w.str(t.name);
    w.u32(t.shape.length);
    for (const d of t.shape.slice().reverse()) w.u64(d);   // ggml order: ne0 = innermost
    w.u32(t.type);
    w.u64(off);
    t.off = off;
    off += Math.ceil(t.data.length / 32) * 32;
  }
  w.pad(32);
  for (const t of tensors) { w.bytes(t.data); w.pad(32); }
  return { bytes: w.out(), info: { vocab, dim, layers: L, blockCount: L + 1, nTensors: tensors.length, mtp: o.mtp, ids: id } };
}

export function writeSynth(file, opts = {}) {
  const { bytes, info } = buildSynthGGUF(opts);
  fs.mkdirSync(path.dirname(path.resolve(file)), { recursive: true });
  fs.writeFileSync(file, bytes);
  return { file: path.resolve(file), size: bytes.length, ...info };
}

// ---------------- CLI ----------------
export const DEFAULT_OUT = path.resolve(new URL(".", import.meta.url).pathname, "../../models/synth/qwen35-synth.gguf");
if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname)) {
  const argv = process.argv.slice(2);
  const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
  const out = argv[0] && !argv[0].startsWith("--") ? argv[0] : DEFAULT_OUT;
  const shape = arg("shape") === "27b" ? SHAPE_27B : {};
  const r = writeSynth(out, {
    ...shape,
    ...(arg("layers") ? { layers: +arg("layers") } : {}),
    ...(arg("dim") ? { dim: +arg("dim") } : {}),
    seed: +arg("seed", DEFAULTS.seed),
    inter: +arg("inter", shape.inter || DEFAULTS.inter), mtp: arg("mtp", DEFAULTS.mtp), layerScale: +arg("layer-scale", DEFAULTS.layerScale),
    merges: +arg("vocab-merges", DEFAULTS.merges), eosAt: +arg("eos-at", DEFAULTS.eosAt),
  });
  const { ids, ...rest } = r;
  console.log(JSON.stringify(rest));
}
