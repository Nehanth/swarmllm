// bello engine — WebGPU Llama-architecture inference, layer-shardable.
import { quantizeQ8 } from "./gguf.js";
// Runs identically in browsers and Deno. The golden reference is ref.js.
//
// Sharding model: an engine instance owns layers [lo, hi). The host peer also
// owns embed/head. Mid-pipeline peers only ever call runHidden().

export function parseSafetensors(arrayBuf) {
  const dv = new DataView(arrayBuf);
  const headerLen = Number(dv.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(arrayBuf, 8, headerLen)));
  const base = 8 + headerLen;
  const out = {};
  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    out[name] = { dtype: info.dtype, shape: info.shape, arrayBuf, byteOffset: base + info.data_offsets[0], byteLength: info.data_offsets[1] - info.data_offsets[0] };
  }
  return out;
}

export function tensorF32(t) {
  const n = t.shape.reduce((a, b) => a * b, 1);
  if (t.dtype === "F32") return new Float32Array(t.arrayBuf, t.byteOffset, n);
  if (t.dtype === "BF16") {
    const u16 = new Uint16Array(t.arrayBuf, t.byteOffset, n);
    const out = new Float32Array(n);
    const u32 = new Uint32Array(out.buffer); // view f32 bits directly
    for (let i = 0; i < n; i++) u32[i] = u16[i] << 16;
    return out;
  }
  throw new Error("unsupported dtype " + t.dtype);
}

// ---------- tokenizer (byte-level BPE, same as ref.js) ----------
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
export function shardTensorNames(cfg, [lo, hi], hasEmbed, hasHead) {
  const names = [];
  if (hasEmbed || hasHead) names.push("model.embed_tokens.weight");
  if (hasHead) names.push("model.norm.weight");
  const parts = ["input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight", "post_attention_layernorm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"];
  for (let i = lo; i < hi; i++) for (const s of parts) names.push(`model.layers.${i}.${s}`);
  return names;
}

// Fetch only the named tensors via HTTP Range requests (falls back to a full
// download if the server won't do ranges). Returns the same tensor-map shape
// parseSafetensors produces.
export async function fetchModelShard(url, names, onProgress = () => {}) {
  const tryRange = async (from, to) => {
    const r = await fetch(url, { headers: { Range: `bytes=${from}-${to}` } });
    return r.status === 206 ? await r.arrayBuffer() : null;
  };
  let head = null;
  try { head = await tryRange(0, 7); } catch { head = null; }
  if (head) {
    const headerLen = Number(new DataView(head).getBigUint64(0, true));
    const hbuf = await tryRange(8, 8 + headerLen - 1);
    const hjson = JSON.parse(new TextDecoder().decode(new Uint8Array(hbuf)));
    const base = 8 + headerLen;
    const infos = names.map((n) => {
      if (!hjson[n]) throw new Error("tensor not in file: " + n);
      return [n, hjson[n]];
    });
    const totalBytes = infos.reduce((s, [, i]) => s + i.data_offsets[1] - i.data_offsets[0], 0);
    let done = 0;
    const out = {};
    const queue = [...infos];
    await Promise.all(Array.from({ length: 5 }, async () => {
      while (queue.length) {
        const [name, info] = queue.shift();
        const [b0, b1] = info.data_offsets;
        const buf = await tryRange(base + b0, base + b1 - 1);
        if (!buf || buf.byteLength !== b1 - b0) throw new Error("range fetch failed: " + name);
        out[name] = { dtype: info.dtype, shape: info.shape, arrayBuf: buf, byteOffset: 0, byteLength: b1 - b0 };
        done += b1 - b0;
        onProgress(done / totalBytes, done, totalBytes);
      }
    }));
    return out;
  }
  // fallback: whole file
  const r = await fetch(url);
  const total = +r.headers.get("Content-Length") || 0;
  const reader = r.body.getReader();
  const chunks = [];
  let got = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(total ? got / total : 0, got, total);
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  const all = parseSafetensors(buf.buffer);
  const out = {};
  for (const n of names) out[n] = all[n];
  return out;
}

// ---------- WGSL ----------
export const WGSL = /* wgsl */ `
struct Config {
  dim: u32, kvDim: u32, nH: u32, nKV: u32,
  headDim: u32, inter: u32, vocab: u32, maxSeq: u32,
  eps: f32, theta: f32, qDim: u32,
};
struct Frame { pos: u32, seqLen: u32 };
struct Shape { dOut: u32, dIn: u32 };

@group(0) @binding(0) var<uniform> cfg: Config;
@group(0) @binding(1) var<uniform> frame: Frame;

// --- matvec: y = W x, W is [dOut, dIn] row-major ---
@group(1) @binding(0) var<storage, read> mv_w: array<f32>;
@group(1) @binding(1) var<storage, read> mv_x: array<f32>;
@group(1) @binding(2) var<storage, read_write> mv_y: array<f32>;
@group(1) @binding(3) var<uniform> mv_shape: Shape;
@compute @workgroup_size(64)
fn matvec(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= mv_shape.dOut) { return; }
  var acc: f32 = 0.0;
  let off = r * mv_shape.dIn;
  for (var c: u32 = 0u; c < mv_shape.dIn; c++) {
    acc += mv_w[off + c] * mv_x[c];
  }
  mv_y[r] = acc;
}

// --- matvec_q8: y = W x with W in Q8_0 (int8 + per-32-block f32 scale) ---
@group(1) @binding(0) var<storage, read> q8_qs: array<u32>;   // int8s packed 4/word
@group(1) @binding(1) var<storage, read> q8_sc: array<f32>;   // scale per 32-block
@group(1) @binding(2) var<storage, read> q8_x: array<f32>;
@group(1) @binding(3) var<storage, read_write> q8_y: array<f32>;
@group(1) @binding(4) var<uniform> q8_shape: Shape;
@compute @workgroup_size(64)
fn matvec_q8(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= q8_shape.dOut) { return; }
  let dIn = q8_shape.dIn;
  let nb = dIn / 32u;
  var acc: f32 = 0.0;
  let rowWords = r * dIn / 4u;
  let rowSc = r * nb;
  for (var b: u32 = 0u; b < nb; b++) {
    var sum: f32 = 0.0;
    let wBase = rowWords + b * 8u;
    let xBase = b * 32u;
    for (var w: u32 = 0u; w < 8u; w++) {
      let word = bitcast<i32>(q8_qs[wBase + w]);
      let x0 = xBase + w * 4u;
      sum += f32(extractBits(word, 0u, 8u)) * q8_x[x0]
           + f32(extractBits(word, 8u, 8u)) * q8_x[x0 + 1u]
           + f32(extractBits(word, 16u, 8u)) * q8_x[x0 + 2u]
           + f32(extractBits(word, 24u, 8u)) * q8_x[x0 + 3u];
    }
    acc += q8_sc[rowSc + b] * sum;
  }
  q8_y[r] = acc;
}

// --- per-head rmsnorm (qwen3 QK-norm): each head's headDim slice normalized,
// scaled by a shared [headDim] weight ---
@group(1) @binding(0) var<storage, read_write> hn_v: array<f32>;
@group(1) @binding(1) var<storage, read> hn_w: array<f32>;
@group(1) @binding(2) var<uniform> hn_nheads: u32;
@compute @workgroup_size(32)
fn head_norm(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  if (h >= hn_nheads) { return; }
  let off = h * cfg.headDim;
  var ss: f32 = 0.0;
  for (var i: u32 = 0u; i < cfg.headDim; i++) { let v = hn_v[off + i]; ss += v * v; }
  let inv = inverseSqrt(ss / f32(cfg.headDim) + cfg.eps);
  for (var i: u32 = 0u; i < cfg.headDim; i++) { hn_v[off + i] *= inv * hn_w[i]; }
}

// --- matvec_q4: y = W x with W in Q4_0 (packed nibbles + per-32-block f32 scale) ---
// nibble layout per block: byte j holds elem j (low nibble) and elem j+16 (high)
@group(1) @binding(0) var<storage, read> q4_qs: array<u32>;
@group(1) @binding(1) var<storage, read> q4_sc: array<f32>;
@group(1) @binding(2) var<storage, read> q4_x: array<f32>;
@group(1) @binding(3) var<storage, read_write> q4_y: array<f32>;
@group(1) @binding(4) var<uniform> q4_shape: Shape;
@compute @workgroup_size(64)
fn matvec_q4(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= q4_shape.dOut) { return; }
  let dIn = q4_shape.dIn;
  let nb = dIn / 32u;
  var acc: f32 = 0.0;
  let rowWords = r * dIn / 8u;   // 4 bits/weight -> dIn/8 u32 words per row
  let rowSc = r * nb;
  for (var b: u32 = 0u; b < nb; b++) {
    var sum: f32 = 0.0;
    let wBase = rowWords + b * 4u;
    let xBase = b * 32u;
    for (var w: u32 = 0u; w < 4u; w++) {
      let word = q4_qs[wBase + w];
      let j = xBase + w * 4u;
      sum += (f32(extractBits(word, 0u, 4u)) - 8.0) * q4_x[j]
           + (f32(extractBits(word, 4u, 4u)) - 8.0) * q4_x[j + 16u]
           + (f32(extractBits(word, 8u, 4u)) - 8.0) * q4_x[j + 1u]
           + (f32(extractBits(word, 12u, 4u)) - 8.0) * q4_x[j + 17u]
           + (f32(extractBits(word, 16u, 4u)) - 8.0) * q4_x[j + 2u]
           + (f32(extractBits(word, 20u, 4u)) - 8.0) * q4_x[j + 18u]
           + (f32(extractBits(word, 24u, 4u)) - 8.0) * q4_x[j + 3u]
           + (f32(extractBits(word, 28u, 4u)) - 8.0) * q4_x[j + 19u];
    }
    acc += q4_sc[rowSc + b] * sum;
  }
  q4_y[r] = acc;
}

// --- rmsnorm: y = x * invRms(x) * w ---
@group(1) @binding(0) var<storage, read> rn_x: array<f32>;
@group(1) @binding(1) var<storage, read> rn_w: array<f32>;
@group(1) @binding(2) var<storage, read_write> rn_y: array<f32>;
@group(1) @binding(3) var<uniform> rn_n: u32;
var<workgroup> rn_partial: array<f32, 256>;
@compute @workgroup_size(256)
fn rmsnorm(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  var ss: f32 = 0.0;
  for (var i: u32 = t; i < rn_n; i += 256u) { let v = rn_x[i]; ss += v * v; }
  rn_partial[t] = ss;
  workgroupBarrier();
  var stride: u32 = 128u;
  while (stride > 0u) {
    if (t < stride) { rn_partial[t] += rn_partial[t + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let inv = inverseSqrt(rn_partial[0] / f32(rn_n) + cfg.eps);
  for (var i: u32 = t; i < rn_n; i += 256u) { rn_y[i] = rn_x[i] * inv * rn_w[i]; }
}

// --- rope: rotate pairs (i, i+half) in each head, at frame.pos ---
@group(1) @binding(0) var<storage, read_write> rp_v: array<f32>;
@group(1) @binding(1) var<uniform> rp_nheads: u32;
@compute @workgroup_size(64)
fn rope(@builtin(global_invocation_id) gid: vec3<u32>) {
  let half = cfg.headDim / 2u;
  let total = rp_nheads * half;
  let idx = gid.x;
  if (idx >= total) { return; }
  let h = idx / half;
  let i = idx % half;
  let off = h * cfg.headDim;
  let freq = pow(cfg.theta, -f32(2u * i) / f32(cfg.headDim));
  let ang = f32(frame.pos) * freq;
  let c = cos(ang); let s = sin(ang);
  let a = rp_v[off + i]; let b = rp_v[off + i + half];
  rp_v[off + i] = a * c - b * s;
  rp_v[off + i + half] = b * c + a * s;
}

// --- attention scores: scores[h*maxSeq+t] = dot(q_h, kCache[t]_kvH) / sqrt(hd) ---
@group(1) @binding(0) var<storage, read> at_q: array<f32>;
@group(1) @binding(1) var<storage, read> at_kc: array<f32>;
@group(1) @binding(2) var<storage, read_write> at_scores: array<f32>;
@compute @workgroup_size(64)
fn attn_scores(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = cfg.nH * frame.seqLen;
  let idx = gid.x;
  if (idx >= total) { return; }
  let h = idx / frame.seqLen;
  let t = idx % frame.seqLen;
  let kvH = h / (cfg.nH / cfg.nKV);
  let qOff = h * cfg.headDim;
  let kOff = t * cfg.kvDim + kvH * cfg.headDim;
  var acc: f32 = 0.0;
  for (var i: u32 = 0u; i < cfg.headDim; i++) {
    acc += at_q[qOff + i] * at_kc[kOff + i];
  }
  at_scores[h * cfg.maxSeq + t] = acc / sqrt(f32(cfg.headDim));
}

// --- attention softmax: per head, serial (seqLen is small) ---
@group(1) @binding(0) var<storage, read_write> sm_scores: array<f32>;
@compute @workgroup_size(1)
fn attn_softmax(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  if (h >= cfg.nH) { return; }
  let off = h * cfg.maxSeq;
  var mx: f32 = -3.0e38;
  for (var t: u32 = 0u; t < frame.seqLen; t++) { mx = max(mx, sm_scores[off + t]); }
  var sum: f32 = 0.0;
  for (var t: u32 = 0u; t < frame.seqLen; t++) {
    let e = exp(sm_scores[off + t] - mx);
    sm_scores[off + t] = e;
    sum += e;
  }
  for (var t: u32 = 0u; t < frame.seqLen; t++) { sm_scores[off + t] /= sum; }
}

// --- attention out: out[h*hd+i] = sum_t scores[h,t] * vCache[t, kvH*hd+i] ---
@group(1) @binding(0) var<storage, read> ao_scores: array<f32>;
@group(1) @binding(1) var<storage, read> ao_vc: array<f32>;
@group(1) @binding(2) var<storage, read_write> ao_out: array<f32>;
@compute @workgroup_size(64)
fn attn_out(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= cfg.qDim) { return; }
  let h = idx / cfg.headDim;
  let i = idx % cfg.headDim;
  let kvH = h / (cfg.nH / cfg.nKV);
  var acc: f32 = 0.0;
  for (var t: u32 = 0u; t < frame.seqLen; t++) {
    acc += ao_scores[h * cfg.maxSeq + t] * ao_vc[t * cfg.kvDim + kvH * cfg.headDim + i];
  }
  ao_out[idx] = acc;
}

// --- silu-gate: g = silu(g) * u ---
@group(1) @binding(0) var<storage, read_write> sg_g: array<f32>;
@group(1) @binding(1) var<storage, read> sg_u: array<f32>;
@compute @workgroup_size(64)
fn silu_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= cfg.inter) { return; }
  let g = sg_g[i];
  sg_g[i] = (g / (1.0 + exp(-g))) * sg_u[i];
}

// --- residual add: x += y (n = dim) ---
@group(1) @binding(0) var<storage, read_write> ad_x: array<f32>;
@group(1) @binding(1) var<storage, read> ad_y: array<f32>;
@compute @workgroup_size(64)
fn add_res(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= cfg.dim) { return; }
  ad_x[i] += ad_y[i];
}

`;

// ============ cooperative matvec family (generated) ============
// One workgroup of WG threads computes ROWS output rows together (the shape
// llama.cpp's WebGPU backend, web-llm's generated kernels and zero-tvm all
// converge on for decode GEMV). Thread t = (block-lane bl = t/4) x (quarter
// qt = t%4, an 8-element slice of a 32-element quant block): consecutive
// threads read consecutive words of the same row (coalesced) and each
// thread's activation slice is loaded once and reused across all ROWS rows.
// Scalar accumulators only: a dynamically-indexed local array spills to
// scratch memory and ran 3x slower. Reduction is a portable shared-memory
// halving tree (no subgroups: absent from shipping Safari 26).
export function coopWGSL(WG = 256, ROWS = 4) {
  const LANES = WG / 4;                  // 32-elem blocks in flight per iteration
  const accDecl = Array.from({ length: ROWS }, (_, r) => `var acc${r} = 0.0;`).join(" ");
  const fullBody = (term) => Array.from({ length: ROWS }, (_, r) => `      acc${r} += ${term(r)};`).join("\n");
  const tailBody = (term, dOut) => Array.from({ length: ROWS - 1 }, (_, r) =>
    `      if (row0 + ${r}u < ${dOut}) { acc${r} += ${term(r)}; }`).join("\n");
  const store = Array.from({ length: ROWS }, (_, r) => `  mvc_part[${r * WG}u + t] = acc${r};`).join("\n");
  const treeAdd = Array.from({ length: ROWS }, (_, r) =>
    `      mvc_part[${r * WG}u + t] += mvc_part[${r * WG}u + t + stride];`).join("\n");
  const reduce = `
${store}
  workgroupBarrier();
  var stride: u32 = ${WG / 2}u;
  while (stride > 0u) {
    if (t < stride) {
${treeAdd}
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }`;
  return /* wgsl */ `
var<workgroup> mvc_part: array<f32, ${WG * ROWS}>;   // [${ROWS} rows][${WG} threads]

// vec4 views of the same buffers the scalar kernels bind (same @group/@binding
// is legal as long as no single entry point references both views). All x/w
// buffers are multiples of 16 bytes (dims divisible by 4).
@group(1) @binding(0) var<storage, read> mv_w4: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> mv_x4: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read> q8_x4: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read> q4_x4: array<vec4<f32>>;

fn mvf_row(off4: u32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
  return dot(mv_w4[off4], xa) + dot(mv_w4[off4 + 1u], xb);
}
@compute @workgroup_size(${WG})
fn matvec_coop(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let qt = t & 3u;
  let bl = t >> 2u;
  let dIn = mv_shape.dIn;
  let nb = dIn / 32u;
  let dIn4 = dIn / 4u;
  let row0 = wg.x * ${ROWS}u;
  let full = row0 + ${ROWS - 1}u < mv_shape.dOut;
  ${accDecl}
  for (var b: u32 = bl; b < nb; b += ${LANES}u) {
    let c4 = b * 8u + qt * 2u;
    let xa = mv_x4[c4];
    let xb = mv_x4[c4 + 1u];
    let off4 = row0 * dIn4 + c4;
    if (full) {
${fullBody((r) => `mvf_row(off4 + ${r}u * dIn4, xa, xb)`)}
    } else {
${tailBody((r) => `mvf_row(off4 + ${r}u * dIn4, xa, xb)`, "mv_shape.dOut")}
    }
  }
${reduce}
  if (t < ${ROWS}u) {
    let row = row0 + t;
    if (row < mv_shape.dOut) { mv_y[row] = mvc_part[t * ${WG}u]; }
  }
}

fn q8_row(wBase: u32, sc: f32, xa: vec4<f32>, xb: vec4<f32>) -> f32 {
  let w0 = bitcast<i32>(q8_qs[wBase]);
  let w1 = bitcast<i32>(q8_qs[wBase + 1u]);
  // shift-based sign extension (extractBits goes through slow polyfill paths)
  let d0 = vec4<f32>(f32((w0 << 24u) >> 24u), f32((w0 << 16u) >> 24u),
                     f32((w0 << 8u) >> 24u), f32(w0 >> 24u));
  let d1 = vec4<f32>(f32((w1 << 24u) >> 24u), f32((w1 << 16u) >> 24u),
                     f32((w1 << 8u) >> 24u), f32(w1 >> 24u));
  return sc * (dot(d0, xa) + dot(d1, xb));
}
@compute @workgroup_size(${WG})
fn matvec_q8_coop(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let qt = t & 3u;
  let bl = t >> 2u;
  let dIn = q8_shape.dIn;
  let nb = dIn / 32u;
  let rowWords = dIn / 4u;
  let row0 = wg.x * ${ROWS}u;
  let full = row0 + ${ROWS - 1}u < q8_shape.dOut;
  ${accDecl}
  for (var b: u32 = bl; b < nb; b += ${LANES}u) {
    let x4 = b * 8u + qt * 2u;
    let xa = q8_x4[x4];
    let xb = q8_x4[x4 + 1u];
    let wBase = row0 * rowWords + b * 8u + qt * 2u;
    let scBase = row0 * nb + b;
    if (full) {
${fullBody((r) => `q8_row(wBase + ${r}u * rowWords, q8_sc[scBase + ${r}u * nb], xa, xb)`)}
    } else {
${tailBody((r) => `q8_row(wBase + ${r}u * rowWords, q8_sc[scBase + ${r}u * nb], xa, xb)`, "q8_shape.dOut")}
    }
  }
${reduce}
  if (t < ${ROWS}u) {
    let row = row0 + t;
    if (row < q8_shape.dOut) { q8_y[row] = mvc_part[t * ${WG}u]; }
  }
}

fn q4_row(wIdx: u32, sc: f32, xlo: vec4<f32>, xhi: vec4<f32>) -> f32 {
  let word = q4_qs[wIdx];
  let lo = vec4<f32>(f32(word & 0xFu), f32((word >> 8u) & 0xFu),
                     f32((word >> 16u) & 0xFu), f32((word >> 24u) & 0xFu)) - vec4<f32>(8.0);
  let hi = vec4<f32>(f32((word >> 4u) & 0xFu), f32((word >> 12u) & 0xFu),
                     f32((word >> 20u) & 0xFu), f32((word >> 28u) & 0xFu)) - vec4<f32>(8.0);
  return sc * (dot(lo, xlo) + dot(hi, xhi));
}
@compute @workgroup_size(${WG})
fn matvec_q4_coop(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let qt = t & 3u;
  let bl = t >> 2u;
  let dIn = q4_shape.dIn;
  let nb = dIn / 32u;
  let rowWords = dIn / 8u;   // 4 bits/weight -> dIn/8 u32 words per row
  let row0 = wg.x * ${ROWS}u;
  let full = row0 + ${ROWS - 1}u < q4_shape.dOut;
  ${accDecl}
  for (var b: u32 = bl; b < nb; b += ${LANES}u) {
    // word qt of block b covers x[j..j+3] (low nibbles) and x[j+16..j+19] (high)
    let xlo = q4_x4[b * 8u + qt];
    let xhi = q4_x4[b * 8u + qt + 4u];
    let wIdx = row0 * rowWords + b * 4u + qt;
    let scBase = row0 * nb + b;
    if (full) {
${fullBody((r) => `q4_row(wIdx + ${r}u * rowWords, q4_sc[scBase + ${r}u * nb], xlo, xhi)`)}
    } else {
${tailBody((r) => `q4_row(wIdx + ${r}u * rowWords, q4_sc[scBase + ${r}u * nb], xlo, xhi)`, "q4_shape.dOut")}
    }
  }
${reduce}
  if (t < ${ROWS}u) {
    let row = row0 + t;
    if (row < q4_shape.dOut) { q4_y[row] = mvc_part[t * ${WG}u]; }
  }
}

// ---- batched (4-column) variants for prefill: each loaded weight word is
// reused across 4 prompt tokens, so prefill reads the weights once per 4
// tokens instead of once per token. x is [4][xs4] vec4s, y is [4][ys] f32s
// (strides carried in BShape; slices are 256-byte aligned by the engine).
// Column accumulation reuses the same 4-row register set per column and the
// same shared-memory tree, one column at a time (keeps workgroup storage at
// the portable minimum).
struct BShape { dOut: u32, dIn: u32, xs4: u32, ys: u32 };
@group(1) @binding(3) var<uniform> mvb_shape: BShape;
@group(1) @binding(4) var<uniform> qb_shape: BShape;

${["", "_q8", "_q4"].map((kind) => `
@compute @workgroup_size(${WG})
fn matvec${kind}_coop_b(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let qt = t & 3u;
  let bl = t >> 2u;
  let dIn = ${kind === "" ? "mvb_shape" : "qb_shape"}.dIn;
  let dOut = ${kind === "" ? "mvb_shape" : "qb_shape"}.dOut;
  let xs4 = ${kind === "" ? "mvb_shape" : "qb_shape"}.xs4;
  let ys = ${kind === "" ? "mvb_shape" : "qb_shape"}.ys;
  let nb = dIn / 32u;
  let row0 = wg.x * ${ROWS}u;
  let full = row0 + ${ROWS - 1}u < dOut;
${Array.from({ length: 4 }, (_, m) => `  ${accDecl.replace(/acc(\d)/g, `acc$1_${m}`)}`).join("\n")}
  for (var b: u32 = bl; b < nb; b += ${LANES}u) {
${kind === "_q4" ? `    let wIdx0 = row0 * (dIn / 8u) + b * 4u + qt;
    let scBase = row0 * nb + b;` : kind === "_q8" ? `    let wBase0 = row0 * (dIn / 4u) + b * 8u + qt * 2u;
    let scBase = row0 * nb + b;` : `    let off40 = row0 * (dIn / 4u) + b * 8u + qt * 2u;`}
${Array.from({ length: 4 }, (_, m) => kind === "_q4" ? `    {
      let xlo = q4_x4[${m}u * xs4 + b * 8u + qt];
      let xhi = q4_x4[${m}u * xs4 + b * 8u + qt + 4u];
      if (full) {
${Array.from({ length: ROWS }, (_, r) => `        acc${r}_${m} += q4_row(wIdx0 + ${r}u * (dIn / 8u), q4_sc[scBase + ${r}u * nb], xlo, xhi);`).join("\n")}
      } else {
${Array.from({ length: ROWS - 1 }, (_, r) => `        if (row0 + ${r}u < dOut) { acc${r}_${m} += q4_row(wIdx0 + ${r}u * (dIn / 8u), q4_sc[scBase + ${r}u * nb], xlo, xhi); }`).join("\n")}
      }
    }` : kind === "_q8" ? `    {
      let xa = q8_x4[${m}u * xs4 + b * 8u + qt * 2u];
      let xb = q8_x4[${m}u * xs4 + b * 8u + qt * 2u + 1u];
      if (full) {
${Array.from({ length: ROWS }, (_, r) => `        acc${r}_${m} += q8_row(wBase0 + ${r}u * (dIn / 4u), q8_sc[scBase + ${r}u * nb], xa, xb);`).join("\n")}
      } else {
${Array.from({ length: ROWS - 1 }, (_, r) => `        if (row0 + ${r}u < dOut) { acc${r}_${m} += q8_row(wBase0 + ${r}u * (dIn / 4u), q8_sc[scBase + ${r}u * nb], xa, xb); }`).join("\n")}
      }
    }` : `    {
      let xa = mv_x4[${m}u * xs4 + b * 8u + qt * 2u];
      let xb = mv_x4[${m}u * xs4 + b * 8u + qt * 2u + 1u];
      if (full) {
${Array.from({ length: ROWS }, (_, r) => `        acc${r}_${m} += mvf_row(off40 + ${r}u * (dIn / 4u), xa, xb);`).join("\n")}
      } else {
${Array.from({ length: ROWS - 1 }, (_, r) => `        if (row0 + ${r}u < dOut) { acc${r}_${m} += mvf_row(off40 + ${r}u * (dIn / 4u), xa, xb); }`).join("\n")}
      }
    }`).join("\n")}
  }
${Array.from({ length: 4 }, (_, m) => `
${Array.from({ length: ROWS }, (_, r) => `  mvc_part[${r * WG}u + t] = acc${r}_${m};`).join("\n")}
  workgroupBarrier();
  {
    var stride: u32 = ${WG / 2}u;
    while (stride > 0u) {
      if (t < stride) {
${Array.from({ length: ROWS }, (_, r) => `        mvc_part[${r * WG}u + t] += mvc_part[${r * WG}u + t + stride];`).join("\n")}
      }
      workgroupBarrier();
      stride = stride >> 1u;
    }
  }
  if (t < ${ROWS}u) {
    let row = row0 + t;
    if (row < dOut) { ${kind === "" ? "mv_y" : kind === "_q8" ? "q8_y" : "q4_y"}[${m}u * ys + row] = mvc_part[t * ${WG}u]; }
  }
  workgroupBarrier();`).join("\n")}
}`).join("\n")}
`;
}


// ---------- weight entries ----------
// A weight entry is {kind:"f32", data:Float32Array} or {kind:"q8", qs:Uint8Array,
// scales:Float32Array}. The engine consumes a normalized structure:
// { embed?, head?, finalNorm?, layers: [{inNorm,q,k,v,o,postNorm,gate,up,down,qNorm?,kNorm?}] }

export function weightsFromSafetensors(tensors, { lo, hi, hasEmbed, hasHead }) {
  const f32 = (name) => ({ kind: "f32", data: tensorF32(tensors[name]) });
  const layers = [];
  for (let i = lo; i < hi; i++) {
    const p = `model.layers.${i}.`;
    layers.push({
      inNorm: f32(p + "input_layernorm.weight"),
      q: f32(p + "self_attn.q_proj.weight"),
      k: f32(p + "self_attn.k_proj.weight"),
      v: f32(p + "self_attn.v_proj.weight"),
      o: f32(p + "self_attn.o_proj.weight"),
      postNorm: f32(p + "post_attention_layernorm.weight"),
      gate: f32(p + "mlp.gate_proj.weight"),
      up: f32(p + "mlp.up_proj.weight"),
      down: f32(p + "mlp.down_proj.weight"),
    });
  }
  const out = { layers };
  if (hasEmbed || hasHead) out.embed = f32("model.embed_tokens.weight");
  if (hasHead) out.finalNorm = f32("model.norm.weight");
  return out;
}

// ---------- engine ----------
export class BelloEngine {
  // opts: { device, cfg, tensors?|weights?, layerRange, hasEmbed, hasHead, maxSeq }
  reset() { this.pos = 0; }   // fresh context; the KV cache is overwritten from position 0

  static async create(opts) {
    const e = new BelloEngine();
    await e._init(opts);
    return e;
  }

  async _init({ device, cfg, tensors, weights, layerRange, hasEmbed = true, hasHead = true, maxSeq = 512, matvecVariant = "coop", coopWG = 256, coopRows = 4 }) {
    this.device = device;
    this.cfg = cfg;
    this.maxSeq = maxSeq;
    this.mvVariant = matvecVariant;
    this.coopWG = coopWG; this.coopRows = coopRows;
    const dim = cfg.hidden_size;
    const nH = cfg.num_attention_heads;
    const nKV = cfg.num_key_value_heads;
    const headDim = cfg.head_dim || dim / nH;
    const qDim = nH * headDim;
    const kvDim = nKV * headDim;
    const inter = cfg.intermediate_size;
    const vocab = cfg.vocab_size;
    this.dims = { dim, nH, nKV, headDim, qDim, kvDim, inter, vocab };
    const [lo, hi] = layerRange || [0, cfg.num_hidden_layers];
    this.lo = lo; this.hi = hi;
    this.hasEmbed = hasEmbed; this.hasHead = hasHead;
    this.pos = 0;

    const W = weights || weightsFromSafetensors(tensors, { lo, hi, hasEmbed, hasHead });

    const mod = device.createShaderModule({ code: WGSL + coopWGSL(coopWG, coopRows) });
    const C = GPUShaderStage.COMPUTE;
    const layout0 = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: C, buffer: { type: "uniform" } },
        { binding: 1, visibility: C, buffer: { type: "uniform" } },
      ],
    });
    const G1 = {
      matvec: ["ro", "ro", "rw", "u"], matvec_q8: ["ro", "ro", "ro", "rw", "u"],
      matvec_q4: ["ro", "ro", "ro", "rw", "u"],
      matvec_coop: ["ro", "ro", "rw", "u"], matvec_q8_coop: ["ro", "ro", "ro", "rw", "u"],
      matvec_q4_coop: ["ro", "ro", "ro", "rw", "u"],
      matvec_coop_b: ["ro", "ro", "rw", "u"], matvec_q8_coop_b: ["ro", "ro", "ro", "rw", "u"],
      matvec_q4_coop_b: ["ro", "ro", "ro", "rw", "u"],
      rmsnorm: ["ro", "ro", "rw", "u"], head_norm: ["rw", "ro", "u"],
      rope: ["rw", "u"], attn_scores: ["ro", "ro", "rw"], attn_softmax: ["rw"],
      attn_out: ["ro", "ro", "rw"], silu_mul: ["rw", "ro"], add_res: ["rw", "ro"],
    };
    const bufType = { u: "uniform", ro: "read-only-storage", rw: "storage" };
    this.pipes = {};
    for (const [name, spec] of Object.entries(G1)) {
      const layout1 = device.createBindGroupLayout({
        entries: spec.map((t, i) => ({ binding: i, visibility: C, buffer: { type: bufType[t] } })),
      });
      this.pipes[name] = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout0, layout1] }),
        compute: { module: mod, entryPoint: name },
      });
    }

    // uniforms
    const cfgData = new ArrayBuffer(48);
    const cu = new Uint32Array(cfgData), cf = new Float32Array(cfgData);
    cu.set([dim, kvDim, nH, nKV, headDim, inter, vocab, maxSeq], 0);
    cf[8] = cfg.rms_norm_eps; cf[9] = cfg.rope_theta; cu[10] = qDim;
    this.cfgBuf = this._buf(cfgData, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.frameBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._shapes = {};
    this.nBufDim = this._buf(new Uint32Array([dim]), GPUBufferUsage.UNIFORM);
    this.nHBuf = this._buf(new Uint32Array([nH]), GPUBufferUsage.UNIFORM);
    this.nKVBuf = this._buf(new Uint32Array([nKV]), GPUBufferUsage.UNIFORM);

    // working buffers
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.x = device.createBuffer({ size: dim * 4, usage: S });
    this.xn = device.createBuffer({ size: dim * 4, usage: S });
    this.q = device.createBuffer({ size: qDim * 4, usage: S });
    this.k = device.createBuffer({ size: kvDim * 4, usage: S });
    this.v = device.createBuffer({ size: kvDim * 4, usage: S });
    this.attnOut = device.createBuffer({ size: qDim * 4, usage: S });
    this.tmpDim = device.createBuffer({ size: dim * 4, usage: S });
    this.g = device.createBuffer({ size: inter * 4, usage: S });
    this.u = device.createBuffer({ size: inter * 4, usage: S });
    this.scores = device.createBuffer({ size: nH * maxSeq * 4, usage: S });
    this.stageX = device.createBuffer({ size: dim * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    // weight upload
    const up = (e) => {
      if (!e) return null;
      if (e.gpu) return e.gpu;            // already streamed onto the GPU during download
      let r;
      if (e.kind === "q8" || e.kind === "q4") r = { kind: e.kind, qs: this._buf(e.qs, GPUBufferUsage.STORAGE), sc: this._buf(e.scales, GPUBufferUsage.STORAGE) };
      else r = { kind: "f32", buf: this._buf(e.data, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) };
      if (e !== this.cpuEmbed) e.qs = e.scales = e.data = null; // release CPU copy (embed rows stay for lookups)
      return r;
    };
    this.layers = W.layers.map((l) => ({
      inNorm: up(l.inNorm), postNorm: up(l.postNorm),
      qNorm: up(l.qNorm), kNorm: up(l.kNorm),
      wq: up(l.q), wk: up(l.k), wv: up(l.v), wo: up(l.o),
      wgate: up(l.gate), wup: up(l.up), wdown: up(l.down),
      kCache: device.createBuffer({ size: maxSeq * kvDim * 4, usage: S }),
      vCache: device.createBuffer({ size: maxSeq * kvDim * 4, usage: S }),
    }));

    if (hasEmbed || hasHead) {
      if (W.embed.kind === "f32") {
        this.embedGPU = this._buf(W.embed.data, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        this.headEntry = { kind: "f32", buf: this.embedGPU };
      } else {
        this.cpuEmbed = W.embed; // dequant rows on CPU per token
        if (hasHead) this.headEntry = up(W.embed);
      }
      if (W.head) this.headEntry = up(W.head); // untied lm_head
    }
    if (hasHead) {
      this.finalNorm = up(W.finalNorm);
      this.logits = device.createBuffer({ size: vocab * 4, usage: S });
      this.stageLogits = device.createBuffer({ size: vocab * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }

    // bind groups
    this.bgCommonFor = {};
    for (const [k2, p] of Object.entries(this.pipes))
      this.bgCommonFor[k2] = this._bg(p, 0, [this.cfgBuf, this.frameBuf]);

    const coop = this.mvVariant === "coop";
    const mv = (w, x, y, dOut, dIn) => {
      const base = w.kind === "q8" ? "matvec_q8" : w.kind === "q4" ? "matvec_q4" : "matvec";
      const pipe = coop ? base + "_coop" : base;
      const bufs = w.kind === "f32" ? [w.buf, x, y, this._shape(dOut, dIn)] : [w.qs, w.sc, x, y, this._shape(dOut, dIn)];
      return { pipe, wgs: coop ? Math.ceil(dOut / this.coopRows) : Math.ceil(dOut / 64), bg: this._bg(this.pipes[pipe], 1, bufs) };
    };
    this._mv = mv;
    const bgNorm = (x, w, y) => this._bg(this.pipes.rmsnorm, 1, [x, w.buf, y, this.nBufDim]);

    this.layerBGs = this.layers.map((L2) => ({
      norm1: bgNorm(this.x, L2.inNorm, this.xn),
      q: mv(L2.wq, this.xn, this.q, qDim, dim),
      k: mv(L2.wk, this.xn, this.k, kvDim, dim),
      v: mv(L2.wv, this.xn, this.v, kvDim, dim),
      qNorm: L2.qNorm ? this._bg(this.pipes.head_norm, 1, [this.q, L2.qNorm.buf, this.nHBuf]) : null,
      kNorm: L2.kNorm ? this._bg(this.pipes.head_norm, 1, [this.k, L2.kNorm.buf, this.nKVBuf]) : null,
      scores: this._bg(this.pipes.attn_scores, 1, [this.q, L2.kCache, this.scores]),
      softmax: this._bg(this.pipes.attn_softmax, 1, [this.scores]),
      attnOut: this._bg(this.pipes.attn_out, 1, [this.scores, L2.vCache, this.attnOut]),
      o: mv(L2.wo, this.attnOut, this.tmpDim, dim, qDim),
      norm2: bgNorm(this.x, L2.postNorm, this.xn),
      gate: mv(L2.wgate, this.xn, this.g, inter, dim),
      up: mv(L2.wup, this.xn, this.u, inter, dim),
      down: mv(L2.wdown, this.g, this.tmpDim, dim, inter),
    }));
    this.bgRopeQ = this._bg(this.pipes.rope, 1, [this.q, this.nHBuf]);
    this.bgRopeK = this._bg(this.pipes.rope, 1, [this.k, this.nKVBuf]);
    this.bgSilu = this._bg(this.pipes.silu_mul, 1, [this.g, this.u]);
    this.bgAddTmp = this._bg(this.pipes.add_res, 1, [this.x, this.tmpDim]);
    if (hasHead) {
      this.bgFinalNorm = bgNorm(this.x, this.finalNorm, this.xn);
      this.headOp = mv(this.headEntry, this.xn, this.logits, vocab, dim);
    }
  }

  _shape(dOut, dIn) {
    const key = dOut + "," + dIn;
    if (!this._shapes[key])
      this._shapes[key] = this._buf(new Uint32Array([dOut, dIn, 0, 0]), GPUBufferUsage.UNIFORM);
    return this._shapes[key];
  }

  _shapeB(dOut, dIn, xs4, ys) {
    const key = "b" + dOut + "," + dIn + "," + xs4 + "," + ys;
    if (!this._shapes[key])
      this._shapes[key] = this._buf(new Uint32Array([dOut, dIn, xs4, ys]), GPUBufferUsage.UNIFORM);
    return this._shapes[key];
  }

  _buf(data, usage) {
    const src = ArrayBuffer.isView(data) ? data : new Uint8Array(data);
    const size = Math.ceil(src.byteLength / 4) * 4;
    const buf = this.device.createBuffer({ size, usage, mappedAtCreation: true });
    new Uint8Array(buf.getMappedRange()).set(
      new Uint8Array(src.buffer, src.byteOffset, src.byteLength));
    buf.unmap();
    return buf;
  }

  _bg(pipe, group, buffers) {
    return this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(group),
      entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
  }

  _dispatch(pass, pipeName, bg, threads, wgSize = 64) {
    pass.setPipeline(this.pipes[pipeName]);
    pass.setBindGroup(0, this.bgCommonFor[pipeName]);
    pass.setBindGroup(1, bg);
    pass.dispatchWorkgroups(Math.ceil(threads / wgSize));
  }

  _dispatchOp(pass, op) {
    pass.setPipeline(this.pipes[op.pipe]);
    pass.setBindGroup(0, this.bgCommonFor[op.pipe]);
    pass.setBindGroup(1, op.bg);
    pass.dispatchWorkgroups(op.wgs);
  }

  _setFrame(pos, seqLen) {
    this.device.queue.writeBuffer(this.frameBuf, 0, new Uint32Array([pos, seqLen]));
  }

  _encodeLayer(enc, i) {
    const { qDim, nH, nKV, headDim, kvDim, inter, dim } = this.dims;
    const L = this.layers[i], BG = this.layerBGs[i];
    const seqLen = this.pos + 1;
    {
      const pass = enc.beginComputePass();
      this._dispatch(pass, "rmsnorm", BG.norm1, 256, 256);
      this._dispatchOp(pass, BG.q);
      this._dispatchOp(pass, BG.k);
      this._dispatchOp(pass, BG.v);
      if (BG.qNorm) this._dispatch(pass, "head_norm", BG.qNorm, nH, 32);
      if (BG.kNorm) this._dispatch(pass, "head_norm", BG.kNorm, nKV, 32);
      this._dispatch(pass, "rope", this.bgRopeQ, nH * headDim / 2);
      this._dispatch(pass, "rope", this.bgRopeK, nKV * headDim / 2);
      pass.end();
    }
    enc.copyBufferToBuffer(this.k, 0, L.kCache, this.pos * kvDim * 4, kvDim * 4);
    enc.copyBufferToBuffer(this.v, 0, L.vCache, this.pos * kvDim * 4, kvDim * 4);
    {
      const pass = enc.beginComputePass();
      this._dispatch(pass, "attn_scores", BG.scores, nH * seqLen);
      this._dispatch(pass, "attn_softmax", BG.softmax, nH, 1);
      this._dispatch(pass, "attn_out", BG.attnOut, qDim);
      this._dispatchOp(pass, BG.o);
      this._dispatch(pass, "add_res", this.bgAddTmp, dim);
      this._dispatch(pass, "rmsnorm", BG.norm2, 256, 256);
      this._dispatchOp(pass, BG.gate);
      this._dispatchOp(pass, BG.up);
      this._dispatch(pass, "silu_mul", this.bgSilu, inter);
      this._dispatchOp(pass, BG.down);
      this._dispatch(pass, "add_res", this.bgAddTmp, dim);
      pass.end();
    }
  }

  _embedRowF32(id) {
    const { dim } = this.dims;
    const e = this.cpuEmbed;
    if (e.kind === "f32") return e.data.subarray(id * dim, (id + 1) * dim);
    const nb = dim / 32;
    const out = new Float32Array(dim);
    const rowB = id * nb;
    for (let b = 0; b < nb; b++) {
      const s = e.scales[rowB + b];
      if (e.kind === "q4") {
        const qBase = (rowB + b) * 16;
        for (let j = 0; j < 16; j++) {
          const q = e.qs[qBase + j];
          out[b * 32 + j] = s * ((q & 0xF) - 8);
          out[b * 32 + j + 16] = s * ((q >> 4) - 8);
        }
      } else {
        const qBase = (rowB + b) * 32;
        for (let i = 0; i < 32; i++) {
          const q = e.qs[qBase + i];
          out[b * 32 + i] = s * (q > 127 ? q - 256 : q);
        }
      }
    }
    return out;
  }

  _stageEmbed(tokenId) {
    const { dim } = this.dims;
    if (this.embedGPU) {
      const enc = this.device.createCommandEncoder();
      enc.copyBufferToBuffer(this.embedGPU, tokenId * dim * 4, this.x, 0, dim * 4);
      this.device.queue.submit([enc.finish()]);
    } else {
      this.device.queue.writeBuffer(this.x, 0, this._embedRowF32(tokenId));
    }
  }

  async _readback(srcBuf, stageBuf, n) {
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(srcBuf, 0, stageBuf, 0, n * 4);
    this.device.queue.submit([enc.finish()]);
    await stageBuf.mapAsync(GPUMapMode.READ);
    const out = Float32Array.from(new Float32Array(stageBuf.getMappedRange(), 0, n));
    stageBuf.unmap();
    return out;
  }

  // host peer: full forward for one token -> logits.
  // Whole token (all layers + head) is recorded into ONE command encoder and
  // submitted once: per-submit validation/IPC used to cost ~67 submits/token.
  async forwardToken(tokenId, debugCapture) {
    const { dim, vocab } = this.dims;
    this._setFrame(this.pos, this.pos + 1);
    this._stageEmbed(tokenId);
    if (debugCapture) {           // slow path: per-layer readback for tests
      for (let i = 0; i < this.layers.length; i++) {
        const e2 = this.device.createCommandEncoder();
        this._encodeLayer(e2, i);
        this.device.queue.submit([e2.finish()]);
        debugCapture[this.lo + i] = (await this._readback(this.x, this.stageX, dim)).slice(0, 8);
      }
      const e3 = this.device.createCommandEncoder();
      const pass = e3.beginComputePass();
      this._dispatch(pass, "rmsnorm", this.bgFinalNorm, 256, 256);
      this._dispatchOp(pass, this.headOp);
      pass.end();
      this.device.queue.submit([e3.finish()]);
    } else {
      const enc = this.device.createCommandEncoder();
      for (let i = 0; i < this.layers.length; i++) this._encodeLayer(enc, i);
      const pass = enc.beginComputePass();
      this._dispatch(pass, "rmsnorm", this.bgFinalNorm, 256, 256);
      this._dispatchOp(pass, this.headOp);
      pass.end();
      this.device.queue.submit([enc.finish()]);
    }
    const logits = await this._readback(this.logits, this.stageLogits, vocab);
    this.pos++;
    return logits;
  }

  // ---- batched prefill (4 prompt tokens per pass; weights read once per 4) ----
  _initBatch() {
    const { dim, qDim, kvDim, inter, nH, nKV } = this.dims;
    const dev = this.device;
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const al = (n) => Math.ceil(n * 4 / 256) * 256;         // 256-aligned slice stride, bytes
    const mkB = (n) => ({ buf: dev.createBuffer({ size: 4 * al(n), usage: S }), stride: al(n), n });
    const B = this.B = {
      x: mkB(dim), xn: mkB(dim), q: mkB(qDim), k: mkB(kvDim), v: mkB(kvDim),
      attnOut: mkB(qDim), tmpDim: mkB(dim), g: mkB(inter), u: mkB(inter),
    };
    const slice = (b, c) => ({ buffer: b.buf, offset: c * b.stride, size: b.n * 4 });
    this._bslice = slice;
    // per-column frame uniforms + per-column group0 for the per-token kernels
    this.frameBufsB = [0, 1, 2, 3].map(() => dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    const colPipes = ["rmsnorm", "head_norm", "rope", "attn_scores", "attn_softmax", "attn_out", "silu_mul", "add_res"];
    this.bgCommonB = [0, 1, 2, 3].map((c) => {
      const m = {};
      for (const name of colPipes)
        m[name] = this._bg2g0(this.pipes[name], [{ buffer: this.cfgBuf }, { buffer: this.frameBufsB[c] }]);
      return m;
    });
    // batched matvec op builder: whole B-buffers bound, strides in the uniform
    const mvB = (w, xB, yB, dOut, dIn) => {
      const base = w.kind === "q8" ? "matvec_q8" : w.kind === "q4" ? "matvec_q4" : "matvec";
      const pipe = base + "_coop_b";
      const shp = this._shapeB(dOut, dIn, xB.stride / 16, yB.stride / 4);
      const bufs = w.kind === "f32" ? [w.buf, xB.buf, yB.buf, shp] : [w.qs, w.sc, xB.buf, yB.buf, shp];
      return { pipe, wgs: Math.ceil(dOut / this.coopRows), bg: this._bg(this.pipes[pipe], 1, bufs) };
    };
    // per-layer batched resources
    this.layerB = this.layers.map((L) => {
      const bgNormC = (xB, w, yB, c) => this._bg2res(this.pipes.rmsnorm,
        [slice(xB, c), { buffer: w.buf }, slice(yB, c), { buffer: this.nBufDim }]);
      return {
        qkv: [mvB(L.wq, B.xn, B.q, qDim, dim), mvB(L.wk, B.xn, B.k, kvDim, dim), mvB(L.wv, B.xn, B.v, kvDim, dim)],
        o: mvB(L.wo, B.attnOut, B.tmpDim, dim, qDim),
        gateUp: [mvB(L.wgate, B.xn, B.g, inter, dim), mvB(L.wup, B.xn, B.u, inter, dim)],
        down: mvB(L.wdown, B.g, B.tmpDim, dim, inter),
        cols: [0, 1, 2, 3].map((c) => ({
          norm1: bgNormC(B.x, L.inNorm, B.xn, c),
          norm2: bgNormC(B.x, L.postNorm, B.xn, c),
          qNorm: L.qNorm ? this._bg2res(this.pipes.head_norm, [slice(B.q, c), { buffer: L.qNorm.buf }, { buffer: this.nHBuf }]) : null,
          kNorm: L.kNorm ? this._bg2res(this.pipes.head_norm, [slice(B.k, c), { buffer: L.kNorm.buf }, { buffer: this.nKVBuf }]) : null,
          ropeQ: this._bg2res(this.pipes.rope, [slice(B.q, c), { buffer: this.nHBuf }]),
          ropeK: this._bg2res(this.pipes.rope, [slice(B.k, c), { buffer: this.nKVBuf }]),
          scores: this._bg2res(this.pipes.attn_scores, [slice(B.q, c), { buffer: L.kCache }, { buffer: this.scores }]),
          softmax: this._bg2res(this.pipes.attn_softmax, [{ buffer: this.scores }]),
          attnOut: this._bg2res(this.pipes.attn_out, [{ buffer: this.scores }, { buffer: L.vCache }, slice(B.attnOut, c)]),
          addTmp: this._bg2res(this.pipes.add_res, [slice(B.x, c), slice(B.tmpDim, c)]),
          silu: this._bg2res(this.pipes.silu_mul, [slice(B.g, c), slice(B.u, c)]),
        })),
      };
    });
  }

  _bg2g0(pipe, resources) {
    return this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: resources.map((r, i) => ({ binding: i, resource: r })),
    });
  }
  _bg2res(pipe, resources) {
    return this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(1),
      entries: resources.map((r, i) => ({ binding: i, resource: r })),
    });
  }
  _dCol(pass, name, col, bg, threads, wgSize = 64) {
    pass.setPipeline(this.pipes[name]);
    pass.setBindGroup(0, this.bgCommonB[col][name]);
    pass.setBindGroup(1, bg);
    pass.dispatchWorkgroups(Math.ceil(threads / wgSize));
  }

  _encodeLayerBatch(enc, i, basePos) {
    const { qDim, nH, nKV, headDim, kvDim, inter, dim } = this.dims;
    const L = this.layers[i], LB = this.layerB[i], B = this.B;
    {
      const pass = enc.beginComputePass();
      for (let c = 0; c < 4; c++) this._dCol(pass, "rmsnorm", c, LB.cols[c].norm1, 256, 256);
      for (const op of LB.qkv) this._dispatchOp(pass, op);
      for (let c = 0; c < 4; c++) {
        const C = LB.cols[c];
        if (C.qNorm) this._dCol(pass, "head_norm", c, C.qNorm, nH, 32);
        if (C.kNorm) this._dCol(pass, "head_norm", c, C.kNorm, nKV, 32);
        this._dCol(pass, "rope", c, C.ropeQ, nH * headDim / 2);
        this._dCol(pass, "rope", c, C.ropeK, nKV * headDim / 2);
      }
      pass.end();
    }
    for (let c = 0; c < 4; c++) {
      enc.copyBufferToBuffer(B.k.buf, c * B.k.stride, L.kCache, (basePos + c) * kvDim * 4, kvDim * 4);
      enc.copyBufferToBuffer(B.v.buf, c * B.v.stride, L.vCache, (basePos + c) * kvDim * 4, kvDim * 4);
    }
    {
      const pass = enc.beginComputePass();
      for (let c = 0; c < 4; c++) {
        const C = LB.cols[c];
        this._dCol(pass, "attn_scores", c, C.scores, nH * (basePos + c + 1));
        this._dCol(pass, "attn_softmax", c, C.softmax, nH, 1);
        this._dCol(pass, "attn_out", c, C.attnOut, qDim);
      }
      this._dispatchOp(pass, LB.o);
      for (let c = 0; c < 4; c++) this._dCol(pass, "add_res", c, LB.cols[c].addTmp, dim);
      for (let c = 0; c < 4; c++) this._dCol(pass, "rmsnorm", c, LB.cols[c].norm2, 256, 256);
      for (const op of LB.gateUp) this._dispatchOp(pass, op);
      for (let c = 0; c < 4; c++) this._dCol(pass, "silu_mul", c, LB.cols[c].silu, inter);
      this._dispatchOp(pass, LB.down);
      for (let c = 0; c < 4; c++) this._dCol(pass, "add_res", c, LB.cols[c].addTmp, dim);
      pass.end();
    }
  }

  // consume prompt tokens (no logits): chunks of 4 through the batched path,
  // remainder through the single-token fast path.
  async prefillTokens(ids) {
    if (!this.B && this.hasEmbed) this._initBatch();
    let i = 0;
    let sinceSync = 0;
    while (this.B && ids.length - i >= 4) {
      const basePos = this.pos;
      for (let c = 0; c < 4; c++) {
        this.device.queue.writeBuffer(this.frameBufsB[c], 0, new Uint32Array([basePos + c, basePos + c + 1]));
        if (this.embedGPU) {
          const enc0 = this.device.createCommandEncoder();
          enc0.copyBufferToBuffer(this.embedGPU, ids[i + c] * this.dims.dim * 4, this.B.x.buf, c * this.B.x.stride, this.dims.dim * 4);
          this.device.queue.submit([enc0.finish()]);
        } else {
          this.device.queue.writeBuffer(this.B.x.buf, c * this.B.x.stride, this._embedRowF32(ids[i + c]));
        }
      }
      const enc = this.device.createCommandEncoder();
      for (let l = 0; l < this.layers.length; l++) this._encodeLayerBatch(enc, l, basePos);
      // hidden of the last column becomes the running x for any tail tokens
      enc.copyBufferToBuffer(this.B.x.buf, 3 * this.B.x.stride, this.x, 0, this.dims.dim * 4);
      this.device.queue.submit([enc.finish()]);
      this.pos += 4;
      i += 4;
      if (++sinceSync >= 4) { await this.device.queue.onSubmittedWorkDone(); sinceSync = 0; }
    }
    for (; i < ids.length; i++) {
      await this.prefillToken(ids[i]);
      if (i % 8 === 7) await this.device.queue.onSubmittedWorkDone();
    }
    await this.device.queue.onSubmittedWorkDone();
  }

  // prefill fast path: run the layers for a prompt token, skip head + readback
  // (the head is ~11% of the weight traffic and the logits go unused).
  async prefillToken(tokenId) {
    this._setFrame(this.pos, this.pos + 1);
    this._stageEmbed(tokenId);
    const enc = this.device.createCommandEncoder();
    for (let i = 0; i < this.layers.length; i++) this._encodeLayer(enc, i);
    this.device.queue.submit([enc.finish()]);
    this.pos++;
    // fire-and-forget: the queue is ordered, so later work sees this token's
    // caches. Callers apply backpressure every few tokens via
    // device.queue.onSubmittedWorkDone() to bound queued work.
  }

  // host peer, split mode: embed + local layers -> hidden for next peer
  async embedRun(tokenId, pos) {
    const { dim } = this.dims;
    this.pos = pos;
    this._setFrame(pos, pos + 1);
    this._stageEmbed(tokenId);
    const enc = this.device.createCommandEncoder();
    for (let i = 0; i < this.layers.length; i++) this._encodeLayer(enc, i);
    this.device.queue.submit([enc.finish()]);
    return await this._readback(this.x, this.stageX, dim);
  }

  // host peer, split mode: final norm + lm_head over returned hidden
  async headFromHidden(xIn) {
    const { vocab } = this.dims;
    this.device.queue.writeBuffer(this.x, 0, xIn);
    const enc = this.device.createCommandEncoder();
    {
      const pass = enc.beginComputePass();
      this._dispatch(pass, "rmsnorm", this.bgFinalNorm, 256, 256);
      this._dispatchOp(pass, this.headOp);
      pass.end();
    }
    this.device.queue.submit([enc.finish()]);
    return await this._readback(this.logits, this.stageLogits, vocab);
  }

  // worker peer: hidden in, my layers, hidden out
  async runHidden(xIn, pos) {
    const { dim } = this.dims;
    this.pos = pos;
    this._setFrame(pos, pos + 1);
    this.device.queue.writeBuffer(this.x, 0, xIn);
    const enc = this.device.createCommandEncoder();
    for (let i = 0; i < this.layers.length; i++) this._encodeLayer(enc, i);
    this.device.queue.submit([enc.finish()]);
    return await this._readback(this.x, this.stageX, dim);
  }
}

export function argmax(a) {
  let bi = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i;
  return bi;
}


// ---------- GPU kernel self-test ----------
// Runs a tiny synthetic Llama model through the f32, q8 and q4 kernel paths on
// the given device and compares logits with a CPU reference. Names the broken
// path on GPUs whose drivers/compilers disagree with the spec.
export function quantizeQ4(data) {
  const n = data.length, nb = Math.ceil(n / 32);
  const qs = new Uint8Array(nb * 16);
  const scales = new Float32Array(nb);
  for (let b = 0; b < nb; b++) {
    let amax = 0, maxv = 0;
    for (let i = b * 32; i < Math.min(n, b * 32 + 32); i++) {
      if (Math.abs(data[i]) > amax) { amax = Math.abs(data[i]); maxv = data[i]; }
    }
    const d = maxv / -8 || 1;
    scales[b] = d;
    for (let j = 0; j < 16; j++) {
      const lo = Math.max(0, Math.min(15, Math.round((data[b * 32 + j] || 0) / d) + 8));
      const hi = Math.max(0, Math.min(15, Math.round((data[b * 32 + j + 16] || 0) / d) + 8));
      qs[b * 16 + j] = lo | (hi << 4);
    }
  }
  return { qs, scales };
}
function dequantQ4(q, n) {
  const out = new Float32Array(n);
  for (let b = 0; b < n / 32; b++) for (let j = 0; j < 16; j++) {
    const byte = q.qs[b * 16 + j];
    out[b * 32 + j] = q.scales[b] * ((byte & 0xF) - 8);
    out[b * 32 + j + 16] = q.scales[b] * ((byte >> 4) - 8);
  }
  return out;
}
function dequantQ8(q, n) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) { const v = q.qs[i]; out[i] = q.scales[(i / 32) | 0] * (v > 127 ? v - 256 : v); }
  return out;
}

export async function gpuSelfTest(device) {
  const cfg = { hidden_size: 64, num_attention_heads: 4, num_key_value_heads: 2, head_dim: 16,
    intermediate_size: 128, vocab_size: 96, num_hidden_layers: 2, rms_norm_eps: 1e-5, rope_theta: 10000 };
  const { hidden_size: dim, num_attention_heads: nH, num_key_value_heads: nKV, head_dim: hd,
    intermediate_size: inter, vocab_size: vocab, num_hidden_layers: L, rms_norm_eps: eps, rope_theta: theta } = cfg;
  const qDim = nH * hd, kvDim = nKV * hd;
  let seed = 12345;
  const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff - 0.5; };
  const mat = (r, c, s) => { const a = new Float32Array(r * c); for (let i = 0; i < a.length; i++) a[i] = rnd() * s; return a; };
  const vec1 = (n) => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = 1 + rnd() * 0.2; return a; };
  const raw = { embed: mat(vocab, dim, 1), finalNorm: vec1(dim), layers: [] };
  for (let l = 0; l < L; l++) raw.layers.push({
    inNorm: vec1(dim), postNorm: vec1(dim),
    q: mat(qDim, dim, 0.3), k: mat(kvDim, dim, 0.3), v: mat(kvDim, dim, 0.3), o: mat(dim, qDim, 0.3),
    gate: mat(inter, dim, 0.3), up: mat(inter, dim, 0.3), down: mat(dim, inter, 0.3),
  });
  const ids = [3, 17, 42];

  // CPU reference on the (possibly dequantized) weights
  const cpuForward = (W) => {
    const rms = (x, w) => { let ss = 0; for (const v of x) ss += v * v; const inv = 1 / Math.sqrt(ss / x.length + eps); return x.map((v, i) => v * inv * w[i]); };
    const mm = (M, x, dO, dI) => { const o = new Float32Array(dO); for (let r = 0; r < dO; r++) { let a = 0; for (let c = 0; c < dI; c++) a += M[r * dI + c] * x[c]; o[r] = a; } return o; };
    const rope = (v, heads, pos) => { const half = hd / 2; for (let h = 0; h < heads; h++) for (let i = 0; i < half; i++) {
      const ang = pos * Math.pow(theta, -(2 * i) / hd), c = Math.cos(ang), s = Math.sin(ang);
      const a = v[h * hd + i], b = v[h * hd + i + half]; v[h * hd + i] = a * c - b * s; v[h * hd + i + half] = b * c + a * s; } };
    const kc = W.layers.map(() => []), vc = W.layers.map(() => []);
    let logits = null;
    ids.forEach((id, pos) => {
      const x = Float32Array.from(W.embed.subarray(id * dim, (id + 1) * dim));
      W.layers.forEach((Wl, li) => {
        const xn = rms(x, Wl.inNorm);
        const q = mm(Wl.q, xn, qDim, dim), k = mm(Wl.k, xn, kvDim, dim), v = mm(Wl.v, xn, kvDim, dim);
        rope(q, nH, pos); rope(k, nKV, pos); kc[li].push(k); vc[li].push(v);
        const att = new Float32Array(qDim), T = kc[li].length, sc = new Float32Array(T);
        for (let h = 0; h < nH; h++) {
          const kvH = Math.floor(h / (nH / nKV));
          for (let t = 0; t < T; t++) { let a = 0; for (let i = 0; i < hd; i++) a += q[h * hd + i] * kc[li][t][kvH * hd + i]; sc[t] = a / Math.sqrt(hd); }
          let mx = -Infinity; for (let t = 0; t < T; t++) mx = Math.max(mx, sc[t]);
          let sum = 0; for (let t = 0; t < T; t++) { sc[t] = Math.exp(sc[t] - mx); sum += sc[t]; }
          for (let t = 0; t < T; t++) for (let i = 0; i < hd; i++) att[h * hd + i] += (sc[t] / sum) * vc[li][t][kvH * hd + i];
        }
        const pr = mm(Wl.o, att, dim, qDim); for (let i = 0; i < dim; i++) x[i] += pr[i];
        const xn2 = rms(x, Wl.postNorm);
        const g = mm(Wl.gate, xn2, inter, dim), u = mm(Wl.up, xn2, inter, dim);
        for (let i = 0; i < inter; i++) g[i] = (g[i] / (1 + Math.exp(-g[i]))) * u[i];
        const d = mm(Wl.down, g, dim, inter); for (let i = 0; i < dim; i++) x[i] += d[i];
      });
      logits = mm(W.embed, rms(x, W.finalNorm), vocab, dim);
    });
    return logits;
  };

  const results = {};
  for (const kind of ["f32", "q8", "q4"]) {
    const conv = (m, r, c) => kind === "f32" ? { entry: { kind: "f32", data: m }, deq: m }
      : kind === "q8" ? (() => { const q = quantizeQ8(m); return { entry: { kind: "q8", ...q, shape: [r, c] }, deq: dequantQ8(q, m.length) }; })()
      : (() => { const q = quantizeQ4(m); return { entry: { kind: "q4", ...q, shape: [r, c] }, deq: dequantQ4(q, m.length) }; })();
    const E = conv(raw.embed, vocab, dim);
    const weights = { embed: E.entry, finalNorm: { kind: "f32", data: raw.finalNorm }, layers: [] };
    const deq = { embed: E.deq, finalNorm: raw.finalNorm, layers: [] };
    for (const l of raw.layers) {
      const Q = conv(l.q, qDim, dim), K = conv(l.k, kvDim, dim), V = conv(l.v, kvDim, dim), O = conv(l.o, dim, qDim);
      const G2 = conv(l.gate, inter, dim), U = conv(l.up, inter, dim), D = conv(l.down, dim, inter);
      weights.layers.push({ inNorm: { kind: "f32", data: l.inNorm }, postNorm: { kind: "f32", data: l.postNorm },
        q: Q.entry, k: K.entry, v: V.entry, o: O.entry, gate: G2.entry, up: U.entry, down: D.entry });
      deq.layers.push({ inNorm: l.inNorm, postNorm: l.postNorm, q: Q.deq, k: K.deq, v: V.deq, o: O.deq, gate: G2.deq, up: U.deq, down: D.deq });
    }
    const ref = cpuForward(deq);
    let gpu = null, err = null;
    try {
      const eng = await BelloEngine.create({ device, cfg, weights, maxSeq: 8 });
      for (const id of ids) gpu = await eng.forwardToken(id);
    } catch (e) { err = e.message; }
    let maxDiff = Infinity, nan = false;
    if (gpu) {
      maxDiff = 0; let scale = 1e-6;
      for (let i = 0; i < ref.length; i++) { if (!Number.isFinite(gpu[i])) nan = true; maxDiff = Math.max(maxDiff, Math.abs(gpu[i] - ref[i])); scale = Math.max(scale, Math.abs(ref[i])); }
      maxDiff /= scale;
    }
    results[kind] = { ok: !err && !nan && maxDiff < 2e-3, maxDiff: +maxDiff.toFixed(5), nan, err };
  }
  const ok = results.f32.ok && results.q8.ok && results.q4.ok;
  const detail = ["f32", "q8", "q4"].map(k => `${k}:${results[k].ok ? "ok" : (results[k].err ? "ERR " + results[k].err : results[k].nan ? "NaN" : "diff " + results[k].maxDiff)}`).join(" · ");
  return { ok, ...results, detail };
}

// ---------- per-kernel micro-tests ----------
// Builds a tiny qwen-shaped engine (QK-norm, head_dim != dim/nH) and checks
// every kernel individually against CPU math. Returns the first failing kernel.
export async function kernelMicroTests(device) {
  const cfg = { hidden_size: 128, num_attention_heads: 4, num_key_value_heads: 2, head_dim: 64,
    intermediate_size: 256, vocab_size: 512, num_hidden_layers: 1, rms_norm_eps: 1e-6, rope_theta: 1000000 };
  const dim = 128, nH = 4, nKV = 2, hd = 64, qDim = 256, kvDim = 128, inter = 256, vocab = 512, eps = 1e-6, theta = 1e6;
  let seed = 777;
  const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff - 0.5; };
  const mat = (r, c, s) => { const a = new Float32Array(r * c); for (let i = 0; i < a.length; i++) a[i] = rnd() * s; return a; };
  const vec1 = (n) => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = 1 + rnd() * 0.2; return a; };
  const q8 = (m, r, c) => { const q = quantizeQ8(m); return { entry: { kind: "q8", ...q, shape: [r, c] }, deq: dequantQ8(q, m.length) }; };
  const Wq = q8(mat(qDim, dim, 0.3), qDim, dim), Wk = q8(mat(kvDim, dim, 0.3), kvDim, dim), Wv = q8(mat(kvDim, dim, 0.3), kvDim, dim);
  const Wo = q8(mat(dim, qDim, 0.3), dim, qDim), Wg = q8(mat(inter, dim, 0.3), inter, dim), Wu = q8(mat(inter, dim, 0.3), inter, dim), Wd = q8(mat(dim, inter, 0.3), dim, inter);
  const inNorm = vec1(dim), postNorm = vec1(dim), qNorm = vec1(hd), kNorm = vec1(hd), finalNorm = vec1(dim);
  const E = q8(mat(vocab, dim, 1), vocab, dim);
  const weights = { embed: E.entry, finalNorm: { kind: "f32", data: finalNorm }, layers: [{
    inNorm: { kind: "f32", data: inNorm }, postNorm: { kind: "f32", data: postNorm },
    qNorm: { kind: "f32", data: qNorm }, kNorm: { kind: "f32", data: kNorm },
    q: Wq.entry, k: Wk.entry, v: Wv.entry, o: Wo.entry, gate: Wg.entry, up: Wu.entry, down: Wd.entry }] };
  const eng = await BelloEngine.create({ device, cfg, weights, maxSeq: 8 });
  const BG = eng.layerBGs[0];
  const stage = (n) => device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const run = (fn) => { const enc = device.createCommandEncoder(); const p = enc.beginComputePass(); fn(p); p.end(); device.queue.submit([enc.finish()]); };
  const rb = (buf, n) => eng._readback(buf, stage(n), n);
  const wr = (buf, arr) => device.queue.writeBuffer(buf, 0, arr);
  const cmp = (got, want) => {
    let md = 0, sc = 1e-6, nan = false;
    for (let i = 0; i < want.length; i++) { if (!Number.isFinite(got[i])) nan = true; md = Math.max(md, Math.abs(got[i] - want[i])); sc = Math.max(sc, Math.abs(want[i])); }
    return nan ? "NaN" : (md / sc < 2e-3 ? "ok" : "diff " + (md / sc).toFixed(4));
  };
  const cpuRms = (x, w, n, off = 0) => { let ss = 0; for (let i = 0; i < n; i++) ss += x[off + i] * x[off + i]; const inv = 1 / Math.sqrt(ss / n + eps); const o = new Float32Array(n); for (let i = 0; i < n; i++) o[i] = x[off + i] * inv * w[i]; return o; };
  const cpuMm = (M, x, dO, dI) => { const o = new Float32Array(dO); for (let r = 0; r < dO; r++) { let a = 0; for (let c = 0; c < dI; c++) a += M[r * dI + c] * x[c]; o[r] = a; } return o; };
  const results = {};
  const X = mat(1, dim, 1), Q = mat(1, qDim, 1), G2 = mat(1, inter, 1), U = mat(1, inter, 1), T2 = mat(1, dim, 1);
  try {
    // rmsnorm
    wr(eng.x, X); run((p) => eng._dispatch(p, "rmsnorm", BG.norm1, 256, 256));
    results.rmsnorm = cmp(await rb(eng.xn, dim), cpuRms(X, inNorm, dim));
    // matvec_q8 (q projection from xn)
    const xn = cpuRms(X, inNorm, dim);
    run((p) => eng._dispatchOp(p, BG.q));
    results.matvec_q8 = cmp(await rb(eng.q, qDim), cpuMm(Wq.deq, xn, qDim, dim));
    // head_norm on q
    wr(eng.q, Q); run((p) => eng._dispatch(p, "head_norm", BG.qNorm, nH, 32));
    const hnWant = new Float32Array(qDim);
    for (let h = 0; h < nH; h++) hnWant.set(cpuRms(Q, qNorm, hd, h * hd), h * hd);
    results.head_norm = cmp(await rb(eng.q, qDim), hnWant);
    // rope at pos 3
    wr(eng.q, Q); eng._setFrame(3, 4); run((p) => eng._dispatch(p, "rope", eng.bgRopeQ, nH * hd / 2));
    const rWant = Float32Array.from(Q);
    for (let h = 0; h < nH; h++) for (let i = 0; i < hd / 2; i++) {
      const ang = 3 * Math.pow(theta, -(2 * i) / hd), c = Math.cos(ang), s = Math.sin(ang);
      const a = Q[h * hd + i], b = Q[h * hd + i + hd / 2];
      rWant[h * hd + i] = a * c - b * s; rWant[h * hd + i + hd / 2] = b * c + a * s;
    }
    results.rope = cmp(await rb(eng.q, qDim), rWant);
    // softmax over seqLen 5 (scores laid out h*maxSeq + t)
    const S = new Float32Array(nH * 8); for (let i = 0; i < S.length; i++) S[i] = rnd() * 4;
    wr(eng.scores, S); eng._setFrame(4, 5); run((p) => eng._dispatch(p, "attn_softmax", BG.softmax, nH, 1));
    const smWant = Float32Array.from(S);
    for (let h = 0; h < nH; h++) { let mx = -Infinity; for (let t = 0; t < 5; t++) mx = Math.max(mx, S[h * 8 + t]); let sum = 0; for (let t = 0; t < 5; t++) { smWant[h * 8 + t] = Math.exp(S[h * 8 + t] - mx); sum += smWant[h * 8 + t]; } for (let t = 0; t < 5; t++) smWant[h * 8 + t] /= sum; }
    const smGot = await rb(eng.scores, nH * 8);
    let smOk = "ok"; for (let h = 0; h < nH; h++) for (let t = 0; t < 5; t++) { const g = smGot[h * 8 + t], w = smWant[h * 8 + t]; if (!Number.isFinite(g)) smOk = "NaN"; else if (Math.abs(g - w) > 2e-3 && smOk === "ok") smOk = "diff"; }
    results.attn_softmax = smOk;
    // silu_mul
    wr(eng.g, G2); wr(eng.u, U); run((p) => eng._dispatch(p, "silu_mul", eng.bgSilu, inter));
    results.silu_mul = cmp(await rb(eng.g, inter), G2.map((g, i) => (g / (1 + Math.exp(-g))) * U[i]));
    // add_res
    wr(eng.x, X); wr(eng.tmpDim, T2); run((p) => eng._dispatch(p, "add_res", eng.bgAddTmp, dim));
    results.add_res = cmp(await rb(eng.x, dim), X.map((v, i) => v + T2[i]));
    // full attention block end-to-end at pos 0 via forwardToken (covers scores/attn_out/kv copy)
    eng.pos = 0; const lg = await eng.forwardToken(5);
    results.full_layer = lg.some((v) => !Number.isFinite(v)) ? "NaN" : "ok";
  } catch (e) { results.error = e.message; }
  const firstFail = Object.entries(results).find(([, v]) => v !== "ok");
  return { ok: !firstFail, firstFail: firstFail ? firstFail[0] + ": " + firstFail[1] : null,
    detail: Object.entries(results).map(([k, v]) => `${k}:${v}`).join(" · ") };
}
