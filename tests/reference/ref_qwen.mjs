// Qwen3 (dense) CPU reference from a GGUF file. Golden source for the
// quantized WebGPU path. Deltas vs ref.js (llama): decoupled head_dim
// (qDim = nH*headDim != hidden), per-head QK-norm before rope.
// usage: node ref_qwen.mjs "prompt" [numTokens] [--golden out.json]
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { parseGGUFHeader, dequantF32, ggmlLayerNames, GGML_EMBED, GGML_FINAL_NORM, GGML_OUTPUT } from "../engine/gguf.js";

const DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), process.env.QWEN_DIR || "qwen");

// ---- tokenizer: reuse ref.js's BPE via a tiny re-implementation import ----
import { makeTokenizer } from "../engine/engine.js";

function rmsnorm(x, w, eps, out = new Float32Array(x.length)) {
  let ss = 0;
  for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
  const inv = 1 / Math.sqrt(ss / x.length + eps);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * inv * w[i];
  return out;
}
function matmul(W, x, dOut, dIn) {
  const out = new Float32Array(dOut);
  for (let r = 0; r < dOut; r++) {
    let acc = 0;
    const off = r * dIn;
    for (let c = 0; c < dIn; c++) acc += W[off + c] * x[c];
    out[r] = acc;
  }
  return out;
}
function rope(vec, nHeads, headDim, pos, theta) {
  const half = headDim / 2;
  for (let h = 0; h < nHeads; h++) {
    const off = h * headDim;
    for (let i = 0; i < half; i++) {
      const ang = pos * Math.pow(theta, -(2 * i) / headDim);
      const c = Math.cos(ang), s = Math.sin(ang);
      const a = vec[off + i], b = vec[off + i + half];
      vec[off + i] = a * c - b * s;
      vec[off + i + half] = b * c + a * s;
    }
  }
}
function headNorm(vec, nHeads, headDim, w, eps) {
  // per-head rmsnorm with shared [headDim] weight (qwen3 QK-norm)
  for (let h = 0; h < nHeads; h++) {
    const off = h * headDim;
    let ss = 0;
    for (let i = 0; i < headDim; i++) ss += vec[off + i] * vec[off + i];
    const inv = 1 / Math.sqrt(ss / headDim + eps);
    for (let i = 0; i < headDim; i++) vec[off + i] *= inv * w[i];
  }
}
function softmaxInPlace(x) {
  let mx = -Infinity;
  for (const v of x) if (v > mx) mx = v;
  let sum = 0;
  for (let i = 0; i < x.length; i++) { x[i] = Math.exp(x[i] - mx); sum += x[i]; }
  for (let i = 0; i < x.length; i++) x[i] /= sum;
}
const argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };

// ---- load ----
const cfg = JSON.parse(fs.readFileSync(path.join(DIR, "config.json"), "utf8"));
const raw = fs.readFileSync(path.join(DIR, process.env.MODEL_GGUF || "model.gguf"));
const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
const G = parseGGUFHeader(buf);
const bytesOf = (info) => new Uint8Array(buf, info.byteOffset, info.byteLength);
const F32 = (name) => {
  const info = G.tensors[name];
  if (!info) throw new Error("missing tensor " + name);
  return dequantF32(info, bytesOf(info));
};

console.error("gguf arch:", G.meta["general.architecture"], "| tensors:", Object.keys(G.tensors).length);
const L = cfg.num_hidden_layers;
const dim = cfg.hidden_size, nH = cfg.num_attention_heads, nKV = cfg.num_key_value_heads;
const headDim = cfg.head_dim, qDim = nH * headDim, kvDim = nKV * headDim;
const inter = cfg.intermediate_size, eps = cfg.rms_norm_eps, theta = cfg.rope_theta;

const layers = [];
for (let i = 0; i < L; i++) {
  const N = ggmlLayerNames(i);
  layers.push({
    inNorm: F32(N.inNorm), q: F32(N.q), k: F32(N.k), v: F32(N.v), o: F32(N.o),
    qNorm: F32(N.qNorm), kNorm: F32(N.kNorm),
    postNorm: F32(N.postNorm), gate: F32(N.gate), up: F32(N.up), down: F32(N.down),
  });
}
const embed = F32(GGML_EMBED);
const finalNorm = F32(GGML_FINAL_NORM);
const lmHead = G.tensors[GGML_OUTPUT] ? F32(GGML_OUTPUT) : embed;
console.error("weights dequantized");

const kCache = layers.map(() => []), vCache = layers.map(() => []);
let pos = 0;

function forward(tokenId, capture) {
  const x = Float32Array.from(embed.subarray(tokenId * dim, (tokenId + 1) * dim));
  const group = nH / nKV;
  for (let li = 0; li < L; li++) {
    const W = layers[li];
    const xn = rmsnorm(x, W.inNorm, eps);
    const q = matmul(W.q, xn, qDim, dim);
    const k = matmul(W.k, xn, kvDim, dim);
    const v = matmul(W.v, xn, kvDim, dim);
    headNorm(q, nH, headDim, W.qNorm, eps);
    headNorm(k, nKV, headDim, W.kNorm, eps);
    rope(q, nH, headDim, pos, theta);
    rope(k, nKV, headDim, pos, theta);
    kCache[li].push(k); vCache[li].push(v);

    const attnOut = new Float32Array(qDim);
    const seqLen = kCache[li].length;
    const scores = new Float32Array(seqLen);
    for (let h = 0; h < nH; h++) {
      const kvH = Math.floor(h / group);
      const qOff = h * headDim, kvOff = kvH * headDim;
      for (let t = 0; t < seqLen; t++) {
        let acc = 0;
        const kt = kCache[li][t];
        for (let i = 0; i < headDim; i++) acc += q[qOff + i] * kt[kvOff + i];
        scores[t] = acc / Math.sqrt(headDim);
      }
      softmaxInPlace(scores);
      for (let t = 0; t < seqLen; t++) {
        const vt = vCache[li][t], s = scores[t];
        for (let i = 0; i < headDim; i++) attnOut[qOff + i] += s * vt[kvOff + i];
      }
    }
    const proj = matmul(W.o, attnOut, dim, qDim);
    for (let i = 0; i < dim; i++) x[i] += proj[i];

    const xn2 = rmsnorm(x, W.postNorm, eps);
    const g = matmul(W.gate, xn2, inter, dim);
    const u = matmul(W.up, xn2, inter, dim);
    for (let i = 0; i < inter; i++) g[i] = (g[i] / (1 + Math.exp(-g[i]))) * u[i];
    const d = matmul(W.down, g, dim, inter);
    for (let i = 0; i < dim; i++) x[i] += d[i];
    if (capture) capture[li] = Array.from(x.subarray(0, 8)).map((v2) => +v2.toFixed(6));
  }
  const xn = rmsnorm(x, finalNorm, eps);
  const logits = matmul(lmHead, xn, cfg.vocab_size, dim);
  pos++;
  return logits;
}

// ---- main ----
const args = process.argv.slice(2);
const gi = args.indexOf("--golden");
let goldenPath = null;
if (gi >= 0) { goldenPath = args[gi + 1]; args.splice(gi, 2); }
const prompt = args[0] || "The capital of France is";
const numTokens = parseInt(args[1] || "8", 10);

const tok = makeTokenizer(JSON.parse(fs.readFileSync(path.join(DIR, "tokenizer.json"), "utf8")));
const ids = tok.encode(prompt);
console.error("prompt ids:", JSON.stringify(ids));

const golden = goldenPath ? { prompt, ids, perLayer: null, logitsTop: null, generated: [] } : null;
let logits = null;
for (let i = 0; i < ids.length; i++) {
  const cap = golden && i === ids.length - 1 ? {} : null;
  logits = forward(ids[i], cap);
  if (cap) golden.perLayer = cap;
}
if (golden) {
  const top = [...logits.keys()].sort((a, b) => logits[b] - logits[a]).slice(0, 5);
  golden.logitsTop = top.map((i) => [i, +logits[i].toFixed(4)]);
}
const out = [];
for (let i = 0; i < numTokens; i++) {
  const next = argmax(logits);
  out.push(next);
  if (golden) golden.generated.push(next);
  process.stderr.write(".");
  logits = forward(next);
}
console.error("");
console.log("completion:", JSON.stringify(tok.decode(out)));
if (golden) {
  fs.writeFileSync(goldenPath, JSON.stringify(golden, null, 1));
  console.error("golden written to", goldenPath);
}
