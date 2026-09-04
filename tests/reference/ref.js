// bello engine — reference implementation (Node, CPU, fp32).
// Llama-architecture forward pass over safetensors weights. This is the golden
// reference the WebGPU kernels must match, and the spec for the layer-split:
// everything between `embed` and `head` is what gets sharded across peers.
//
// usage: node ref.js "prompt text" [numTokens] [--golden out.json]
const fs = require("fs");
const path = require("path");

const MODEL_DIR = path.join(__dirname, "model");

// ---------- safetensors ----------
function loadSafetensors(file) {
  const buf = fs.readFileSync(file);
  const headerLen = Number(buf.readBigUInt64LE(0));
  const header = JSON.parse(buf.subarray(8, 8 + headerLen).toString("utf8"));
  const base = 8 + headerLen;
  const tensors = {};
  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    tensors[name] = { ...info, buf, base };
  }
  return tensors;
}

function tensorF32(t) {
  // decode BF16 or F32 to Float32Array
  const [b0, b1] = t.data_offsets;
  const bytes = t.buf.subarray(t.base + b0, t.base + b1);
  const n = t.shape.reduce((a, b) => a * b, 1);
  if (t.dtype === "F32")
    return new Float32Array(bytes.buffer, bytes.byteOffset, n);
  if (t.dtype === "BF16") {
    const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, n);
    const out = new Float32Array(n);
    const u32 = new Uint32Array(1);
    const f32 = new Float32Array(u32.buffer);
    for (let i = 0; i < n; i++) { u32[0] = u16[i] << 16; out[i] = f32[0]; }
    return out;
  }
  throw new Error("unsupported dtype " + t.dtype);
}

// ---------- byte-level BPE tokenizer ----------
function makeTokenizer(tokJsonPath) {
  const tj = JSON.parse(fs.readFileSync(tokJsonPath, "utf8"));
  const vocab = tj.model.vocab;                       // token string -> id
  const idToTok = Object.fromEntries(Object.entries(vocab).map(([t, i]) => [i, t]));
  const ranks = new Map();
  tj.model.merges.forEach((m, i) => {
    const pair = Array.isArray(m) ? m.join(" ") : m;
    ranks.set(pair, i);
  });

  // GPT-2 byte<->unicode table
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const byteToChar = {}, charToByte = {};
  bs.forEach((b, i) => { byteToChar[b] = String.fromCharCode(cs[i]); charToByte[String.fromCharCode(cs[i])] = b; });

  const pat = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

  function bpe(word) {
    let parts = [...word];
    while (parts.length > 1) {
      let best = null, bestRank = Infinity, bestI = -1;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = ranks.get(parts[i] + " " + parts[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best === null) break;
      parts = [...parts.slice(0, best), parts[best] + parts[best + 1], ...parts.slice(best + 2)];
    }
    return parts;
  }

  function encode(text) {
    const ids = [];
    for (const piece of text.match(pat) || []) {
      const bytes = Buffer.from(piece, "utf8");
      let word = "";
      for (const b of bytes) word += byteToChar[b];
      for (const tok of bpe(word)) {
        if (vocab[tok] === undefined) throw new Error("unknown token piece: " + tok);
        ids.push(vocab[tok]);
      }
    }
    return ids;
  }

  function decode(ids) {
    const bytes = [];
    for (const id of ids) {
      const tok = idToTok[id];
      if (tok === undefined) continue;
      for (const ch of tok) {
        const b = charToByte[ch];
        if (b !== undefined) bytes.push(b);
      }
    }
    return Buffer.from(bytes).toString("utf8");
  }

  return { encode, decode, vocab };
}

// ---------- math ----------
function rmsnorm(x, w, eps) {
  const n = x.length;
  let ss = 0;
  for (let i = 0; i < n; i++) ss += x[i] * x[i];
  const inv = 1 / Math.sqrt(ss / n + eps);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
  return out;
}

// y = W x, W shape [dOut, dIn] row-major
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
  // llama-style: pairs (i, i + headDim/2) within each head
  const half = headDim / 2;
  for (let h = 0; h < nHeads; h++) {
    const off = h * headDim;
    for (let i = 0; i < half; i++) {
      const freq = Math.pow(theta, -(2 * i) / headDim);
      const ang = pos * freq;
      const cos = Math.cos(ang), sin = Math.sin(ang);
      const a = vec[off + i], b = vec[off + i + half];
      vec[off + i] = a * cos - b * sin;
      vec[off + i + half] = b * cos + a * sin;
    }
  }
}

function softmaxInPlace(x) {
  let max = -Infinity;
  for (const v of x) if (v > max) max = v;
  let sum = 0;
  for (let i = 0; i < x.length; i++) { x[i] = Math.exp(x[i] - max); sum += x[i]; }
  for (let i = 0; i < x.length; i++) x[i] /= sum;
}

// ---------- model ----------
function loadModel() {
  const cfg = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, "config.json"), "utf8"));
  const T = loadSafetensors(path.join(MODEL_DIR, "model.safetensors"));
  const get = (name) => tensorF32(T[name]);
  const L = cfg.num_hidden_layers;
  const layers = [];
  for (let i = 0; i < L; i++) {
    const p = `model.layers.${i}.`;
    layers.push({
      inNorm: get(p + "input_layernorm.weight"),
      q: get(p + "self_attn.q_proj.weight"),
      k: get(p + "self_attn.k_proj.weight"),
      v: get(p + "self_attn.v_proj.weight"),
      o: get(p + "self_attn.o_proj.weight"),
      postNorm: get(p + "post_attention_layernorm.weight"),
      gate: get(p + "mlp.gate_proj.weight"),
      up: get(p + "mlp.up_proj.weight"),
      down: get(p + "mlp.down_proj.weight"),
    });
  }
  return {
    cfg,
    embed: get("model.embed_tokens.weight"),  // [vocab, dim]; also the (tied) lm_head
    finalNorm: get("model.norm.weight"),
    layers,
  };
}

function makeState(model) {
  const { cfg } = model;
  const nKV = cfg.num_key_value_heads;
  const headDim = cfg.hidden_size / cfg.num_attention_heads;
  return {
    pos: 0,
    // per layer: array over positions of Float32Array(nKV*headDim)
    kCache: model.layers.map(() => []),
    vCache: model.layers.map(() => []),
    nKV, headDim,
  };
}

// run one token through layers [lo, hi); returns hidden state
function runLayers(model, state, x, lo, hi, capture) {
  const { cfg } = model;
  const dim = cfg.hidden_size;
  const nH = cfg.num_attention_heads;
  const { nKV, headDim } = state;
  const group = nH / nKV;
  const kvDim = nKV * headDim;
  const pos = state.pos;

  for (let li = lo; li < hi; li++) {
    const W = model.layers[li];
    // attention
    const xn = rmsnorm(x, W.inNorm, cfg.rms_norm_eps);
    const q = matmul(W.q, xn, dim, dim);
    const k = matmul(W.k, xn, kvDim, dim);
    const v = matmul(W.v, xn, kvDim, dim);
    rope(q, nH, headDim, pos, cfg.rope_theta);
    rope(k, nKV, headDim, pos, cfg.rope_theta);
    state.kCache[li].push(k);
    state.vCache[li].push(v);

    const attnOut = new Float32Array(dim);
    const seqLen = state.kCache[li].length;
    const scores = new Float32Array(seqLen);
    for (let h = 0; h < nH; h++) {
      const kvH = Math.floor(h / group);
      const qOff = h * headDim, kvOff = kvH * headDim;
      for (let t = 0; t < seqLen; t++) {
        const kt = state.kCache[li][t];
        let acc = 0;
        for (let i = 0; i < headDim; i++) acc += q[qOff + i] * kt[kvOff + i];
        scores[t] = acc / Math.sqrt(headDim);
      }
      softmaxInPlace(scores);
      for (let t = 0; t < seqLen; t++) {
        const vt = state.vCache[li][t], s = scores[t];
        for (let i = 0; i < headDim; i++) attnOut[qOff + i] += s * vt[kvOff + i];
      }
    }
    const attnProj = matmul(W.o, attnOut, dim, dim);
    for (let i = 0; i < dim; i++) x[i] += attnProj[i];

    // mlp
    const xn2 = rmsnorm(x, W.postNorm, cfg.rms_norm_eps);
    const g = matmul(W.gate, xn2, cfg.intermediate_size, dim);
    const u = matmul(W.up, xn2, cfg.intermediate_size, dim);
    for (let i = 0; i < g.length; i++) g[i] = (g[i] / (1 + Math.exp(-g[i]))) * u[i]; // silu(g)*u
    const d = matmul(W.down, g, dim, cfg.intermediate_size);
    for (let i = 0; i < dim; i++) x[i] += d[i];

    if (capture) capture[li] = Float32Array.from(x);
  }
  return x;
}

function forward(model, state, tokenId, capture) {
  const { cfg } = model;
  const dim = cfg.hidden_size;
  const x = Float32Array.from(model.embed.subarray(tokenId * dim, (tokenId + 1) * dim));
  runLayers(model, state, x, 0, cfg.num_hidden_layers, capture);
  const xn = rmsnorm(x, model.finalNorm, cfg.rms_norm_eps);
  const logits = matmul(model.embed, xn, cfg.vocab_size, dim); // tied lm_head
  state.pos++;
  return logits;
}

function argmax(a) {
  let bi = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i;
  return bi;
}

// ---------- main ----------
function main() {
  const args = process.argv.slice(2);
  const goldenIdx = args.indexOf("--golden");
  let goldenPath = null;
  if (goldenIdx >= 0) { goldenPath = args[goldenIdx + 1]; args.splice(goldenIdx, 2); }
  const prompt = args[0] || "The capital of France is";
  const numTokens = parseInt(args[1] || "10", 10);

  console.error("loading model…");
  const t0 = Date.now();
  const model = loadModel();
  console.error(`loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  const tok = makeTokenizer(path.join(MODEL_DIR, "tokenizer.json"));
  const state = makeState(model);

  const ids = tok.encode(prompt);
  console.error("prompt tokens:", JSON.stringify(ids));

  const golden = goldenPath ? { prompt, ids, perLayer: null, logitsTop: null, generated: [] } : null;

  let logits = null;
  for (let i = 0; i < ids.length; i++) {
    const capture = golden && i === ids.length - 1 ? {} : null;
    logits = forward(model, state, ids[i], capture);
    if (capture) golden.perLayer = Object.fromEntries(
      Object.entries(capture).map(([li, v]) => [li, Array.from(v.subarray(0, 8)).map(x => +x.toFixed(6))]));
  }

  if (golden) {
    const top = [...logits.keys()].sort((a, b) => logits[b] - logits[a]).slice(0, 5);
    golden.logitsTop = top.map(i => [i, +logits[i].toFixed(4)]);
  }

  const out = [];
  process.stderr.write("generating: ");
  for (let i = 0; i < numTokens; i++) {
    const next = argmax(logits);
    out.push(next);
    if (golden) golden.generated.push(next);
    process.stderr.write(".");
    logits = forward(model, state, next);
  }
  process.stderr.write("\n");
  console.log("completion:", JSON.stringify(tok.decode(out)));
  if (golden) {
    fs.writeFileSync(goldenPath, JSON.stringify(golden, null, 1));
    console.error("golden written to", goldenPath);
  }
}

main();
