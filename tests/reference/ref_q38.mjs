// Qwen3.8-27B stateful CPU reference: multi-token, layers 0..maxLayer.
// Validates DeltaNet state carry + RoPE(pos>0) against llama-eval-callback.
// usage: node ref_q38.mjs "<prompt>" <maxLayer>
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { parseGGUFHeader, dequantF32, tokenizerFromGGUF } from "../engine/gguf.js";
import { makeTokenizer } from "../engine/engine.js";

const DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), "q38");
const fd = fs.openSync(path.join(DIR, "model.gguf"), "r");
const headBuf = Buffer.alloc(16 * 1024 * 1024);
fs.readSync(fd, headBuf, 0, headBuf.length, 0);
const G = parseGGUFHeader(headBuf.buffer.slice(headBuf.byteOffset, headBuf.byteOffset + headBuf.length));
const T = (name) => {
  const info = G.tensors[name];
  if (!info) throw new Error("missing " + name);
  const raw = Buffer.alloc(info.byteLength);
  fs.readSync(fd, raw, 0, info.byteLength, info.byteOffset);
  return dequantF32(info, new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength));
};

const M = G.meta;
const dModel = M["qwen35.embedding_length"];
const eps = M["qwen35.attention.layer_norm_rms_epsilon"];
const dConv = M["qwen35.ssm.conv_kernel"];
const dState = M["qwen35.ssm.state_size"];
const nKHeads = M["qwen35.ssm.group_count"];
const nVHeads = M["qwen35.ssm.time_step_rank"];
const dInner = M["qwen35.ssm.inner_size"];
const keyDim = dState * nKHeads;
const convDim = keyDim * 2 + dInner;
const nH = M["qwen35.attention.head_count"];
const nKV = M["qwen35.attention.head_count_kv"];
const hd = M["qwen35.attention.key_length"];
const nRot = M["qwen35.rope.dimension_count"];
const ropeTheta = M["qwen35.rope.freq_base"];
const ffnDim = M["qwen35.feed_forward_length"];

const matmul = (W, x, dOut, dIn) => {
  const out = new Float32Array(dOut);
  for (let r = 0; r < dOut; r++) {
    let acc = 0;
    const off = r * dIn;
    for (let c = 0; c < dIn; c++) acc += W[off + c] * x[c];
    out[r] = acc;
  }
  return out;
};
const rmsnorm = (x, w, n, off = 0) => {
  let ss = 0;
  for (let i = 0; i < n; i++) ss += x[off + i] * x[off + i];
  const inv = 1 / Math.sqrt(ss / n + eps);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = x[off + i] * inv * w[i];
  return out;
};
const silu = (v) => v / (1 + Math.exp(-v));
const sigmoid = (v) => 1 / (1 + Math.exp(-v));
const softplus = (v) => v > 20 ? v : Math.log1p(Math.exp(v));
const l2n = (vec, off, n) => {
  let ss = 0;
  for (let i = 0; i < n; i++) ss += vec[off + i] * vec[off + i];
  const inv = 1 / Math.max(Math.sqrt(ss), eps);
  for (let i = 0; i < n; i++) vec[off + i] *= inv;
};
const ropeNeox = (vec, heads, pos) => {
  const half = nRot / 2;
  for (let h = 0; h < heads; h++) {
    const off = h * hd;
    for (let i = 0; i < half; i++) {
      const ang = pos * Math.pow(ropeTheta, -(2 * i) / nRot);
      const c = Math.cos(ang), s = Math.sin(ang);
      const a = vec[off + i], b = vec[off + i + half];
      vec[off + i] = a * c - b * s;
      vec[off + i + half] = b * c + a * s;
    }
  }
};
const p8 = (a) => Array.from(a.slice(0, 8)).map(v => +v.toFixed(5));

// weight cache so multi-token runs do not re-read the file per token
const wcache = new Map();
const W = (name) => {
  if (!wcache.has(name)) wcache.set(name, T(name));
  return wcache.get(name);
};

function deltaLayer(li, x, st) {
  const p = "blk." + li + ".";
  const xn = rmsnorm(x, W(p + "attn_norm.weight"), dModel);
  const qkv = matmul(W(p + "attn_qkv.weight"), xn, convDim, dModel);
  const z = matmul(W(p + "attn_gate.weight"), xn, dInner, dModel);
  const beta = matmul(W(p + "ssm_beta.weight"), xn, nVHeads, dModel).map(sigmoid);
  const alpha = matmul(W(p + "ssm_alpha.weight"), xn, nVHeads, dModel);
  const dtBias = W(p + "ssm_dt.bias"), ssmA = W(p + "ssm_a");
  const gate = new Float32Array(nVHeads);
  for (let h = 0; h < nVHeads; h++) gate[h] = softplus(alpha[h] + dtBias[h]) * ssmA[h];

  const conv = W(p + "ssm_conv1d.weight");
  const K1 = dConv - 1;
  const co = new Float32Array(convDim);
  for (let c = 0; c < convDim; c++) {
    let acc = conv[c * dConv + K1] * qkv[c];
    for (let j = 0; j < K1; j++) acc += conv[c * dConv + j] * st.conv[c * K1 + j];
    co[c] = silu(acc);
    for (let j = 0; j < K1 - 1; j++) st.conv[c * K1 + j] = st.conv[c * K1 + j + 1];
    st.conv[c * K1 + K1 - 1] = qkv[c];
  }
  const q = co.slice(0, keyDim), k = co.slice(keyDim, keyDim * 2), v = co.slice(keyDim * 2);
  for (let h = 0; h < nKHeads; h++) { l2n(q, h * dState, dState); l2n(k, h * dState, dState); }

  const scale = 1 / Math.sqrt(dState);
  const out = new Float32Array(dInner);
  for (let h = 0; h < nVHeads; h++) {
    const kh = h % nKHeads;
    const kOff = kh * dState, vOff = h * dState, Sb = h * dState * dState;
    const decay = Math.exp(gate[h]);
    for (let e = 0; e < dState * dState; e++) st.S[Sb + e] *= decay;
    const vhat = new Float32Array(dState);
    for (let i = 0; i < dState; i++) {
      const ki = k[kOff + i];
      if (ki === 0) continue;
      const row = Sb + i * dState;
      for (let j = 0; j < dState; j++) vhat[j] += st.S[row + j] * ki;
    }
    const d = new Float32Array(dState);
    for (let j = 0; j < dState; j++) d[j] = (v[vOff + j] - vhat[j]) * beta[h];
    for (let i = 0; i < dState; i++) {
      const ki = k[kOff + i], row = Sb + i * dState;
      for (let j = 0; j < dState; j++) st.S[row + j] += ki * d[j];
    }
    for (let i = 0; i < dState; i++) {
      const qi = q[kOff + i] * scale, row = Sb + i * dState;
      for (let j = 0; j < dState; j++) out[vOff + j] += st.S[row + j] * qi;
    }
  }

  const ssmNorm = W(p + "ssm_norm.weight");
  const gated = new Float32Array(dInner);
  for (let h = 0; h < nVHeads; h++) {
    const norm = rmsnorm(out, ssmNorm, dState, h * dState);
    for (let j = 0; j < dState; j++) gated[h * dState + j] = norm[j] * silu(z[h * dState + j]);
  }
  console.log("final_output-" + li + ":", p8(gated));
  return matmul(W(p + "ssm_out.weight"), gated, dModel, dInner);
}

function fullLayer(li, x, st, pos) {
  const p = "blk." + li + ".";
  const xn = rmsnorm(x, W(p + "attn_norm.weight"), dModel);
  const qFull = matmul(W(p + "attn_q.weight"), xn, nH * hd * 2, dModel);
  const kRaw = matmul(W(p + "attn_k.weight"), xn, nKV * hd, dModel);
  const v = matmul(W(p + "attn_v.weight"), xn, nKV * hd, dModel);
  const qNormW = W(p + "attn_q_norm.weight"), kNormW = W(p + "attn_k_norm.weight");

  const q = new Float32Array(nH * hd), g = new Float32Array(nH * hd);
  for (let h = 0; h < nH; h++) {
    q.set(qFull.subarray(h * 2 * hd, h * 2 * hd + hd), h * hd);
    g.set(qFull.subarray(h * 2 * hd + hd, (h + 1) * 2 * hd), h * hd);
  }
  const qn = new Float32Array(nH * hd), kn = new Float32Array(nKV * hd);
  for (let h = 0; h < nH; h++) qn.set(rmsnorm(q, qNormW, hd, h * hd), h * hd);
  for (let h = 0; h < nKV; h++) kn.set(rmsnorm(kRaw, kNormW, hd, h * hd), h * hd);
  ropeNeox(qn, nH, pos);
  ropeNeox(kn, nKV, pos);
  console.log("Qcur_roped-" + li + ":", p8(qn));
  st.k.push(Float32Array.from(kn));
  st.v.push(Float32Array.from(v));

  const group = nH / nKV, seqLen = st.k.length;
  const attn = new Float32Array(nH * hd);
  const sc = new Float32Array(seqLen);
  for (let h = 0; h < nH; h++) {
    const kvH = Math.floor(h / group);
    for (let t = 0; t < seqLen; t++) {
      let acc = 0;
      const kt = st.k[t];
      for (let i = 0; i < hd; i++) acc += qn[h * hd + i] * kt[kvH * hd + i];
      sc[t] = acc / Math.sqrt(hd);
    }
    let mx = -Infinity;
    for (let t = 0; t < seqLen; t++) if (sc[t] > mx) mx = sc[t];
    let sum = 0;
    for (let t = 0; t < seqLen; t++) { sc[t] = Math.exp(sc[t] - mx); sum += sc[t]; }
    for (let t = 0; t < seqLen; t++) {
      const wt = sc[t] / sum, vt = st.v[t];
      for (let i = 0; i < hd; i++) attn[h * hd + i] += wt * vt[kvH * hd + i];
    }
  }
  for (let i = 0; i < attn.length; i++) attn[i] *= sigmoid(g[i]);
  console.log("attn_gated-" + li + ":", p8(attn));
  return matmul(W(p + "attn_output.weight"), attn, dModel, nH * hd);
}

function ffn(li, x) {
  const p = "blk." + li + ".";
  const xn = rmsnorm(x, W(p + "post_attention_norm.weight"), dModel);
  const gg = matmul(W(p + "ffn_gate.weight"), xn, ffnDim, dModel);
  const u = matmul(W(p + "ffn_up.weight"), xn, ffnDim, dModel);
  for (let i = 0; i < ffnDim; i++) gg[i] = silu(gg[i]) * u[i];
  return matmul(W(p + "ffn_down.weight"), gg, dModel, ffnDim);
}

// ---- main ----
const prompt = process.argv[2] || "The";
const maxLayer = parseInt(process.argv[3] || "0", 10);
const tok = makeTokenizer(tokenizerFromGGUF(M));
const ids = tok.encode(prompt);
console.log("prompt ids:", JSON.stringify(ids), "layers 0.." + maxLayer);

const states = [];
for (let li = 0; li <= maxLayer; li++)
  states.push(li % 4 === 3
    ? { k: [], v: [] }
    : { conv: new Float32Array(convDim * (dConv - 1)), S: new Float32Array(nVHeads * dState * dState) });

const embedInfo = G.tensors["token_embd.weight"];
const rowBytes = (dModel / 32) * 18;
for (let ti = 0; ti < ids.length; ti++) {
  console.log("\n######## token " + ti + " (id " + ids[ti] + ") ########");
  const rowRaw = Buffer.alloc(rowBytes);
  fs.readSync(fd, rowRaw, 0, rowBytes, embedInfo.byteOffset + ids[ti] * rowBytes);
  let x = dequantF32({ ggmlType: 2, nElems: dModel, shape: [dModel] }, new Uint8Array(rowRaw.buffer, rowRaw.byteOffset, rowBytes));
  for (let li = 0; li <= maxLayer; li++) {
    const isFull = li % 4 === 3;
    const attnOut = isFull ? fullLayer(li, x, states[li], ti) : deltaLayer(li, x, states[li]);
    const x2 = Float32Array.from(x);
    for (let i = 0; i < dModel; i++) x2[i] += attnOut[i];
    const f = ffn(li, x2);
    for (let i = 0; i < dModel; i++) x2[i] += f[i];
    console.log("layer_out-" + li + ":", p8(x2));
    x = x2;
  }
}
