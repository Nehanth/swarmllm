// Qwen3.8-27B single-layer CPU reference (linear attention / Gated DeltaNet).
// Ported line-by-line from llama.cpp qwen35.cpp + delta-net-base.cpp.
// Purpose: validate the math translation against llama-eval-callback dumps,
// then serve as the golden reference for the WGSL port.
// usage: node ref_q38_layer.mjs <tokenId> [layerIdx]
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { parseGGUFHeader, dequantF32 } from "./gguf.js";

const DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), "q38");
const FILE = path.join(DIR, "model.gguf");

// ---- fd-based tensor access (15GB file: never read whole) ----
const fd = fs.openSync(FILE, "r");
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

// ---- hparams from metadata ----
const M = G.meta;
const dModel = M["qwen35.embedding_length"];        // 5120
const eps = M["qwen35.attention.layer_norm_rms_epsilon"]; // ~1e-6
const dConv = M["qwen35.ssm.conv_kernel"];          // 4
const dState = M["qwen35.ssm.state_size"];          // 128 (head dim, k and v)
const nKHeads = M["qwen35.ssm.group_count"];        // 16
const nVHeads = M["qwen35.ssm.time_step_rank"];     // 48
const dInner = M["qwen35.ssm.inner_size"];          // 6144 = 48*128
const keyDim = dState * nKHeads;                    // 2048
console.log(`hparams: d=${dModel} conv=${dConv} S=${dState} Hk=${nKHeads} Hv=${nVHeads}`);

// ---- math ----
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
const rmsnorm = (x, w, n = x.length, off = 0, wOff = 0) => {
  let ss = 0;
  for (let i = 0; i < n; i++) ss += x[off + i] * x[off + i];
  const inv = 1 / Math.sqrt(ss / n + eps);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = x[off + i] * inv * w[wOff + i];
  return out;
};
const silu = (v) => v / (1 + Math.exp(-v));
const sigmoid = (v) => 1 / (1 + Math.exp(-v));
const softplus = (v) => v > 20 ? v : Math.log1p(Math.exp(v));
// ggml_l2_norm: x / max(sqrt(sum x^2), eps)
const l2normHead = (vec, off, n) => {
  let ss = 0;
  for (let i = 0; i < n; i++) ss += vec[off + i] * vec[off + i];
  const inv = 1 / Math.max(Math.sqrt(ss), eps);
  for (let i = 0; i < n; i++) vec[off + i] *= inv;
};
const p8 = (a, off = 0) => Array.from(a.slice(off, off + 8)).map(v => +v.toFixed(5));

// ---- linear attention layer (single token, zero states) ----
function linearAttnLayer(li, x /* Float32Array dModel, post-norm input NOT applied */) {
  const p = `blk.${li}.`;
  const attnNorm = T(p + "attn_norm.weight");
  const wqkv = T(p + "attn_qkv.weight");        // [10240, 5120]
  const wz = T(p + "attn_gate.weight");         // [6144, 5120]
  const wBeta = T(p + "ssm_beta.weight");       // [48, 5120]
  const wAlpha = T(p + "ssm_alpha.weight");     // [48, 5120]
  const dtBias = T(p + "ssm_dt.bias");          // [48]
  const ssmA = T(p + "ssm_a");                  // [48]
  const conv = T(p + "ssm_conv1d.weight");      // [10240, 4] torch order -> [ch][tap]
  const ssmNorm = T(p + "ssm_norm.weight");     // [128]
  const wOut = T(p + "ssm_out.weight");         // [5120, 6144]

  const xn = rmsnorm(x, attnNorm, dModel);
  console.log("attn_norm[0..8]:", p8(xn));

  const qkvMixed = matmul(wqkv, xn, keyDim * 2 + dInner, dModel);
  console.log("linear_attn_qkv_mixed[0..8]:", p8(qkvMixed));
  const z = matmul(wz, xn, dInner, dModel);
  console.log("z[0..8]:", p8(z));

  const beta = matmul(wBeta, xn, nVHeads, dModel).map(sigmoid);
  console.log("beta_sigmoid[0..8]:", p8(Float32Array.from(beta)));
  const alpha = matmul(wAlpha, xn, nVHeads, dModel);
  const gate = new Float32Array(nVHeads);
  for (let h = 0; h < nVHeads; h++) gate[h] = softplus(alpha[h] + dtBias[h]) * ssmA[h];
  console.log("gate[0..8]:", p8(gate));

  // conv over [zeros(3), current] per channel, then silu
  const convDim = keyDim * 2 + dInner;
  const convOut = new Float32Array(convDim);
  for (let c = 0; c < convDim; c++)
    convOut[c] = silu(conv[c * dConv + (dConv - 1)] * qkvMixed[c]);
  console.log("conv_output_silu[0..8]:", p8(convOut));

  // split + per-head l2 norm on q,k
  const q = convOut.slice(0, keyDim);
  const k = convOut.slice(keyDim, keyDim * 2);
  const v = convOut.slice(keyDim * 2);
  for (let h = 0; h < nKHeads; h++) { l2normHead(q, h * dState, dState); l2normHead(k, h * dState, dState); }
  console.log("q_conv_predelta[0..8]:", p8(q));
  console.log("k_conv_predelta[0..8]:", p8(k));

  // delta rule, S = 0 at pos 0; kv-head mapping = h % nKHeads (ggml_repeat tiling)
  const scale = 1 / Math.sqrt(dState);
  const out = new Float32Array(dInner);
  for (let h = 0; h < nVHeads; h++) {
    const kh = h % nKHeads;
    const qOff = kh * dState, vOff = h * dState;
    const decay = Math.exp(gate[h]); // multiplies S (zero here, but kept for structure)
    // S = S*decay (=0); v_hat = S.k = 0; d = (v - v_hat) * beta
    // S[i][j] += k[i]*d[j]; o[j] = sum_i S[i][j] * q[i]*scale
    // with S starting at 0: o[j] = d[j] * sum_i k[i]*q[i]*scale
    let kq = 0;
    for (let i = 0; i < dState; i++) kq += k[qOff + i] * q[qOff + i] * scale;
    for (let j = 0; j < dState; j++) out[vOff + j] = (v[vOff + j] * beta[h]) * kq;
  }
  console.log("core_attn_out[0..8]:", p8(out));

  // gated norm: rmsnorm per head (weight [128]) * silu(z)
  const gated = new Float32Array(dInner);
  for (let h = 0; h < nVHeads; h++) {
    const norm = rmsnorm(out, ssmNorm, dState, h * dState, 0);
    for (let j = 0; j < dState; j++) gated[h * dState + j] = norm[j] * silu(z[h * dState + j]);
  }
  console.log("final_output[0..8]:", p8(gated));

  const proj = matmul(wOut, gated, dModel, dInner);
  console.log("linear_attn_out[0..8]:", p8(proj));
  return proj;
}

// ---- full attention layer (gated, single token pos 0) ----
function fullAttnLayer(li, x) {
  const p = `blk.${li}.`;
  const attnNorm = T(p + "attn_norm.weight");
  const wq = T(p + "attn_q.weight");     // [12288, 5120] per head: q(256) then gate(256)
  const wk = T(p + "attn_k.weight");     // [1024, 5120]
  const wv = T(p + "attn_v.weight");     // [1024, 5120]
  const wo = T(p + "attn_output.weight");// [5120, 6144]
  const qNormW = T(p + "attn_q_norm.weight"); // [256]
  const kNormW = T(p + "attn_k_norm.weight"); // [256]
  const nH = M["qwen35.attention.head_count"];      // 24
  const nKV = M["qwen35.attention.head_count_kv"];  // 4
  const hd = M["qwen35.attention.key_length"];      // 256
  const nRot = M["qwen35.rope.dimension_count"];    // 64
  const theta = M["qwen35.rope.freq_base"];         // 1e7

  const xn = rmsnorm(x, attnNorm, dModel);
  console.log("attn_norm[0..8]:", p8(xn));
  const qFull = matmul(wq, xn, nH * hd * 2, dModel);
  const kRaw = matmul(wk, xn, nKV * hd, dModel);
  const v = matmul(wv, xn, nKV * hd, dModel);

  // split q/gate per head, per-head rmsnorm on q and k
  const q = new Float32Array(nH * hd), gate = new Float32Array(nH * hd);
  for (let h = 0; h < nH; h++) {
    q.set(qFull.subarray(h * 2 * hd, h * 2 * hd + hd), h * hd);
    gate.set(qFull.subarray(h * 2 * hd + hd, (h + 1) * 2 * hd), h * hd);
  }
  const qn = new Float32Array(nH * hd);
  for (let h = 0; h < nH; h++) qn.set(rmsnorm(q, qNormW, hd, h * hd, 0), h * hd);
  const kn = new Float32Array(nKV * hd);
  for (let h = 0; h < nKV; h++) kn.set(rmsnorm(kRaw, kNormW, hd, h * hd, 0), h * hd);
  console.log("Qcur_normed[0..8]:", p8(qn));
  console.log("Kcur_normed[0..8]:", p8(kn));
  // pos 0: rope is identity (cos 0 = 1, sin 0 = 0); skip rotation

  // attention over a single cached token: softmax over 1 -> weights = 1 -> out = v per group
  const group = nH / nKV;
  const attn = new Float32Array(nH * hd);
  for (let h = 0; h < nH; h++) {
    const kvH = Math.floor(h / group);
    attn.set(v.subarray(kvH * hd, (kvH + 1) * hd), h * hd);
  }
  console.log("attn_pregate[0..8]:", p8(attn));
  for (let i = 0; i < attn.length; i++) attn[i] *= sigmoid(gate[i]);
  console.log("attn_gated[0..8]:", p8(attn));
  const proj = matmul(wo, attn, dModel, nH * hd);
  console.log("attn_output[0..8]:", p8(proj));
  return proj;
}

// ---- ffn ----
function ffn(li, x) {
  const p = `blk.${li}.`;
  const norm = T(p + "post_attention_norm.weight");
  const wg = T(p + "ffn_gate.weight"), wu = T(p + "ffn_up.weight"), wd = T(p + "ffn_down.weight");
  const inter = M["qwen35.feed_forward_length"];
  const xn = rmsnorm(x, norm, dModel);
  const g = matmul(wg, xn, inter, dModel);
  const u = matmul(wu, xn, inter, dModel);
  for (let i = 0; i < inter; i++) g[i] = silu(g[i]) * u[i];
  const d = matmul(wd, g, dModel, inter);
  console.log("ffn_out[0..8]:", p8(d));
  return d;
}

// ---- main ----
const tokenId = parseInt(process.argv[2] || "760", 10); // "The" in qwen3.8 vocab
const maxLayer = parseInt(process.argv[3] || "0", 10);
console.log(`token ${tokenId}, layers 0..${maxLayer}`);

const embedInfo = G.tensors["token_embd.weight"];
const rowBlocks = dModel / 32;
const rowBytes = rowBlocks * 18;
const rowRaw = Buffer.alloc(rowBytes);
fs.readSync(fd, rowRaw, 0, rowBytes, embedInfo.byteOffset + tokenId * rowBytes);
let x = dequantF32({ ggmlType: 2, nElems: dModel, shape: [dModel] }, new Uint8Array(rowRaw.buffer, rowRaw.byteOffset, rowBytes));
console.log("inp_embd[0..8]:", p8(x));

for (let li = 0; li <= maxLayer; li++) {
  const isFull = li % 4 === 3;
  console.log(`\n=== layer ${li} (${isFull ? "full-attention" : "delta-net"}) ===`);
  const attnOut = isFull ? fullAttnLayer(li, x) : linearAttnLayer(li, x);
  const x2 = Float32Array.from(x);
  for (let i = 0; i < dModel; i++) x2[i] += attnOut[i];
  console.log(`attn_residual-${li}[0..8]:`, p8(x2));
  const f = ffn(li, x2);
  for (let i = 0; i < dModel; i++) x2[i] += f[i];
  console.log(`layer_out-${li}[0..8]:`, p8(x2));
  x = x2;
}
