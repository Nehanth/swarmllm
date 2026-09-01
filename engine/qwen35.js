// bello Qwen3.5/3.8 engine — hybrid Gated-DeltaNet + gated-attention WebGPU
// inference, layer-shardable like BelloEngine. Golden reference: ref_q38.mjs
// (validated line-by-line against llama.cpp eval-callback dumps).
import { WGSL } from "./engine.js";

const WGSL2 = /* wgsl */ `
struct DN {
  convDim: u32, dState: u32, nKH: u32, nVH: u32,
  keyDim: u32, nRot: u32, hd: u32, dInner: u32,
  ropeTheta: f32, eps2: f32, pad0: u32, pad1: u32,
};

// --- causal conv (K=4) + silu; updates rolling state ---
@group(1) @binding(0) var<storage, read> cv_x: array<f32>;      // qkv_mixed [convDim]
@group(1) @binding(1) var<storage, read> cv_w: array<f32>;      // [convDim, 4]
@group(1) @binding(2) var<storage, read_write> cv_st: array<f32>; // [convDim, 3]
@group(1) @binding(3) var<storage, read_write> cv_y: array<f32>;  // [convDim]
@group(1) @binding(4) var<uniform> cv_dn: DN;
@compute @workgroup_size(64)
fn dn_conv(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= cv_dn.convDim) { return; }
  let x = cv_x[c];
  var acc = cv_w[c * 4u + 3u] * x;
  acc += cv_w[c * 4u + 0u] * cv_st[c * 3u + 0u];
  acc += cv_w[c * 4u + 1u] * cv_st[c * 3u + 1u];
  acc += cv_w[c * 4u + 2u] * cv_st[c * 3u + 2u];
  cv_y[c] = acc / (1.0 + exp(-acc));
  cv_st[c * 3u + 0u] = cv_st[c * 3u + 1u];
  cv_st[c * 3u + 1u] = cv_st[c * 3u + 2u];
  cv_st[c * 3u + 2u] = x;
}

// --- per-head gates: beta = sigmoid(betaRaw); decay = exp(softplus(alpha+dt)*A) ---
@group(1) @binding(0) var<storage, read> gt_alpha: array<f32>;
@group(1) @binding(1) var<storage, read> gt_beta: array<f32>;
@group(1) @binding(2) var<storage, read> gt_dt: array<f32>;
@group(1) @binding(3) var<storage, read> gt_a: array<f32>;
@group(1) @binding(4) var<storage, read_write> gt_bout: array<f32>;
@group(1) @binding(5) var<storage, read_write> gt_dout: array<f32>;
@group(1) @binding(6) var<uniform> gt_dn: DN;
@compute @workgroup_size(64)
fn dn_gates(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  if (h >= gt_dn.nVH) { return; }
  gt_bout[h] = 1.0 / (1.0 + exp(-gt_beta[h]));
  let av = gt_alpha[h] + gt_dt[h];
  var sp: f32;
  if (av > 20.0) { sp = av; } else { sp = log(1.0 + exp(av)); }
  gt_dout[h] = exp(sp * gt_a[h]);
}

// --- per-head L2 norm (q/k slices bound with offset) ---
@group(1) @binding(0) var<storage, read_write> l2_v: array<f32>;
@group(1) @binding(1) var<uniform> l2_heads: u32;
@group(1) @binding(2) var<uniform> l2_dn: DN;
@compute @workgroup_size(32)
fn dn_l2(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  if (h >= l2_heads) { return; }
  let off = h * l2_dn.dState;
  var ss: f32 = 0.0;
  for (var i: u32 = 0u; i < l2_dn.dState; i++) { let v = l2_v[off + i]; ss += v * v; }
  let inv = 1.0 / max(sqrt(ss), l2_dn.eps2);
  for (var i: u32 = 0u; i < l2_dn.dState; i++) { l2_v[off + i] *= inv; }
}

// --- gated delta rule: one workgroup per v-head, thread j = state column ---
@group(1) @binding(0) var<storage, read> dl_q: array<f32>;      // [keyDim] (l2-normed)
@group(1) @binding(1) var<storage, read> dl_k: array<f32>;      // [keyDim]
@group(1) @binding(2) var<storage, read> dl_v: array<f32>;      // [dInner]
@group(1) @binding(3) var<storage, read> dl_beta: array<f32>;   // [nVH]
@group(1) @binding(4) var<storage, read> dl_decay: array<f32>;  // [nVH]
@group(1) @binding(5) var<storage, read_write> dl_s: array<f32>; // [nVH, dState, dState]
@group(1) @binding(6) var<storage, read_write> dl_o: array<f32>; // [dInner]
@group(1) @binding(7) var<uniform> dl_dn: DN;
@compute @workgroup_size(128)
fn dn_delta(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x;
  let j = lid.x;
  if (h >= dl_dn.nVH || j >= dl_dn.dState) { return; }
  let kh = h % dl_dn.nKH;
  let kOff = kh * dl_dn.dState;
  let vOff = h * dl_dn.dState;
  let Sb = h * dl_dn.dState * dl_dn.dState;
  let decay = dl_decay[h];
  let scale = inverseSqrt(f32(dl_dn.dState));
  var vhat: f32 = 0.0;
  var sq: f32 = 0.0;
  var kq: f32 = 0.0;
  for (var i: u32 = 0u; i < dl_dn.dState; i++) {
    let idx = Sb + i * dl_dn.dState + j;
    let sdec = dl_s[idx] * decay;
    dl_s[idx] = sdec;
    let ki = dl_k[kOff + i];
    let qi = dl_q[kOff + i];
    vhat += sdec * ki;
    sq += sdec * qi;
    kq += ki * qi;
  }
  let d = (dl_v[vOff + j] - vhat) * dl_beta[h];
  for (var i: u32 = 0u; i < dl_dn.dState; i++) {
    let idx = Sb + i * dl_dn.dState + j;
    dl_s[idx] += dl_k[kOff + i] * d;
  }
  dl_o[vOff + j] = (sq + d * kq) * scale;
}

// --- gated norm: rmsnorm per head (w[dState]) * silu(z) ---
@group(1) @binding(0) var<storage, read> gn_x: array<f32>;   // [dInner]
@group(1) @binding(1) var<storage, read> gn_z: array<f32>;   // [dInner]
@group(1) @binding(2) var<storage, read> gn_w: array<f32>;   // [dState]
@group(1) @binding(3) var<storage, read_write> gn_y: array<f32>;
@group(1) @binding(4) var<uniform> gn_dn: DN;
var<workgroup> gn_partial: array<f32, 128>;
@compute @workgroup_size(128)
fn dn_gatenorm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x;
  let j = lid.x;
  let off = h * gn_dn.dState;
  let v = gn_x[off + j];
  gn_partial[j] = v * v;
  workgroupBarrier();
  var stride: u32 = 64u;
  while (stride > 0u) {
    if (j < stride) { gn_partial[j] += gn_partial[j + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let inv = inverseSqrt(gn_partial[0] / f32(gn_dn.dState) + cfg.eps);
  let z = gn_z[off + j];
  gn_y[off + j] = gn_x[off + j] * inv * gn_w[j] * (z / (1.0 + exp(-z)));
}

// --- split interleaved [q|gate] per head from q_full ---
@group(1) @binding(0) var<storage, read> qs_full: array<f32>;  // [nH*2*hd]
@group(1) @binding(1) var<storage, read_write> qs_q: array<f32>;
@group(1) @binding(2) var<storage, read_write> qs_g: array<f32>;
@group(1) @binding(3) var<uniform> qs_dn: DN;
@compute @workgroup_size(64)
fn qsplit(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = cfg.nH * qs_dn.hd;
  if (idx >= total) { return; }
  let h = idx / qs_dn.hd;
  let i = idx % qs_dn.hd;
  qs_q[idx] = qs_full[h * 2u * qs_dn.hd + i];
  qs_g[idx] = qs_full[h * 2u * qs_dn.hd + qs_dn.hd + i];
}

// --- partial neox rope: rotate first nRot dims of each head ---
@group(1) @binding(0) var<storage, read_write> rp2_v: array<f32>;
@group(1) @binding(1) var<uniform> rp2_heads: u32;
@group(1) @binding(2) var<uniform> rp2_dn: DN;
@compute @workgroup_size(64)
fn rope_part(@builtin(global_invocation_id) gid: vec3<u32>) {
  let half = rp2_dn.nRot / 2u;
  let total = rp2_heads * half;
  let idx = gid.x;
  if (idx >= total) { return; }
  let h = idx / half;
  let i = idx % half;
  let off = h * rp2_dn.hd;
  let ang = f32(frame.pos) * pow(rp2_dn.ropeTheta, -f32(2u * i) / f32(rp2_dn.nRot));
  let c = cos(ang); let s = sin(ang);
  let a = rp2_v[off + i]; let b = rp2_v[off + i + half];
  rp2_v[off + i] = a * c - b * s;
  rp2_v[off + i + half] = b * c + a * s;
}

// --- a *= sigmoid(g) ---
@group(1) @binding(0) var<storage, read_write> sm_a: array<f32>;
@group(1) @binding(1) var<storage, read> sm_g: array<f32>;
@group(1) @binding(2) var<uniform> sm_n: u32;
@compute @workgroup_size(64)
fn sigmoid_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= sm_n) { return; }
  let g = sm_g[i];
  sm_a[i] = sm_a[i] * (1.0 / (1.0 + exp(-g)));
}
`;

export class Qwen35Engine {
  static async create(opts) {
    const e = new Qwen35Engine();
    await e._init(opts);
    return e;
  }

  // Fresh context: forget the conversation so far. Recurrent (DeltaNet) states and
  // conv windows are zeroed; the KV caches are simply overwritten from position 0.
  reset() {
    const enc = this.device.createCommandEncoder();
    for (const R of this.layers) {
      if (R.convState) enc.clearBuffer(R.convState);
      if (R.S) enc.clearBuffer(R.S);
    }
    this.device.queue.submit([enc.finish()]);
    this.pos = 0;
  }

  // opts: { device, meta (gguf meta), weights, layerRange, hasEmbed, hasHead, maxSeq }
  async _init({ device, meta, weights, layerRange, hasEmbed = true, hasHead = true, maxSeq = 512, vocab: vocabOpt, matvecVariant = "coop" }) {
    this.device = device;
    this.mvVariant = matvecVariant;
    const M = meta;
    const dim = M["qwen35.embedding_length"];
    const nH = M["qwen35.attention.head_count"];
    const nKV = M["qwen35.attention.head_count_kv"];
    const hd = M["qwen35.attention.key_length"];
    const nRot = M["qwen35.rope.dimension_count"];
    const ropeTheta = M["qwen35.rope.freq_base"];
    const eps = M["qwen35.attention.layer_norm_rms_epsilon"];
    const inter = M["qwen35.feed_forward_length"];
    const dState = M["qwen35.ssm.state_size"];
    const nKH = M["qwen35.ssm.group_count"];
    const nVH = M["qwen35.ssm.time_step_rank"];
    const dInner = M["qwen35.ssm.inner_size"];
    const keyDim = dState * nKH;
    const convDim = keyDim * 2 + dInner;
    // workers parse the header without the vocab; the embedding's row count says the same thing
    const vocab = M["tokenizer.ggml.tokens"]?.length ?? vocabOpt ?? M["qwen35.vocab_size"] ?? 248320;
    const qDim = nH * hd, kvDim = nKV * hd;
    this.dims = { dim, nH, nKV, hd, nRot, inter, dState, nKH, nVH, dInner, keyDim, convDim, vocab, qDim, kvDim };
    this.maxSeq = maxSeq;
    const [lo, hi] = layerRange;
    this.lo = lo; this.hi = hi;
    this.hasEmbed = hasEmbed; this.hasHead = hasHead;
    this.pos = 0;

    // ---- pipelines with explicit layouts ----
    const mod = device.createShaderModule({ code: WGSL + WGSL2 });
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
      rmsnorm: ["ro", "ro", "rw", "u"], head_norm: ["rw", "ro", "u"],
      attn_scores: ["ro", "ro", "rw"], attn_softmax: ["rw"],
      attn_out: ["ro", "ro", "rw"], silu_mul: ["rw", "ro"], add_res: ["rw", "ro"],
      dn_conv: ["ro", "ro", "rw", "rw", "u"],
      dn_gates: ["ro", "ro", "ro", "ro", "rw", "rw", "u"],
      dn_l2: ["rw", "u", "u"],
      dn_delta: ["ro", "ro", "ro", "ro", "ro", "rw", "rw", "u"],
      dn_gatenorm: ["ro", "ro", "ro", "rw", "u"],
      qsplit: ["ro", "rw", "rw", "u"],
      rope_part: ["rw", "u", "u"],
      sigmoid_mul: ["rw", "ro", "u"],
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

    // ---- uniforms ----
    const cfgData = new ArrayBuffer(48);
    const cu = new Uint32Array(cfgData), cf = new Float32Array(cfgData);
    cu.set([dim, kvDim, nH, nKV, hd, inter, vocab, maxSeq], 0);
    cf[8] = eps; cf[9] = ropeTheta; cu[10] = qDim;
    this.cfgBuf = this._buf(cfgData, GPUBufferUsage.UNIFORM);
    this.frameBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const dnData = new ArrayBuffer(48);
    const du = new Uint32Array(dnData), df = new Float32Array(dnData);
    du.set([convDim, dState, nKH, nVH, keyDim, nRot, hd, dInner], 0);
    df[8] = ropeTheta; df[9] = eps;
    this.dnBuf = this._buf(dnData, GPUBufferUsage.UNIFORM);
    this._shapes = {};
    this.uNH = this._buf(new Uint32Array([nH]), GPUBufferUsage.UNIFORM);
    this.uNKV = this._buf(new Uint32Array([nKV]), GPUBufferUsage.UNIFORM);
    this.uNKH = this._buf(new Uint32Array([nKH]), GPUBufferUsage.UNIFORM);
    this.uDim = this._buf(new Uint32Array([dim]), GPUBufferUsage.UNIFORM);
    this.uQDim = this._buf(new Uint32Array([qDim]), GPUBufferUsage.UNIFORM);
    this.uHd = this._buf(new Uint32Array([hd]), GPUBufferUsage.UNIFORM);

    // ---- working buffers ----
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.x = device.createBuffer({ size: dim * 4, usage: S });
    this.xn = device.createBuffer({ size: dim * 4, usage: S });
    this.tmpDim = device.createBuffer({ size: dim * 4, usage: S });
    this.g = device.createBuffer({ size: inter * 4, usage: S });
    this.u = device.createBuffer({ size: inter * 4, usage: S });
    // delta-net
    this.qkv = device.createBuffer({ size: convDim * 4, usage: S });
    this.convOut = device.createBuffer({ size: convDim * 4, usage: S });
    this.z = device.createBuffer({ size: dInner * 4, usage: S });
    this.alpha = device.createBuffer({ size: 64 * 4, usage: S });
    this.betaRaw = device.createBuffer({ size: 64 * 4, usage: S });
    this.beta = device.createBuffer({ size: 64 * 4, usage: S });
    this.decay = device.createBuffer({ size: 64 * 4, usage: S });
    this.dOut = device.createBuffer({ size: dInner * 4, usage: S });
    this.gated = device.createBuffer({ size: dInner * 4, usage: S });
    // full attention
    this.qFull = device.createBuffer({ size: nH * hd * 2 * 4, usage: S });
    this.q = device.createBuffer({ size: qDim * 4, usage: S });
    this.gAttn = device.createBuffer({ size: qDim * 4, usage: S });
    this.k = device.createBuffer({ size: kvDim * 4, usage: S });
    this.v = device.createBuffer({ size: kvDim * 4, usage: S });
    this.attnOut = device.createBuffer({ size: qDim * 4, usage: S });
    this.scores = device.createBuffer({ size: nH * maxSeq * 4, usage: S });
    this.stageX = device.createBuffer({ size: dim * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    // ---- weights + per-layer resources ----
    const up = (e2) => {
      if (!e2) return null;
      if (e2.gpu) return e2.gpu;          // already streamed onto the GPU during download
      let r;
      if (e2.kind === "q8" || e2.kind === "q4")
        r = { kind: e2.kind, qs: this._buf(e2.qs, GPUBufferUsage.STORAGE), sc: this._buf(e2.scales, GPUBufferUsage.STORAGE) };
      else r = { kind: "f32", buf: this._buf(e2.data, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) };
      e2.qs = e2.scales = e2.data = null; // release CPU copy once it lives on the GPU
      return r;
    };
    const coop = this.mvVariant === "coop";
    const mv = (w, x, y, dOut, dIn) => {
      const base = w.kind === "q8" ? "matvec_q8" : w.kind === "q4" ? "matvec_q4" : "matvec";
      const pipe = coop ? base + "_coop" : base;
      const bufs = w.kind === "f32" ? [w.buf, x, y, this._shape(dOut, dIn)] : [w.qs, w.sc, x, y, this._shape(dOut, dIn)];
      return { pipe, wgs: coop ? Math.ceil(dOut / 4) : Math.ceil(dOut / 64), bg: this._bg(this.pipes[pipe], 1, bufs) };
    };
    this._mv = mv;
    const bgNorm = (x, w, y) => this._bg(this.pipes.rmsnorm, 1, [x, w.buf, y, this.uDim]);

    this.layers = [];
    for (const L of weights.layers) {
      const R = { isFull: L.isFull };
      R.attnNorm = up(L.attnNorm); R.postNorm = up(L.postNorm);
      R.ffnGate = up(L.ffnGate); R.ffnUp = up(L.ffnUp); R.ffnDown = up(L.ffnDown);
      R.bgNorm1 = bgNorm(this.x, R.attnNorm, this.xn);
      R.bgNorm2 = bgNorm(this.x, R.postNorm, this.xn);
      R.mvGate = mv(R.ffnGate, this.xn, this.g, inter, dim);
      R.mvUp = mv(R.ffnUp, this.xn, this.u, inter, dim);
      R.mvDown = mv(R.ffnDown, this.g, this.tmpDim, dim, inter);
      if (L.isFull) {
        R.wq = up(L.wq); R.wk = up(L.wk); R.wv = up(L.wv); R.wo = up(L.wo);
        R.qNorm = up(L.qNorm); R.kNorm = up(L.kNorm);
        R.kCache = device.createBuffer({ size: maxSeq * kvDim * 4, usage: S });
        R.vCache = device.createBuffer({ size: maxSeq * kvDim * 4, usage: S });
        R.mvQ = mv(R.wq, this.xn, this.qFull, nH * hd * 2, dim);
        R.mvK = mv(R.wk, this.xn, this.k, kvDim, dim);
        R.mvV = mv(R.wv, this.xn, this.v, kvDim, dim);
        R.mvO = mv(R.wo, this.attnOut, this.tmpDim, dim, qDim);
        R.bgQsplit = this._bg(this.pipes.qsplit, 1, [this.qFull, this.q, this.gAttn, this.dnBuf]);
        R.bgQNorm = this._bg(this.pipes.head_norm, 1, [this.q, R.qNorm.buf, this.uNH]);
        R.bgKNorm = this._bg(this.pipes.head_norm, 1, [this.k, R.kNorm.buf, this.uNKV]);
        R.bgRopeQ = this._bg(this.pipes.rope_part, 1, [this.q, this.uNH, this.dnBuf]);
        R.bgRopeK = this._bg(this.pipes.rope_part, 1, [this.k, this.uNKV, this.dnBuf]);
        R.bgScores = this._bg(this.pipes.attn_scores, 1, [this.q, R.kCache, this.scores]);
        R.bgSoftmax = this._bg(this.pipes.attn_softmax, 1, [this.scores]);
        R.bgAttnOut = this._bg(this.pipes.attn_out, 1, [this.scores, R.vCache, this.attnOut]);
        R.bgSigMul = this._bg(this.pipes.sigmoid_mul, 1, [this.attnOut, this.gAttn, this.uQDim]);
      } else {
        R.wqkv = up(L.wqkv); R.wz = up(L.wz);
        R.wBeta = up(L.wBeta); R.wAlpha = up(L.wAlpha);
        R.wOut = up(L.wOut);
        R.dtBias = this._buf(L.dtBias.data, GPUBufferUsage.STORAGE);
        R.ssmA = this._buf(L.ssmA.data, GPUBufferUsage.STORAGE);
        R.convW = this._buf(L.conv.data, GPUBufferUsage.STORAGE);
        R.ssmNorm = this._buf(L.ssmNorm.data, GPUBufferUsage.STORAGE);
        R.convState = device.createBuffer({ size: convDim * 3 * 4, usage: S });
        R.S = device.createBuffer({ size: nVH * dState * dState * 4, usage: S });
        R.mvQKV = mv(R.wqkv, this.xn, this.qkv, convDim, dim);
        R.mvZ = mv(R.wz, this.xn, this.z, dInner, dim);
        R.mvBeta = mv(R.wBeta, this.xn, this.betaRaw, nVH, dim);
        R.mvAlpha = mv(R.wAlpha, this.xn, this.alpha, nVH, dim);
        R.mvOut = mv(R.wOut, this.gated, this.tmpDim, dim, dInner);
        R.bgConv = this._bg(this.pipes.dn_conv, 1, [this.qkv, R.convW, R.convState, this.convOut, this.dnBuf]);
        R.bgGates = this._bg(this.pipes.dn_gates, 1, [this.alpha, this.betaRaw, R.dtBias, R.ssmA, this.beta, this.decay, this.dnBuf]);
        R.bgL2Q = this._bg2(this.pipes.dn_l2, [
          { buffer: this.convOut, offset: 0, size: keyDim * 4 },
          { buffer: this.uNKH }, { buffer: this.dnBuf }]);
        R.bgL2K = this._bg2(this.pipes.dn_l2, [
          { buffer: this.convOut, offset: keyDim * 4, size: keyDim * 4 },
          { buffer: this.uNKH }, { buffer: this.dnBuf }]);
        R.bgDelta = this._bg2(this.pipes.dn_delta, [
          { buffer: this.convOut, offset: 0, size: keyDim * 4 },
          { buffer: this.convOut, offset: keyDim * 4, size: keyDim * 4 },
          { buffer: this.convOut, offset: keyDim * 2 * 4, size: dInner * 4 },
          { buffer: this.beta }, { buffer: this.decay },
          { buffer: R.S }, { buffer: this.dOut }, { buffer: this.dnBuf }]);
        R.bgGateNorm = this._bg(this.pipes.dn_gatenorm, 1, [this.dOut, this.z, R.ssmNorm, this.gated, this.dnBuf]);
      }
      this.layers.push(R);
    }

    if (hasEmbed || hasHead) this.cpuEmbed = weights.embed;
    if (hasHead) {
      this.finalNorm = up(weights.finalNorm);
      if (weights.head) this.headEntry = up(weights.head);
      else { // tied: upload embed for the head but keep CPU copy for row lookups
        const e = weights.embed;
        this.headEntry = { kind: e.kind, qs: this._buf(e.qs, GPUBufferUsage.STORAGE), sc: this._buf(e.scales, GPUBufferUsage.STORAGE) };
      }
      this.logits = device.createBuffer({ size: vocab * 4, usage: S });
      this.stageLogits = device.createBuffer({ size: vocab * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      this.bgFinalNorm = bgNorm(this.x, this.finalNorm, this.xn);
      this.headOp = mv(this.headEntry, this.xn, this.logits, vocab, dim);
    }

    this.bgCommonFor = {};
    for (const [k2, p] of Object.entries(this.pipes))
      this.bgCommonFor[k2] = this._bg(p, 0, [this.cfgBuf, this.frameBuf]);
    this.bgSilu = this._bg(this.pipes.silu_mul, 1, [this.g, this.u]);
    this.bgAddTmp = this._bg(this.pipes.add_res, 1, [this.x, this.tmpDim]);
  }

  _shape(dOut, dIn) {
    const key = dOut + "," + dIn;
    if (!this._shapes[key])
      this._shapes[key] = this._buf(new Uint32Array([dOut, dIn, 0, 0]), GPUBufferUsage.UNIFORM);
    return this._shapes[key];
  }
  _buf(data, usage) {
    const src = ArrayBuffer.isView(data) ? data : new Uint8Array(data);
    const size = Math.ceil(src.byteLength / 4) * 4;
    const buf = this.device.createBuffer({ size, usage, mappedAtCreation: true });
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(src.buffer, src.byteOffset, src.byteLength));
    buf.unmap();
    return buf;
  }
  _bg(pipe, group, buffers) {
    return this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(group),
      entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
  }
  _bg2(pipe, resources) {
    return this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(1),
      entries: resources.map((r, i) => ({ binding: i, resource: r })),
    });
  }
  _d(pass, name, bg, threads, wg = 64) {
    pass.setPipeline(this.pipes[name]);
    pass.setBindGroup(0, this.bgCommonFor[name]);
    pass.setBindGroup(1, bg);
    pass.dispatchWorkgroups(Math.ceil(threads / wg));
  }
  _dop(pass, op) {
    pass.setPipeline(this.pipes[op.pipe]);
    pass.setBindGroup(0, this.bgCommonFor[op.pipe]);
    pass.setBindGroup(1, op.bg);
    pass.dispatchWorkgroups(op.wgs);
  }
  _setFrame(pos, seqLen) {
    this.device.queue.writeBuffer(this.frameBuf, 0, new Uint32Array([pos, seqLen]));
  }

  _encodeLayer(enc, i) {
    const D = this.dims, L = this.layers[i];
    const seqLen = this.pos + 1;
    if (L.isFull) {
      {
        const p = enc.beginComputePass();
        this._d(p, "rmsnorm", L.bgNorm1, 256, 256);
        this._dop(p, L.mvQ);
        this._dop(p, L.mvK);
        this._dop(p, L.mvV);
        this._d(p, "qsplit", L.bgQsplit, D.nH * D.hd);
        this._d(p, "head_norm", L.bgQNorm, D.nH, 32);
        this._d(p, "head_norm", L.bgKNorm, D.nKV, 32);
        this._d(p, "rope_part", L.bgRopeQ, D.nH * D.nRot / 2);
        this._d(p, "rope_part", L.bgRopeK, D.nKV * D.nRot / 2);
        p.end();
      }
      enc.copyBufferToBuffer(this.k, 0, L.kCache, this.pos * D.kvDim * 4, D.kvDim * 4);
      enc.copyBufferToBuffer(this.v, 0, L.vCache, this.pos * D.kvDim * 4, D.kvDim * 4);
      {
        const p = enc.beginComputePass();
        this._d(p, "attn_scores", L.bgScores, D.nH * seqLen);
        this._d(p, "attn_softmax", L.bgSoftmax, D.nH, 1);
        this._d(p, "attn_out", L.bgAttnOut, D.qDim);
        this._d(p, "sigmoid_mul", L.bgSigMul, D.qDim);
        this._dop(p, L.mvO);
        this._d(p, "add_res", this.bgAddTmp, D.dim);
        p.end();
      }
    } else {
      const p = enc.beginComputePass();
      this._d(p, "rmsnorm", L.bgNorm1, 256, 256);
      this._dop(p, L.mvQKV);
      this._dop(p, L.mvZ);
      this._dop(p, L.mvBeta);
      this._dop(p, L.mvAlpha);
      this._d(p, "dn_gates", L.bgGates, D.nVH);
      this._d(p, "dn_conv", L.bgConv, D.convDim);
      this._d(p, "dn_l2", L.bgL2Q, D.nKH, 32);
      this._d(p, "dn_l2", L.bgL2K, D.nKH, 32);
      this._d(p, "dn_delta", L.bgDelta, D.nVH * 128, 128);
      this._d(p, "dn_gatenorm", L.bgGateNorm, D.nVH * 128, 128);
      this._dop(p, L.mvOut);
      this._d(p, "add_res", this.bgAddTmp, D.dim);
      p.end();
    }
    // ffn
    {
      const p = enc.beginComputePass();
      this._d(p, "rmsnorm", L.bgNorm2, 256, 256);
      this._dop(p, L.mvGate);
      this._dop(p, L.mvUp);
      this._d(p, "silu_mul", this.bgSilu, D.inter);
      this._dop(p, L.mvDown);
      this._d(p, "add_res", this.bgAddTmp, D.dim);
      p.end();
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

  async _readback(srcBuf, stageBuf, n) {
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(srcBuf, 0, stageBuf, 0, n * 4);
    this.device.queue.submit([enc.finish()]);
    await stageBuf.mapAsync(GPUMapMode.READ);
    const out = Float32Array.from(new Float32Array(stageBuf.getMappedRange(), 0, n));
    stageBuf.unmap();
    return out;
  }

  // prefill fast path: layers only, no head, no readback
  async prefillToken(tokenId) {
    this._setFrame(this.pos, this.pos + 1);
    this.device.queue.writeBuffer(this.x, 0, this._embedRowF32(tokenId));
    const enc = this.device.createCommandEncoder();
    for (let i = 0; i < this.layers.length; i++) this._encodeLayer(enc, i);
    this.device.queue.submit([enc.finish()]);
    this.pos++;
    // fire-and-forget; callers batch backpressure via onSubmittedWorkDone()
  }

  async embedRun(tokenId, pos) {
    const { dim } = this.dims;
    this.pos = pos;
    this._setFrame(pos, pos + 1);
    this.device.queue.writeBuffer(this.x, 0, this._embedRowF32(tokenId));
    const enc = this.device.createCommandEncoder();
    for (let i = 0; i < this.layers.length; i++) this._encodeLayer(enc, i);
    this.device.queue.submit([enc.finish()]);
    return await this._readback(this.x, this.stageX, dim);
  }

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

  async headFromHidden(xIn) {
    const { vocab } = this.dims;
    this.device.queue.writeBuffer(this.x, 0, xIn);
    const enc = this.device.createCommandEncoder();
    {
      const p = enc.beginComputePass();
      this._d(p, "rmsnorm", this.bgFinalNorm, 256, 256);
      this._dop(p, this.headOp);
      p.end();
    }
    this.device.queue.submit([enc.finish()]);
    return await this._readback(this.logits, this.stageLogits, vocab);
  }

  // Whole token in one encoder + one submit; no hidden-state readback between
  // the last layer and the head (that round trip cost a full pipeline drain).
  async forwardToken(tokenId) {
    const { vocab } = this.dims;
    this._setFrame(this.pos, this.pos + 1);
    this.device.queue.writeBuffer(this.x, 0, this._embedRowF32(tokenId));
    const enc = this.device.createCommandEncoder();
    for (let i = 0; i < this.layers.length; i++) this._encodeLayer(enc, i);
    {
      const p = enc.beginComputePass();
      this._d(p, "rmsnorm", this.bgFinalNorm, 256, 256);
      this._dop(p, this.headOp);
      p.end();
    }
    this.device.queue.submit([enc.finish()]);
    const logits = await this._readback(this.logits, this.stageLogits, vocab);
    this.pos++;
    return logits;
  }
}
