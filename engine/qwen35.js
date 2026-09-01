// bello Qwen3.5/3.8 engine — hybrid Gated-DeltaNet + gated-attention WebGPU
// inference, layer-shardable like BelloEngine. Golden reference: ref_q38.mjs
// (validated line-by-line against llama.cpp eval-callback dumps).
import { WGSL, coopWGSL } from "./engine.js";
import { f16ToF32 } from "./gguf.js";

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
  async _init({ device, meta, weights, layerRange, hasEmbed = true, hasHead = true, maxSeq = 512, vocab: vocabOpt, matvecVariant = "coop", coopWG = 256, coopRows = 4 }) {
    this.device = device;
    this.mvVariant = matvecVariant;
    this.coopWG = coopWG; this.coopRows = coopRows;
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
    const mod = device.createShaderModule({ code: WGSL + coopWGSL(coopWG, coopRows) + WGSL2 });
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
      matvec_gu: ["ro", "ro", "ro", "rw", "u"], matvec_gu_b: ["ro", "ro", "ro", "rw", "u"],
      matvec_q8_gu: ["ro", "ro", "ro", "ro", "ro", "rw", "u"], matvec_q8_gu_b: ["ro", "ro", "ro", "ro", "ro", "rw", "u"],
      matvec_q4_gu: ["ro", "ro", "ro", "ro", "ro", "rw", "u"], matvec_q4_gu_b: ["ro", "ro", "ro", "ro", "ro", "rw", "u"],
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
    // compile every pipeline in parallel (async): overlaps shader compilation
    // with the rest of setup instead of serializing 20+ compiles
    await Promise.all(Object.entries(G1).map(async ([name, spec]) => {
      const layout1 = device.createBindGroupLayout({
        entries: spec.map((t, i) => ({ binding: i, visibility: C, buffer: { type: bufType[t] } })),
      });
      this.pipes[name] = await device.createComputePipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout0, layout1] }),
        compute: { module: mod, entryPoint: name },
      });
    }));

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
      return { pipe, wgs: coop ? Math.ceil(dOut / this.coopRows) : Math.ceil(dOut / 64), bg: this._bg(this.pipes[pipe], 1, bufs) };
    };
    this._mv = mv;
    const guOp = (wg2, wu2, x, y, dOut, dIn, xB, yB) => {
      if (!coop || !wg2 || !wu2 || wg2.kind !== wu2.kind) return null;
      const base = wg2.kind === "q8" ? "matvec_q8_gu" : wg2.kind === "q4" ? "matvec_q4_gu" : "matvec_gu";
      const pipe = xB ? base + "_b" : base;
      const shp = xB ? this._shapeB(dOut, dIn, xB.stride / 16, yB.stride / 4) : this._shapeB(dOut, dIn, 0, 0);
      const bufs = wg2.kind === "f32" ? [wg2.buf, wu2.buf, x, y, shp] : [wg2.qs, wg2.sc, wu2.qs, wu2.sc, x, y, shp];
      return { pipe, wgs: Math.ceil(dOut / this.coopRows), bg: this._bg(this.pipes[pipe], 1, bufs) };
    };
    this._guOp = guOp;
    const bgNorm = (x, w, y) => this._bg(this.pipes.rmsnorm, 1, [x, w.buf, y, this.uDim]);

    this.layers = [];
    const buildLayer = (L) => {
      const R = { isFull: L.isFull };
      R.attnNorm = up(L.attnNorm); R.postNorm = up(L.postNorm);
      R.ffnGate = up(L.ffnGate); R.ffnUp = up(L.ffnUp); R.ffnDown = up(L.ffnDown);
      R.bgNorm1 = bgNorm(this.x, R.attnNorm, this.xn);
      R.bgNorm2 = bgNorm(this.x, R.postNorm, this.xn);
      R.mvGate = mv(R.ffnGate, this.xn, this.g, inter, dim);
      R.mvUp = mv(R.ffnUp, this.xn, this.u, inter, dim);
      R.gu = guOp(R.ffnGate, R.ffnUp, this.xn, this.g, inter, dim);
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
      return R;
    };
    for (const L of weights.layers) this.layers.push(buildLayer(L));

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

    // ---- multi-token prediction (draft) head ----
    if (weights.mtp && hasHead) {
      const W = weights.mtp;
      this.mtpLayer = buildLayer(W.layer);
      this.mtp = {
        ehProj: up(W.ehProj), enorm: up(W.enorm), hnorm: up(W.hnorm), headNorm: up(W.sharedHeadNorm),
        emb: device.createBuffer({ size: dim * 4, usage: S }),
        ehIn: device.createBuffer({ size: 2 * dim * 4, usage: S }),   // [enorm(e) | hnorm(h)]
        stats: { drafts: 0, accepted: 0 },
      };
      const M2 = this.mtp;
      M2.bgENorm = this._bg2res(this.pipes.rmsnorm, [{ buffer: M2.emb }, { buffer: M2.enorm.buf },
        { buffer: M2.ehIn, offset: 0, size: dim * 4 }, { buffer: this.uDim }]);
      M2.bgHNormX = this._bg2res(this.pipes.rmsnorm, [{ buffer: this.x }, { buffer: M2.hnorm.buf },
        { buffer: M2.ehIn, offset: dim * 4, size: dim * 4 }, { buffer: this.uDim }]);
      M2.proj = mv(M2.ehProj, M2.ehIn, this.x, dim, 2 * dim);           // eh_proj -> MTP residual (in x)
      M2.bgHeadNorm = bgNorm(this.x, M2.headNorm, this.xn);               // shared_head_norm -> xn
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

  _encodeLayer(enc, i) { this._encodeLayerR(enc, this.layers[i], this.pos); }
  _encodeLayerR(enc, L, pos) {
    const D = this.dims;
    const seqLen = pos + 1;
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
      enc.copyBufferToBuffer(this.k, 0, L.kCache, pos * D.kvDim * 4, D.kvDim * 4);
      enc.copyBufferToBuffer(this.v, 0, L.vCache, pos * D.kvDim * 4, D.kvDim * 4);
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
      if (L.gu) this._dop(p, L.gu);
      else {
        this._dop(p, L.mvGate);
        this._dop(p, L.mvUp);
        this._d(p, "silu_mul", this.bgSilu, D.inter);
      }
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
      const si = rowB + b;
      const s = f16ToF32((e.scales[si >> 1] >>> ((si & 1) * 16)) & 0xFFFF);
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

  // ---- batched prefill (4 prompt tokens per pass) ----
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
  _dCol(pass, name, col, bg, threads, wg = 64) {
    pass.setPipeline(this.pipes[name]);
    pass.setBindGroup(0, this.bgCommonB[col][name] || this.bgCommonFor[name]);
    pass.setBindGroup(1, bg);
    pass.dispatchWorkgroups(Math.ceil(threads / wg));
  }

  _initBatch() {
    const D = this.dims;
    const dev = this.device;
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const al = (n) => Math.ceil(n * 4 / 256) * 256;
    const mkB = (n) => ({ buf: dev.createBuffer({ size: 4 * al(n), usage: S }), stride: al(n), n });
    const B = this.B = {
      x: mkB(D.dim), xn: mkB(D.dim), tmpDim: mkB(D.dim), g: mkB(D.inter), u: mkB(D.inter),
      qkv: mkB(D.convDim), convOut: mkB(D.convDim), z: mkB(D.dInner),
      alpha: mkB(D.nVH), betaRaw: mkB(D.nVH), beta: mkB(D.nVH), decay: mkB(D.nVH),
      dOut: mkB(D.dInner), gated: mkB(D.dInner),
      qFull: mkB(D.nH * D.hd * 2), q: mkB(D.qDim), gAttn: mkB(D.qDim),
      k: mkB(D.kvDim), v: mkB(D.kvDim), attnOut: mkB(D.qDim),
    };
    this.stageXB = dev.createBuffer({ size: 4 * D.dim * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    // rollback shadows for the recurrent layers (speculative decoding)
    for (const L of this.layers) if (!L.isFull && !L.S_shadow) {
      L.S_shadow = [0, 1, 2].map(() => dev.createBuffer({ size: L.S.size, usage: S }));
      L.conv_shadow = [0, 1, 2].map(() => dev.createBuffer({ size: L.convState.size, usage: S }));
    }
    this.stageLogitsN = dev.createBuffer({ size: 4 * D.vocab * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const slice = (b, c) => ({ buffer: b.buf, offset: c * b.stride, size: b.n * 4 });
    const part = (b, c, off, size) => ({ buffer: b.buf, offset: c * b.stride + off, size });
    this.frameBufsB = [0, 1, 2, 3].map(() => dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    const colPipes = ["rmsnorm", "head_norm", "attn_scores", "attn_softmax", "attn_out",
      "silu_mul", "add_res", "rope_part", "qsplit", "sigmoid_mul",
      "dn_gates", "dn_conv", "dn_l2", "dn_delta", "dn_gatenorm"];
    this.bgCommonB = [0, 1, 2, 3].map((c) => {
      const m = {};
      for (const name of colPipes)
        m[name] = this._bg2g0(this.pipes[name], [{ buffer: this.cfgBuf }, { buffer: this.frameBufsB[c] }]);
      return m;
    });
    const mvB = (w, xB, yB, dOut, dIn) => {
      const base = w.kind === "q8" ? "matvec_q8" : w.kind === "q4" ? "matvec_q4" : "matvec";
      const pipe = base + "_coop_b";
      const shp = this._shapeB(dOut, dIn, xB.stride / 16, yB.stride / 4);
      const bufs = w.kind === "f32" ? [w.buf, xB.buf, yB.buf, shp] : [w.qs, w.sc, xB.buf, yB.buf, shp];
      return { pipe, wgs: Math.ceil(dOut / this.coopRows), bg: this._bg(this.pipes[pipe], 1, bufs) };
    };
    if (this.hasHead) {
      this.bgFinalNormB = [0, 1, 2, 3].map((c) => this._bg2res(this.pipes.rmsnorm,
        [slice(B.x, c), { buffer: this.finalNorm.buf }, { buffer: this.xn }, { buffer: this.uDim }]));
      if (this.mtp) this.mtp.bgHNormB = [0, 1, 2, 3].map((c) => this._bg2res(this.pipes.rmsnorm,
        [slice(B.x, c), { buffer: this.mtp.hnorm.buf }, { buffer: this.mtp.ehIn, offset: D.dim * 4, size: D.dim * 4 }, { buffer: this.uDim }]));
    }
    this.layerB = this.layers.map((L) => {
      const bgNormC = (w, c) => this._bg2res(this.pipes.rmsnorm,
        [slice(B.x, c), { buffer: w.buf }, slice(B.xn, c), { buffer: this.uDim }]);
      const R = {
        gateUp: [mvB(L.ffnGate, B.xn, B.g, D.inter, D.dim), mvB(L.ffnUp, B.xn, B.u, D.inter, D.dim)],
        gu: this._guOp(L.ffnGate, L.ffnUp, B.xn.buf, B.g.buf, D.inter, D.dim, B.xn, B.g),
        down: mvB(L.ffnDown, B.g, B.tmpDim, D.dim, D.inter),
        cols: [0, 1, 2, 3].map((c) => ({
          norm1: bgNormC(L.attnNorm, c),
          norm2: bgNormC(L.postNorm, c),
          addTmp: this._bg2res(this.pipes.add_res, [slice(B.x, c), slice(B.tmpDim, c)]),
          silu: this._bg2res(this.pipes.silu_mul, [slice(B.g, c), slice(B.u, c)]),
        })),
      };
      if (L.isFull) {
        R.qkvOps = [mvB(L.wq, B.xn, B.qFull, D.nH * D.hd * 2, D.dim),
          mvB(L.wk, B.xn, B.k, D.kvDim, D.dim), mvB(L.wv, B.xn, B.v, D.kvDim, D.dim)];
        R.o = mvB(L.wo, B.attnOut, B.tmpDim, D.dim, D.qDim);
        for (let c = 0; c < 4; c++) Object.assign(R.cols[c], {
          qsplit: this._bg2res(this.pipes.qsplit, [slice(B.qFull, c), slice(B.q, c), slice(B.gAttn, c), { buffer: this.dnBuf }]),
          qNorm: this._bg2res(this.pipes.head_norm, [slice(B.q, c), { buffer: L.qNorm.buf }, { buffer: this.uNH }]),
          kNorm: this._bg2res(this.pipes.head_norm, [slice(B.k, c), { buffer: L.kNorm.buf }, { buffer: this.uNKV }]),
          ropeQ: this._bg2res(this.pipes.rope_part, [slice(B.q, c), { buffer: this.uNH }, { buffer: this.dnBuf }]),
          ropeK: this._bg2res(this.pipes.rope_part, [slice(B.k, c), { buffer: this.uNKV }, { buffer: this.dnBuf }]),
          scores: this._bg2res(this.pipes.attn_scores, [slice(B.q, c), { buffer: L.kCache }, { buffer: this.scores }]),
          softmax: this._bg2res(this.pipes.attn_softmax, [{ buffer: this.scores }]),
          attnOut: this._bg2res(this.pipes.attn_out, [{ buffer: this.scores }, { buffer: L.vCache }, slice(B.attnOut, c)]),
          sigMul: this._bg2res(this.pipes.sigmoid_mul, [slice(B.attnOut, c), slice(B.gAttn, c), { buffer: this.uQDim }]),
        });
      } else {
        R.dnOps = [mvB(L.wqkv, B.xn, B.qkv, D.convDim, D.dim), mvB(L.wz, B.xn, B.z, D.dInner, D.dim),
          mvB(L.wBeta, B.xn, B.betaRaw, D.nVH, D.dim), mvB(L.wAlpha, B.xn, B.alpha, D.nVH, D.dim)];
        R.out = mvB(L.wOut, B.gated, B.tmpDim, D.dim, D.dInner);
        for (let c = 0; c < 4; c++) Object.assign(R.cols[c], {
          gates: this._bg2res(this.pipes.dn_gates, [slice(B.alpha, c), slice(B.betaRaw, c),
            { buffer: L.dtBias }, { buffer: L.ssmA }, slice(B.beta, c), slice(B.decay, c), { buffer: this.dnBuf }]),
          conv: this._bg2res(this.pipes.dn_conv, [slice(B.qkv, c), { buffer: L.convW },
            { buffer: L.convState }, slice(B.convOut, c), { buffer: this.dnBuf }]),
          l2q: this._bg2res(this.pipes.dn_l2, [part(B.convOut, c, 0, D.keyDim * 4),
            { buffer: this.uNKH }, { buffer: this.dnBuf }]),
          l2k: this._bg2res(this.pipes.dn_l2, [part(B.convOut, c, D.keyDim * 4, D.keyDim * 4),
            { buffer: this.uNKH }, { buffer: this.dnBuf }]),
          delta: this._bg2res(this.pipes.dn_delta, [part(B.convOut, c, 0, D.keyDim * 4),
            part(B.convOut, c, D.keyDim * 4, D.keyDim * 4),
            part(B.convOut, c, D.keyDim * 2 * 4, D.dInner * 4),
            slice(B.beta, c), slice(B.decay, c), { buffer: L.S }, slice(B.dOut, c), { buffer: this.dnBuf }]),
          gatenorm: this._bg2res(this.pipes.dn_gatenorm, [slice(B.dOut, c), slice(B.z, c),
            { buffer: L.ssmNorm }, slice(B.gated, c), { buffer: this.dnBuf }]),
        });
      }
      return R;
    });
  }

  // nCols < 4: batched matvecs still compute 4 columns (extra columns are
  // garbage into scratch), but every stateful per-column op runs only for the
  // live columns. snapshotDN: copy recurrent state after column 0 so a
  // rejected speculative column 1 can be rolled back.
  _encodeLayerBatch(enc, i, basePos, nCols = 4, snapshotDN = false) {
    const D = this.dims, L = this.layers[i], LB = this.layerB[i], B = this.B;
    if (L.isFull) {
      {
        const p = enc.beginComputePass();
        for (let c = 0; c < nCols; c++) this._dCol(p, "rmsnorm", c, LB.cols[c].norm1, 256, 256);
        for (const op of LB.qkvOps) this._dop(p, op);
        for (let c = 0; c < nCols; c++) {
          const C = LB.cols[c];
          this._dCol(p, "qsplit", c, C.qsplit, D.nH * D.hd);
          this._dCol(p, "head_norm", c, C.qNorm, D.nH, 32);
          this._dCol(p, "head_norm", c, C.kNorm, D.nKV, 32);
          this._dCol(p, "rope_part", c, C.ropeQ, D.nH * D.nRot / 2);
          this._dCol(p, "rope_part", c, C.ropeK, D.nKV * D.nRot / 2);
        }
        p.end();
      }
      for (let c = 0; c < nCols; c++) {
        enc.copyBufferToBuffer(B.k.buf, c * B.k.stride, L.kCache, (basePos + c) * D.kvDim * 4, D.kvDim * 4);
        enc.copyBufferToBuffer(B.v.buf, c * B.v.stride, L.vCache, (basePos + c) * D.kvDim * 4, D.kvDim * 4);
      }
      {
        const p = enc.beginComputePass();
        for (let c = 0; c < nCols; c++) {
          const C = LB.cols[c];
          this._dCol(p, "attn_scores", c, C.scores, D.nH * (basePos + c + 1));
          this._dCol(p, "attn_softmax", c, C.softmax, D.nH, 1);
          this._dCol(p, "attn_out", c, C.attnOut, D.qDim);
          this._dCol(p, "sigmoid_mul", c, C.sigMul, D.qDim);
        }
        this._dop(p, LB.o);
        for (let c = 0; c < nCols; c++) this._dCol(p, "add_res", c, LB.cols[c].addTmp, D.dim);
        p.end();
      }
    } else {
      {
        const p = enc.beginComputePass();
        for (let c = 0; c < nCols; c++) this._dCol(p, "rmsnorm", c, LB.cols[c].norm1, 256, 256);
        for (const op of LB.dnOps) this._dop(p, op);
        p.end();
      }
      let p = snapshotDN ? null : enc.beginComputePass();
      for (let c = 0; c < nCols; c++) {   // recurrent state: columns strictly in order
        const C = LB.cols[c];
        if (snapshotDN) p = enc.beginComputePass();
        this._dCol(p, "dn_gates", c, C.gates, D.nVH);
        this._dCol(p, "dn_conv", c, C.conv, D.convDim);
        this._dCol(p, "dn_l2", c, C.l2q, D.nKH, 32);
        this._dCol(p, "dn_l2", c, C.l2k, D.nKH, 32);
        this._dCol(p, "dn_delta", c, C.delta, D.nVH * 128, 128);
        this._dCol(p, "dn_gatenorm", c, C.gatenorm, D.nVH * 128, 128);
        if (snapshotDN) p.end();
        if (snapshotDN && c < nCols - 1) {
          enc.copyBufferToBuffer(L.S, 0, L.S_shadow[c], 0, L.S.size);
          enc.copyBufferToBuffer(L.convState, 0, L.conv_shadow[c], 0, L.convState.size);
        }
      }
      if (!snapshotDN) p.end();
      {
        const p = enc.beginComputePass();
        this._dop(p, LB.out);
        for (let c = 0; c < nCols; c++) this._dCol(p, "add_res", c, LB.cols[c].addTmp, D.dim);
        p.end();
      }
    }
    {
      const p = enc.beginComputePass();
      for (let c = 0; c < nCols; c++) this._dCol(p, "rmsnorm", c, LB.cols[c].norm2, 256, 256);
      if (LB.gu) this._dop(p, LB.gu);
      else {
        for (const op of LB.gateUp) this._dop(p, op);
        for (let c = 0; c < nCols; c++) this._dCol(p, "silu_mul", c, LB.cols[c].silu, D.inter);
      }
      this._dop(p, LB.down);
      for (let c = 0; c < nCols; c++) this._dCol(p, "add_res", c, LB.cols[c].addTmp, D.dim);
      p.end();
    }
  }

  async _runBatchAndRead(basePos) {
    const { dim } = this.dims;
    const enc = this.device.createCommandEncoder();
    for (let l = 0; l < this.layers.length; l++) this._encodeLayerBatch(enc, l, basePos);
    for (let c = 0; c < 4; c++) enc.copyBufferToBuffer(this.B.x.buf, c * this.B.x.stride, this.stageXB, c * dim * 4, dim * 4);
    this.device.queue.submit([enc.finish()]);
    await this.stageXB.mapAsync(GPUMapMode.READ);
    const out = Float32Array.from(new Float32Array(this.stageXB.getMappedRange(), 0, 4 * dim));
    this.stageXB.unmap();
    this.pos = basePos + 4;
    return out;
  }
  async embedRunBatch(ids, basePos) {
    if (!this.B) this._initBatch();
    this.pos = basePos;
    for (let c = 0; c < 4; c++) {
      this.device.queue.writeBuffer(this.frameBufsB[c], 0, new Uint32Array([basePos + c, basePos + c + 1]));
      this.device.queue.writeBuffer(this.B.x.buf, c * this.B.x.stride, this._embedRowF32(ids[c]));
    }
    return this._runBatchAndRead(basePos);
  }
  async runHiddenBatch(xs, basePos) {
    if (!this.B) this._initBatch();
    const { dim } = this.dims;
    this.pos = basePos;
    for (let c = 0; c < 4; c++) {
      this.device.queue.writeBuffer(this.frameBufsB[c], 0, new Uint32Array([basePos + c, basePos + c + 1]));
      this.device.queue.writeBuffer(this.B.x.buf, c * this.B.x.stride, xs.subarray(c * dim, (c + 1) * dim));
    }
    return this._runBatchAndRead(basePos);
  }

  // ---- speculative decoding with the MTP head ----
  // Run the draft block for the token `tNext` (at position `pos`) given the
  // trunk hidden of the previous position: srcCol === null reads this.x,
  // otherwise batch column srcCol. Appends to the MTP layer's own KV cache.
  // wantLogits -> returns draft logits (argmax = drafted token).
  async mtpRun(srcCol, tNext, pos, wantLogits) {
    const M2 = this.mtp, { dim, vocab } = this.dims;
    this.device.queue.writeBuffer(M2.emb, 0, this._embedRowF32(tNext));
    this._setFrame(pos, pos + 1);
    const enc = this.device.createCommandEncoder();
    {
      const p = enc.beginComputePass();
      this._d(p, "rmsnorm", M2.bgENorm, 256, 256);
      this._d(p, "rmsnorm", srcCol === null ? M2.bgHNormX : M2.bgHNormB[srcCol], 256, 256);
      this._dop(p, M2.proj);
      p.end();
    }
    this._encodeLayerR(enc, this.mtpLayer, pos);
    if (wantLogits) {
      const p = enc.beginComputePass();
      this._d(p, "rmsnorm", M2.bgHeadNorm, 256, 256);
      this._dop(p, this.headOp);
      p.end();
    }
    this.device.queue.submit([enc.finish()]);
    if (!wantLogits) return null;
    return await this._readback(this.logits, this.stageLogits, vocab);
  }

  // batched verify: tokens[k] at position pos+k (2..4 tokens) -> logits for
  // every column, one readback. DeltaNet state is snapshotted after each
  // non-final column so any rejected suffix can be undone.
  async verifyN(tokens, pos) {
    if (!this.B) this._initBatch();
    const { vocab } = this.dims, n = tokens.length;
    for (let c = 0; c < n; c++) {
      this.device.queue.writeBuffer(this.frameBufsB[c], 0, new Uint32Array([pos + c, pos + c + 1]));
      this.device.queue.writeBuffer(this.B.x.buf, c * this.B.x.stride, this._embedRowF32(tokens[c]));
    }
    const enc = this.device.createCommandEncoder();
    for (let l = 0; l < this.layers.length; l++) this._encodeLayerBatch(enc, l, pos, n, true);
    for (let c = 0; c < n; c++) {
      const p = enc.beginComputePass();
      this._d(p, "rmsnorm", this.bgFinalNormB[c], 256, 256);
      this._dop(p, this.headOp);
      p.end();
      enc.copyBufferToBuffer(this.logits, 0, this.stageLogitsN, c * vocab * 4, vocab * 4);
    }
    this.device.queue.submit([enc.finish()]);
    await this.stageLogitsN.mapAsync(GPUMapMode.READ, 0, n * vocab * 4);
    const all = new Float32Array(this.stageLogitsN.getMappedRange(0, n * vocab * 4)).slice();
    this.stageLogitsN.unmap();
    const out = [];
    for (let c = 0; c < n; c++) out.push(all.subarray(c * vocab, (c + 1) * vocab));
    return out;
  }
  _restoreDN(k) {   // recurrent state as it was after verify column k
    const enc = this.device.createCommandEncoder();
    for (const L of this.layers) if (!L.isFull) {
      enc.copyBufferToBuffer(L.S_shadow[k], 0, L.S, 0, L.S.size);
      enc.copyBufferToBuffer(L.conv_shadow[k], 0, L.convState, 0, L.convState.size);
    }
    this.device.queue.submit([enc.finish()]);
  }
  _adoptHidden(col) {   // batch column -> this.x (the hidden the next draft reads)
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.B.x.buf, col * this.B.x.stride, this.x, 0, this.dims.dim * 4);
    this.device.queue.submit([enc.finish()]);
  }

  // One speculative step with K chained drafts (K <= 3). Precondition: this.x
  // holds the trunk hidden of the previous position and `tNext` is the
  // already-sampled token for this.pos. Returns 1..K+1 new tokens.
  async specStep(tNext, sample, K = 3) {
    const pos = this.pos, M2 = this.mtp;
    K = Math.max(1, Math.min(3, K));
    const drafts = [];
    for (let k = 0; k < K; k++) {
      // after the first call this.x holds the MTP block's own output hidden,
      // which is what chained drafting feeds back in
      const lg = await this.mtpRun(null, k === 0 ? tNext : drafts[k - 1], pos + k, true);
      let d = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[d]) d = i;
      drafts.push(d);
    }
    const lgs = await this.verifyN([tNext, ...drafts], pos);
    const out = [];
    let a = 0;   // accepted drafts
    for (let k = 0; k <= K; k++) {
      const t = sample(lgs[k]);
      out.push(t);
      if (k < K && t === drafts[k]) a++; else break;
    }
    M2.stats.drafts += K; M2.stats.accepted += a;
    if (a < K) this._restoreDN(a);
    // re-fill the draft cache for the accepted positions with exact trunk hiddens
    for (let j = 1; j <= a; j++) await this.mtpRun(j - 1, out[j - 1], pos + j, false);
    this._adoptHidden(a);
    this.pos = pos + a + 1;
    return out;
  }

  async prefillTokens(ids) {
    if (!this.B) this._initBatch();
    let i = 0, sinceSync = 0;
    while (ids.length - i >= 4) {
      const basePos = this.pos;
      for (let c = 0; c < 4; c++) {
        this.device.queue.writeBuffer(this.frameBufsB[c], 0, new Uint32Array([basePos + c, basePos + c + 1]));
        this.device.queue.writeBuffer(this.B.x.buf, c * this.B.x.stride, this._embedRowF32(ids[i + c]));
      }
      const enc = this.device.createCommandEncoder();
      for (let l = 0; l < this.layers.length; l++) this._encodeLayerBatch(enc, l, basePos);
      enc.copyBufferToBuffer(this.B.x.buf, 3 * this.B.x.stride, this.x, 0, this.dims.dim * 4);
      this.device.queue.submit([enc.finish()]);
      if (this.mtp && this.mtpFill !== false)
        for (let c = 0; c < 4; c++) if (i + c + 1 < ids.length) await this.mtpRun(c, ids[i + c + 1], basePos + c + 1, false);
      this.pos += 4;
      i += 4;
      if (++sinceSync >= 4) { await this.device.queue.onSubmittedWorkDone(); sinceSync = 0; }
    }
    for (; i < ids.length; i++) {
      await this.prefillToken(ids[i]);
      if (this.mtp && this.mtpFill !== false && i + 1 < ids.length) await this.mtpRun(null, ids[i + 1], i + 1, false);
      if (i % 8 === 7) await this.device.queue.onSubmittedWorkDone();
    }
    await this.device.queue.onSubmittedWorkDone();
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
