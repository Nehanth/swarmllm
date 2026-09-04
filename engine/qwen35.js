// Qwen35Engine: the hybrid Gated-DeltaNet + attention engine (Qwen 3.5/3.6/3.8),
// with batched prefill/verify paths and multi-token-prediction speculation.
// See docs/architecture.md.
// Qwen3.5/3.8 engine: hybrid Gated-DeltaNet + gated-attention WebGPU
// inference, layer-shardable like DenseEngine. Golden reference: ref_q38.mjs
// (validated line-by-line against llama.cpp eval-callback dumps).
import { WGSL } from "./wgsl/base.js";
import { gemmWGSL, GEMM_S, GEMM_TILE } from "./wgsl/gemm.js";
import { coopWGSL, probeUnpack } from "./wgsl/coop.js";
import { WGSL2 } from "./wgsl/qwen35.js";
import { f16ToF32 } from "./gguf.js";



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
  async _init({ device, meta, weights, layerRange, hasEmbed = true, hasHead = true, maxSeq = 512, vocab: vocabOpt, matvecVariant = "coop", coopWG = 256, coopRows = 4, batchCols = 4, coopRowsB = coopRows, gemm = true }) {
    this.device = device;
    this.mvVariant = matvecVariant;
    this.coopWG = coopWG; this.coopRows = coopRows;
    this.NC = batchCols; this.coopRowsB = coopRowsB;   // batched (prefill/verify) column count, rows per WG
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

    // ---- prefill GEMM plan (docs/research/prefill-gemm-v2.md) ----
    // Only at the full batch width, only for Q4 weights, only for shapes with a
    // pinned split-K factor. Narrower passes (decode, speculative verify, the
    // prompt tail) stay on the GEMV ladder, so the speculative stream is
    // structurally identical to plain decoding.
    this._gemmShapes = new Map();
    if (batchCols >= 16 && gemm !== false) {
      for (const [dOut, dIn] of [[convDim, dim], [dInner, dim], [dim, dInner], [inter, dim], [dim, inter],
                                 [nH * hd * 2, dim], [kvDim, dim], [dim, qDim]]) {
        const S = GEMM_S[`${dOut}x${dIn}`];
        if (S && dOut >= 256 && dIn % 64 === 0 && ((dIn / 32) / 2) % S === 0) this._gemmShapes.set(`${dOut}x${dIn}`, S);
      }
    }
    this.gemmOn = this._gemmShapes.size > 0;
    this._gemmDIns = [...new Set([...this._gemmShapes.keys()].map((k) => +k.split("x")[1]))];
    this._gemmSplits = [...new Set(this._gemmShapes.values())];
    this._gemmPairs = [...new Set([...this._gemmShapes].map(([k, S]) => `${k.split("x")[1]}:${S}`))].map((x) => x.split(":").map(Number));
    this.gemm = this.gemmOn;   // runtime kill switch: engine.gemm = false reproduces the GEMV path

    // ---- pipelines with explicit layouts ----
    const unpack = await probeUnpack(device);
    const mod = device.createShaderModule({ code: WGSL + coopWGSL(coopWG, coopRows, 64, batchCols, coopRowsB, unpack)
      + (this.gemmOn ? gemmWGSL({ N: batchCols, pairs: this._gemmPairs, UNPACK: unpack }) : "") + WGSL2 });
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
      matvec_coop_acc: ["ro", "ro", "rw", "u"], matvec_q8_coop_acc: ["ro", "ro", "ro", "rw", "u"], matvec_q4_coop_acc: ["ro", "ro", "ro", "rw", "u"],
      matvec_coop_b_acc: ["ro", "ro", "rw", "u"], matvec_q8_coop_b_acc: ["ro", "ro", "ro", "rw", "u"], matvec_q4_coop_b_acc: ["ro", "ro", "ro", "rw", "u"],
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
      rmsnorm_mc: ["ro", "ro", "rw", "u"], add_res_mc: ["rw", "ro", "u"],
      dn_gates_mc: ["ro", "ro", "ro", "ro", "rw", "rw", "u"], dn_conv_mc: ["ro", "ro", "rw", "rw", "u", "rw"],
      dn_pre: ["ro", "ro", "ro", "ro", "rw", "rw", "rw", "u"], dn_pre_mc: ["ro", "ro", "ro", "ro", "rw", "rw", "rw", "u", "u"],
      dn_l2_mc: ["rw", "u", "u"], dn_delta_mc: ["ro", "ro", "ro", "rw", "rw", "u", "u", "rw"],
      dn_gatenorm_mc: ["ro", "ro", "ro", "rw", "u", "u"], qsplit_mc: ["ro", "rw", "rw", "u", "u"],
      head_norm_mc: ["rw", "ro", "u"], rope_part_mc: ["rw", "u", "u"], sigmoid_mul_mc: ["rw", "ro", "u"],
      argmax: ["ro", "rw", "u"],
    };
    // narrower twins: a verify or tail pass with w live columns pays for w, not batchCols
    for (const W of [8, 4]) if (batchCols > W) Object.assign(G1, {
      [`matvec_coop_b${W}`]: G1.matvec_coop_b, [`matvec_q8_coop_b${W}`]: G1.matvec_q8_coop_b, [`matvec_q4_coop_b${W}`]: G1.matvec_q4_coop_b,
      [`matvec_coop_b${W}_acc`]: G1.matvec_coop_b, [`matvec_q8_coop_b${W}_acc`]: G1.matvec_q8_coop_b, [`matvec_q4_coop_b${W}_acc`]: G1.matvec_q4_coop_b,
      [`matvec_gu_b${W}`]: G1.matvec_gu_b, [`matvec_q8_gu_b${W}`]: G1.matvec_q8_gu_b, [`matvec_q4_gu_b${W}`]: G1.matvec_q4_gu_b,
    });
    // the prefill GEMM and its split-K reduce / transpose all reuse the
    // matvec_q4_coop_b layout (qs, sc, x, y, shape) verbatim
    if (this.gemmOn) {
      for (const [dIn, S] of this._gemmPairs) G1[`gemm_q4_${dIn}_s${S}`] = G1.matvec_q4_coop_b;
      for (const S of this._gemmSplits) { G1[`gemm_red_s${S}`] = G1.matvec_q4_coop_b; G1[`gemm_red_s${S}_acc`] = G1.matvec_q4_coop_b; }
      G1.gemm_xpose = G1.matvec_q4_coop_b;
    }
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
    // acc: y[row] += W x (coop only) -> the residual add needs no dispatch
    const mv = (w, x, y, dOut, dIn, acc = false) => {
      const base = w.kind === "q8" ? "matvec_q8" : w.kind === "q4" ? "matvec_q4" : "matvec";
      acc = acc && coop;
      const pipe = coop ? base + "_coop" + (acc ? "_acc" : "") : base;
      const bufs = w.kind === "f32" ? [w.buf, x, y, this._shape(dOut, dIn)] : [w.qs, w.sc, x, y, this._shape(dOut, dIn)];
      return { pipe, acc, wgs: coop ? Math.ceil(dOut / this.coopRows) : Math.ceil(dOut / 64), bg: this._bg(this.pipes[pipe], 1, bufs) };
    };
    this._mv = mv;
    const guOp = (wg2, wu2, x, y, dOut, dIn, xB, yB) => {
      if (!coop || !wg2 || !wu2 || wg2.kind !== wu2.kind) return null;
      const base = wg2.kind === "q8" ? "matvec_q8_gu" : wg2.kind === "q4" ? "matvec_q4_gu" : "matvec_gu";
      const pipe = xB ? base + "_b" : base;
      const shp = xB ? this._shapeB(dOut, dIn, xB.stride / 16, yB.stride / 4) : this._shapeB(dOut, dIn, 0, 0);
      const bufs = wg2.kind === "f32" ? [wg2.buf, wu2.buf, x, y, shp] : [wg2.qs, wg2.sc, wu2.qs, wu2.sc, x, y, shp];
      const op = { pipe, wgs: Math.ceil(dOut / (xB ? this._rowsFor(this.NC) : this.coopRows)), bg: this._bg(this.pipes[pipe], 1, bufs) };
      if (xB) for (const W of [8, 4]) if (this.NC > W) {
        op[`pipe${W}`] = `${base}_b${W}`;
        op[`bg${W}`] = this._bg(this.pipes[op[`pipe${W}`]], 1, bufs);
        op[`wgs${W}`] = Math.ceil(dOut / this._rowsFor(W));
      }
      return op;
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
      R.mvDown = coop ? mv(R.ffnDown, this.g, this.x, dim, inter, true) : mv(R.ffnDown, this.g, this.tmpDim, dim, inter);
      if (L.isFull) {
        R.wq = up(L.wq); R.wk = up(L.wk); R.wv = up(L.wv); R.wo = up(L.wo);
        R.qNorm = up(L.qNorm); R.kNorm = up(L.kNorm);
        R.kCache = device.createBuffer({ size: maxSeq * kvDim * 4, usage: S });
        R.vCache = device.createBuffer({ size: maxSeq * kvDim * 4, usage: S });
        R.mvQ = mv(R.wq, this.xn, this.qFull, nH * hd * 2, dim);
        R.mvK = mv(R.wk, this.xn, this.k, kvDim, dim);
        R.mvV = mv(R.wv, this.xn, this.v, kvDim, dim);
        R.mvO = coop ? mv(R.wo, this.attnOut, this.x, dim, qDim, true) : mv(R.wo, this.attnOut, this.tmpDim, dim, qDim);
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
        R.mvOut = coop ? mv(R.wOut, this.gated, this.x, dim, dInner, true) : mv(R.wOut, this.gated, this.tmpDim, dim, dInner);
        R.bgConv = this._bg(this.pipes.dn_conv, 1, [this.qkv, R.convW, R.convState, this.convOut, this.dnBuf]);
        R.bgGates = this._bg(this.pipes.dn_gates, 1, [this.alpha, this.betaRaw, R.dtBias, R.ssmA, this.beta, this.decay, this.dnBuf]);
        R.bgPre = this._bg(this.pipes.dn_pre, 1, [this.alpha, this.betaRaw, R.dtBias, R.ssmA, this.beta, this.decay, this.convOut, this.dnBuf]);
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
      // on-GPU argmax of the logits (draft chain): 8-byte readback instead of 1 MB
      this.argBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      this.stageArg = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      this.bgArgmax = this._bg(this.pipes.argmax, 1, [this.logits, this.argBuf, this._buf(new Uint32Array([vocab, 0, 0, 0]), GPUBufferUsage.UNIFORM)]);
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
    if (this.skip && this.skip.has(name)) return;   // profiling aid (bench_breakdown)
    pass.setPipeline(this.pipes[name]);
    pass.setBindGroup(0, this.bgCommonFor[name]);
    pass.setBindGroup(1, bg);
    pass.dispatchWorkgroups(Math.ceil(threads / wg));
  }
  _dop(pass, op, nCols = 0) {
    if (this.skip && this.skip.has(op.pipe)) return;
    // full-width prefill passes go through the row-stationary GEMM; anything
    // narrower (decode, speculative verify, prompt tail) uses the GEMV ladder
    if (op.gemm && nCols === this.NC && this.gemm !== false) {
      const g = op.gemm, z = (this._gz ^= 1);
      this._d3(pass, g.pipe, g.bg[z], g.wgs);
      this._d3(pass, g.red, g.redBg[z], g.redWgs);
      return;
    }
    const w = this.b4 === false ? this.NC : (nCols > 0 ? nCols : this.NC);
    const W = w <= 4 && op.pipe4 ? 4 : w <= 8 && op.pipe8 ? 8 : 0;
    this._d3(pass, W ? op[`pipe${W}`] : op.pipe, W ? op[`bg${W}`] : op.bg, W ? (op[`wgs${W}`] ?? op.wgs) : op.wgs);
  }
  // rows per workgroup for a W-column batched kernel; mirrors rowsFor() in coop.js
  _rowsFor(W) { return Math.max(1, Math.min(8, Math.round(this.coopRowsB * this.NC / W))); }
  _d3(pass, pipe, bg, wgs) {
    pass.setPipeline(this.pipes[pipe]);
    pass.setBindGroup(0, this.bgCommonFor[pipe]);
    pass.setBindGroup(1, bg);
    if (wgs > 32768) pass.dispatchWorkgroups(32768, Math.ceil(wgs / 32768)); else pass.dispatchWorkgroups(wgs);   // > 65535 per dimension is silently dropped
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
        if (!L.mvO.acc) this._d(p, "add_res", this.bgAddTmp, D.dim);
        p.end();
      }
    } else {
      const p = enc.beginComputePass();
      this._d(p, "rmsnorm", L.bgNorm1, 256, 256);
      this._dop(p, L.mvQKV);
      this._dop(p, L.mvZ);
      this._dop(p, L.mvBeta);
      this._dop(p, L.mvAlpha);
      this._d(p, "dn_conv", L.bgConv, D.convDim);
      this._d(p, "dn_pre", L.bgPre, 128, 128);      // gates + L2(q,k) fused
      this._d(p, "dn_delta", L.bgDelta, D.nVH * 128, 128);
      this._d(p, "dn_gatenorm", L.bgGateNorm, D.nVH * 128, 128);
      this._dop(p, L.mvOut);
      if (!L.mvOut.acc) this._d(p, "add_res", this.bgAddTmp, D.dim);
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
      if (!L.mvDown.acc) this._d(p, "add_res", this.bgAddTmp, D.dim);
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
  _dMC(pass, name, bg, threads, wg, ny, nz = 1) {
    if (this.skip && this.skip.has(name)) return;   // profiling aid (bench_breakdown)
    pass.setPipeline(this.pipes[name]);
    pass.setBindGroup(0, this.bgCommonB[0][name]);
    pass.setBindGroup(1, bg);
    pass.dispatchWorkgroups(Math.ceil(threads / wg), ny, nz);
  }
  _dCol(pass, name, col, bg, threads, wg = 64) {
    if (this.skip && this.skip.has(name)) return;
    pass.setPipeline(this.pipes[name]);
    pass.setBindGroup(0, this.bgCommonB[col][name] || this.bgCommonFor[name]);
    pass.setBindGroup(1, bg);
    pass.dispatchWorkgroups(Math.ceil(threads / wg));
  }

  _initBatch() {
    const D = this.dims;
    const dev = this.device;
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const NC = this.NC, cix = Array.from({ length: NC }, (_, c) => c);
    const al = (n) => Math.ceil(n * 4 / 256) * 256;
    const mkB = (n) => ({ buf: dev.createBuffer({ size: NC * al(n), usage: S }), stride: al(n), n });
    const B = this.B = {
      x: mkB(D.dim), xn: mkB(D.dim), tmpDim: mkB(D.dim), g: mkB(D.inter), u: mkB(D.inter),
      qkv: mkB(D.convDim), convOut: mkB(D.convDim), z: mkB(D.dInner),
      alpha: mkB(D.nVH), betaRaw: mkB(D.nVH), beta: mkB(D.nVH), decay: mkB(D.nVH),
      dOut: mkB(D.dInner), gated: mkB(D.dInner),
      qFull: mkB(D.nH * D.hd * 2), q: mkB(D.qDim), gAttn: mkB(D.qDim),
      k: mkB(D.kvDim), v: mkB(D.kvDim), attnOut: mkB(D.qDim),
    };
    this.stageXB = dev.createBuffer({ size: NC * D.dim * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    if (this.gemmOn) {
      // column-major copies of the three activation tensors the GEMM reads, and
      // two partials buffers so consecutive GEMMs in one pass do not alias
      const T = (n) => dev.createBuffer({ size: n * NC * 4, usage: S });
      B.xnT = T(D.dim); B.gT = T(D.inter); B.aoT = T(Math.max(D.qDim, D.dInner));
      const maxPart = Math.max(...[...this._gemmShapes].map(([k, sp]) => sp * NC * +k.split("x")[0]));
      this.gemmP = [0, 1].map(() => dev.createBuffer({ size: maxPart * 4, usage: S }));
      this._gz = 0;
      const xp = (src, dst, dIn) => ({ pipe: "gemm_xpose", wgs: Math.ceil(dIn * NC / 64),
        bg: this._bg(this.pipes.gemm_xpose, 1, [src.buf, src.buf, src.buf, dst, this._shapeB(0, dIn, src.stride / 16, 0)]) });
      this.xposeXn = xp(B.xn, B.xnT, D.dim);
      this.xposeG = xp(B.g, B.gT, D.inter);
      this.xposeGated = xp(B.gated, B.aoT, D.dInner);
      this.xposeAttnOut = xp(B.attnOut, B.aoT, D.qDim);
    }
    // rollback shadows for the recurrent layers (speculative decoding)
    for (const L of this.layers) if (!L.isFull && !L.S_shadow) {
      L.S_shadow = dev.createBuffer({ size: 7 * L.S.size, usage: S });
      L.conv_shadow = dev.createBuffer({ size: 7 * L.convState.size, usage: S });
    }
    this.stageLogitsN = dev.createBuffer({ size: NC * D.vocab * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const slice = (b, c) => ({ buffer: b.buf, offset: c * b.stride, size: b.n * 4 });
    const part = (b, c, off, size) => ({ buffer: b.buf, offset: c * b.stride + off, size });
    this.frameBufsB = cix.map(() => dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    const colPipes = ["rmsnorm", "head_norm", "attn_scores", "attn_softmax", "attn_out",
      "silu_mul", "add_res", "rope_part", "qsplit", "sigmoid_mul",
      "dn_gates", "dn_conv", "dn_l2", "dn_delta", "dn_gatenorm",
      "rmsnorm_mc", "add_res_mc", "dn_gates_mc", "dn_conv_mc", "dn_l2_mc", "dn_pre_mc", "dn_delta_mc", "dn_gatenorm_mc",
      "qsplit_mc", "head_norm_mc", "rope_part_mc", "sigmoid_mul_mc"];
    this.bgCommonB = cix.map((c) => {
      const m = {};
      for (const name of colPipes)
        m[name] = this._bg2g0(this.pipes[name], [{ buffer: this.cfgBuf }, { buffer: this.frameBufsB[c] }]);
      return m;
    });
    const mvB = (w, xB, yB, dOut, dIn, acc = false, xT = null) => {
      const base = w.kind === "q8" ? "matvec_q8" : w.kind === "q4" ? "matvec_q4" : "matvec";
      const pipe = base + "_coop_b" + (acc ? "_acc" : "");
      const shp = this._shapeB(dOut, dIn, xB.stride / 16, yB.stride / 4);
      const bufs = w.kind === "f32" ? [w.buf, xB.buf, yB.buf, shp] : [w.qs, w.sc, xB.buf, yB.buf, shp];
      const op = { pipe, acc, wgs: Math.ceil(dOut / this._rowsFor(this.NC)), bg: this._bg(this.pipes[pipe], 1, bufs) };
      for (const W of [8, 4]) if (this.NC > W) {
        op[`pipe${W}`] = `${base}_coop_b${W}${acc ? "_acc" : ""}`;
        op[`bg${W}`] = this._bg(this.pipes[op[`pipe${W}`]], 1, bufs);
        op[`wgs${W}`] = Math.ceil(dOut / this._rowsFor(W));   // narrower twins carry more rows per workgroup
      }
      const S2 = this._gemmShapes.get(`${dOut}x${dIn}`);
      if (S2 && w.kind === "q4" && xT) {
        const gp = `gemm_q4_${dIn}_s${S2}`, rp = `gemm_red_s${S2}${acc ? "_acc" : ""}`;
        op.gemm = {
          pipe: gp, wgs: Math.ceil(dOut / GEMM_TILE) * S2,
          bg: [0, 1].map((z) => this._bg(this.pipes[gp], 1, [w.qs, w.sc, xT, this.gemmP[z], shp])),
          red: rp, redBg: [0, 1].map((z) => this._bg(this.pipes[rp], 1, [this.gemmP[z], w.sc, w.sc, yB.buf, shp])),
          redWgs: Math.ceil(this.NC * dOut / 64),
        };
      }
      return op;
    };
    if (this.hasHead) {
      B.logits = mkB(D.vocab);
      this.headB = mvB(this.headEntry, B.xn, B.logits, D.vocab, D.dim);
      this.bgFinalNormB = cix.map((c) => this._bg2res(this.pipes.rmsnorm,
        [slice(B.x, c), { buffer: this.finalNorm.buf }, { buffer: this.xn }, { buffer: this.uDim }]));
      if (this.mtp) this.mtp.bgHNormB = cix.map((c) => this._bg2res(this.pipes.rmsnorm,
        [slice(B.x, c), { buffer: this.mtp.hnorm.buf }, { buffer: this.mtp.ehIn, offset: D.dim * 4, size: D.dim * 4 }, { buffer: this.uDim }]));
    }
    this._mcU = this._mcU || {};
    const mcU = (n, s0 = 0, s1 = 0, s2 = 0) => {
      const k = n + "," + s0 + "," + s1 + "," + s2;
      return this._mcU[k] || (this._mcU[k] = { buffer: this._buf(new Uint32Array([n, s0, s1, s2]), GPUBufferUsage.UNIFORM) });
    };
    const st = (b) => b.stride / 4;   // column stride in floats
    const whole = (b) => ({ buffer: b.buf });
    const dn = { buffer: this.dnBuf };
    if (this.hasHead) this.bgFinalNormMC = this._bg2res(this.pipes.rmsnorm_mc,
      [whole(B.x), { buffer: this.finalNorm.buf }, whole(B.xn), mcU(D.dim, st(B.x), st(B.xn))]);
    this.layerB = this.layers.map((L) => {
      const bgNormC = (w, c) => this._bg2res(this.pipes.rmsnorm,
        [slice(B.x, c), { buffer: w.buf }, slice(B.xn, c), { buffer: this.uDim }]);
      const mc = {
        norm1: this._bg2res(this.pipes.rmsnorm_mc, [whole(B.x), { buffer: L.attnNorm.buf }, whole(B.xn), mcU(D.dim, st(B.x), st(B.xn))]),
        norm2: this._bg2res(this.pipes.rmsnorm_mc, [whole(B.x), { buffer: L.postNorm.buf }, whole(B.xn), mcU(D.dim, st(B.x), st(B.xn))]),
        addTmp: this._bg2res(this.pipes.add_res_mc, [whole(B.x), whole(B.tmpDim), mcU(D.dim, st(B.x), st(B.tmpDim))]),
      };
      if (L.isFull) Object.assign(mc, {
        qsplit: this._bg2res(this.pipes.qsplit_mc, [whole(B.qFull), whole(B.q), whole(B.gAttn), mcU(0, st(B.qFull), st(B.q), st(B.gAttn)), dn]),
        qNorm: this._bg2res(this.pipes.head_norm_mc, [whole(B.q), { buffer: L.qNorm.buf }, mcU(D.nH, st(B.q))]),
        kNorm: this._bg2res(this.pipes.head_norm_mc, [whole(B.k), { buffer: L.kNorm.buf }, mcU(D.nKV, st(B.k))]),
        ropeQ: this._bg2res(this.pipes.rope_part_mc, [whole(B.q), mcU(D.nH, st(B.q)), dn]),
        ropeK: this._bg2res(this.pipes.rope_part_mc, [whole(B.k), mcU(D.nKV, st(B.k)), dn]),
        sigMul: this._bg2res(this.pipes.sigmoid_mul_mc, [whole(B.attnOut), whole(B.gAttn), mcU(D.qDim, st(B.attnOut), st(B.gAttn))]),
      });
      else Object.assign(mc, {
        gates: this._bg2res(this.pipes.dn_gates_mc, [whole(B.alpha), whole(B.betaRaw), { buffer: L.dtBias }, { buffer: L.ssmA },
          whole(B.beta), whole(B.decay), mcU(D.nVH, st(B.alpha))]),
        conv: this._bg2res(this.pipes.dn_conv_mc, [whole(B.qkv), { buffer: L.convW }, { buffer: L.convState }, whole(B.convOut),
          mcU(D.convDim, st(B.qkv), st(B.convOut)), { buffer: L.conv_shadow }]),
        l2: this._bg2res(this.pipes.dn_l2_mc, [whole(B.convOut), mcU(D.nKH, st(B.convOut), D.keyDim), dn]),
        pre: this._bg2res(this.pipes.dn_pre_mc, [whole(B.alpha), whole(B.betaRaw), { buffer: L.dtBias }, { buffer: L.ssmA },
          whole(B.beta), whole(B.decay), whole(B.convOut), mcU(0, st(B.alpha), st(B.convOut)), dn]),
        delta: this._bg2res(this.pipes.dn_delta_mc, [whole(B.convOut), whole(B.beta), whole(B.decay), { buffer: L.S }, whole(B.dOut),
          mcU(0, st(B.convOut), st(B.beta), st(B.dOut)), dn, { buffer: L.S_shadow }]),
        gatenorm: this._bg2res(this.pipes.dn_gatenorm_mc, [whole(B.dOut), whole(B.z), { buffer: L.ssmNorm }, whole(B.gated),
          mcU(0, st(B.dOut), st(B.z), st(B.gated)), dn]),
      });
      const R = {
        mc,
        gateUp: [mvB(L.ffnGate, B.xn, B.g, D.inter, D.dim, false, this.gemmOn ? B.xnT : null), mvB(L.ffnUp, B.xn, B.u, D.inter, D.dim, false, this.gemmOn ? B.xnT : null)],
        gu: this._guOp(L.ffnGate, L.ffnUp, B.xn.buf, B.g.buf, D.inter, D.dim, B.xn, B.g),
        down: mvB(L.ffnDown, B.g, B.x, D.dim, D.inter, true, this.gemmOn ? B.gT : null),
        cols: cix.map((c) => ({
          norm1: bgNormC(L.attnNorm, c),
          norm2: bgNormC(L.postNorm, c),
          addTmp: this._bg2res(this.pipes.add_res, [slice(B.x, c), slice(B.tmpDim, c)]),
          silu: this._bg2res(this.pipes.silu_mul, [slice(B.g, c), slice(B.u, c)]),
        })),
      };
      if (L.isFull) {
        R.qkvOps = [mvB(L.wq, B.xn, B.qFull, D.nH * D.hd * 2, D.dim, false, this.gemmOn ? B.xnT : null),
          mvB(L.wk, B.xn, B.k, D.kvDim, D.dim, false, this.gemmOn ? B.xnT : null), mvB(L.wv, B.xn, B.v, D.kvDim, D.dim, false, this.gemmOn ? B.xnT : null)];
        R.o = mvB(L.wo, B.attnOut, B.x, D.dim, D.qDim, true, this.gemmOn ? B.aoT : null);
        for (let c = 0; c < NC; c++) Object.assign(R.cols[c], {
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
        R.dnOps = [mvB(L.wqkv, B.xn, B.qkv, D.convDim, D.dim, false, this.gemmOn ? B.xnT : null), mvB(L.wz, B.xn, B.z, D.dInner, D.dim, false, this.gemmOn ? B.xnT : null),
          mvB(L.wBeta, B.xn, B.betaRaw, D.nVH, D.dim), mvB(L.wAlpha, B.xn, B.alpha, D.nVH, D.dim)];
        R.out = mvB(L.wOut, B.gated, B.x, D.dim, D.dInner, true, this.gemmOn ? B.aoT : null);
        for (let c = 0; c < NC; c++) Object.assign(R.cols[c], {
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
  // garbage into scratch); every per-column op runs as ONE multi-column
  // dispatch over the live columns. snapshotDN (set through frame.snap) makes
  // the recurrent kernels save their state after each non-final column so a
  // rejected speculative suffix can be rolled back.
  _encodeLayerBatch(enc, i, basePos, nCols = this.NC, snapshotDN = false) {
    const D = this.dims, L = this.layers[i], LB = this.layerB[i], B = this.B, M = LB.mc;
    const G = this.gemmOn && this.gemm !== false && nCols === this.NC;   // full-width pass: GEMM needs transposed activations
    if (L.isFull) {
      {
        const p = enc.beginComputePass();
        this._dMC(p, "rmsnorm_mc", M.norm1, 256, 256, nCols);
        if (G) this._dop(p, this.xposeXn);
        for (const op of LB.qkvOps) this._dop(p, op, nCols);
        this._dMC(p, "qsplit_mc", M.qsplit, D.nH * D.hd, 64, nCols);
        this._dMC(p, "head_norm_mc", M.qNorm, D.nH, 32, nCols);
        this._dMC(p, "head_norm_mc", M.kNorm, D.nKV, 32, nCols);
        this._dMC(p, "rope_part_mc", M.ropeQ, D.nH * D.nRot / 2, 64, nCols);
        this._dMC(p, "rope_part_mc", M.ropeK, D.nKV * D.nRot / 2, 64, nCols);
        p.end();
      }
      for (let c = 0; c < nCols; c++) {
        enc.copyBufferToBuffer(B.k.buf, c * B.k.stride, L.kCache, (basePos + c) * D.kvDim * 4, D.kvDim * 4);
        enc.copyBufferToBuffer(B.v.buf, c * B.v.stride, L.vCache, (basePos + c) * D.kvDim * 4, D.kvDim * 4);
      }
      {
        const p = enc.beginComputePass();
        for (let c = 0; c < nCols; c++) {   // shared score scratch: columns in turn
          const C = LB.cols[c];
          this._dCol(p, "attn_scores", c, C.scores, D.nH * (basePos + c + 1));
          this._dCol(p, "attn_softmax", c, C.softmax, D.nH, 1);
          this._dCol(p, "attn_out", c, C.attnOut, D.qDim);
        }
        this._dMC(p, "sigmoid_mul_mc", M.sigMul, D.qDim, 64, nCols);
        if (G) this._dop(p, this.xposeAttnOut);
        this._dop(p, LB.o, nCols);
        if (!LB.o.acc) this._dMC(p, "add_res_mc", M.addTmp, D.dim, 64, nCols);
        p.end();
      }
    } else {
      const p = enc.beginComputePass();
      this._dMC(p, "rmsnorm_mc", M.norm1, 256, 256, nCols);
      if (G) this._dop(p, this.xposeXn);
      for (const op of LB.dnOps) this._dop(p, op, nCols);
      this._dMC(p, "dn_conv_mc", M.conv, D.convDim, 64, 1);           // loops over columns
      this._dMC(p, "dn_pre_mc", M.pre, 128, 128, nCols);              // gates + L2(q,k) fused, one WG per column
      this._dMC(p, "dn_delta_mc", M.delta, D.nVH * 128, 128, 1);     // loops over columns
      this._dMC(p, "dn_gatenorm_mc", M.gatenorm, D.nVH * 128, 128, nCols);
      if (G) this._dop(p, this.xposeGated);
      this._dop(p, LB.out, nCols);
      if (!LB.out.acc) this._dMC(p, "add_res_mc", M.addTmp, D.dim, 64, nCols);
      p.end();
    }
    {
      const p = enc.beginComputePass();
      this._dMC(p, "rmsnorm_mc", M.norm2, 256, 256, nCols);
      if (G) this._dop(p, this.xposeXn);
      if (LB.gu && !G) this._dop(p, LB.gu, nCols);
      else {   // the GEMM has no fused gate/up: run them separately, then SiLU
        for (const op of LB.gateUp) this._dop(p, op, nCols);
        for (let c = 0; c < nCols; c++) this._dCol(p, "silu_mul", c, LB.cols[c].silu, D.inter);
      }
      if (G) this._dop(p, this.xposeG);
      this._dop(p, LB.down, nCols);
      if (!LB.down.acc) this._dMC(p, "add_res_mc", M.addTmp, D.dim, 64, nCols);
      p.end();
    }
  }

  async _runBatchAndRead(basePos, n = this.NC) {
    const { dim } = this.dims;
    const enc = this.device.createCommandEncoder();
    for (let l = 0; l < this.layers.length; l++) this._encodeLayerBatch(enc, l, basePos, n);
    for (let c = 0; c < n; c++) enc.copyBufferToBuffer(this.B.x.buf, c * this.B.x.stride, this.stageXB, c * dim * 4, dim * 4);
    this.device.queue.submit([enc.finish()]);
    await this.stageXB.mapAsync(GPUMapMode.READ, 0, n * dim * 4);
    const out = Float32Array.from(new Float32Array(this.stageXB.getMappedRange(0, n * dim * 4), 0, n * dim));
    this.stageXB.unmap();
    this.pos = basePos + n;
    return out;
  }
  // ids.length columns (1..4). snapshot: save recurrent state after every
  // non-final column so restoreDN(k) can undo a rejected speculative suffix.
  async embedRunBatch(ids, basePos, snapshot = false) {
    if (!this.B) this._initBatch();
    const n = ids.length;
    this.pos = basePos;
    const sp = !snapshot ? 0 : typeof snapshot === "object"
      ? ((snapshot.total << 8) | (snapshot.base + 1)) : ((n << 8) | 1);
    for (let c = 0; c < n; c++) {
      this.device.queue.writeBuffer(this.frameBufsB[c], 0, new Uint32Array([basePos + c, basePos + c + 1, n, sp]));
      this.device.queue.writeBuffer(this.B.x.buf, c * this.B.x.stride, this._embedRowF32(ids[c]));
    }
    return this._runBatchAndRead(basePos, n);
  }
  async runHiddenBatch(xs, basePos, snapshot = false) {
    if (!this.B) this._initBatch();
    const { dim } = this.dims;
    const n = xs.length / dim;
    this.pos = basePos;
    const sp = !snapshot ? 0 : typeof snapshot === "object"
      ? ((snapshot.total << 8) | (snapshot.base + 1)) : ((n << 8) | 1);
    for (let c = 0; c < n; c++) {
      this.device.queue.writeBuffer(this.frameBufsB[c], 0, new Uint32Array([basePos + c, basePos + c + 1, n, sp]));
      this.device.queue.writeBuffer(this.B.x.buf, c * this.B.x.stride, xs.subarray(c * dim, (c + 1) * dim));
    }
    return this._runBatchAndRead(basePos, n);
  }
  restoreDN(k) { this._restoreDN(k); }
  setHidden(h) { this.device.queue.writeBuffer(this.x, 0, h); }   // final trunk hidden (chain host) for the draft head

  // final norm + LM head for n hidden states (n*dim floats, or null to use the
  // columns already in B.x) -> array of n logits vectors. Leaves the hiddens
  // in B.x so the draft head can read them by column.
  async headBatch(hs, n = hs ? hs.length / this.dims.dim : this.NC) {
    if (!this.B) this._initBatch();
    const { dim, vocab } = this.dims;
    if (hs) for (let c = 0; c < n; c++) this.device.queue.writeBuffer(this.B.x.buf, c * this.B.x.stride, hs.subarray(c * dim, (c + 1) * dim));
    this.device.queue.writeBuffer(this.frameBufsB[0], 0, new Uint32Array([this.pos, this.pos + 1, n, 0]));
    const enc = this.device.createCommandEncoder();
    const p = enc.beginComputePass();
    this._dMC(p, "rmsnorm_mc", this.bgFinalNormMC, 256, 256, n);
    this._dop(p, this.headB, n);
    p.end();
    for (let c = 0; c < n; c++) enc.copyBufferToBuffer(this.B.logits.buf, c * this.B.logits.stride, this.stageLogitsN, c * vocab * 4, vocab * 4);
    this.device.queue.submit([enc.finish()]);
    await this.stageLogitsN.mapAsync(GPUMapMode.READ, 0, n * vocab * 4);
    const all = new Float32Array(this.stageLogitsN.getMappedRange(0, n * vocab * 4)).slice();
    this.stageLogitsN.unmap();
    const out = [];
    for (let c = 0; c < n; c++) out.push(all.subarray(c * vocab, (c + 1) * vocab));
    return out;
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
      if (wantLogits === "argmax") this._d(p, "argmax", this.bgArgmax, 256, 256);
      p.end();
    }
    if (wantLogits === "argmax") enc.copyBufferToBuffer(this.argBuf, 0, this.stageArg, 0, 16);
    this.device.queue.submit([enc.finish()]);
    if (!wantLogits) return null;
    if (wantLogits === "argmax") {
      await this.stageArg.mapAsync(GPUMapMode.READ);
      const id = new Uint32Array(this.stageArg.getMappedRange())[0];
      this.stageArg.unmap();
      return id;
    }
    return await this._readback(this.logits, this.stageLogits, vocab);
  }

  // verify tokens[k] at positions pos+k (2..4 tokens): trunk (local batched
  // pass with DeltaNet snapshots, or a caller-supplied runTrunk for a device
  // chain) then one batched head pass -> logits per column.
  async verifyN(tokens, pos, runTrunk = null) {
    const n = tokens.length, { dim } = this.dims;
    let hs;
    if (runTrunk) hs = await runTrunk(tokens, pos);
    else if (n <= this.NC) hs = await this.embedRunBatch(tokens, pos, true);
    else {
      hs = new Float32Array(n * dim);
      for (let c0 = 0; c0 < n; c0 += this.NC) {
        const m = Math.min(this.NC, n - c0);
        hs.set(await this.embedRunBatch(tokens.slice(c0, c0 + m), pos + c0, { base: c0, total: n }), c0 * dim);
      }
    }
    const lgs = [];
    for (let c0 = 0; c0 < n; c0 += this.NC)
      lgs.push(...await this.headBatch(hs.subarray(c0 * dim, Math.min(n, c0 + this.NC) * dim), Math.min(this.NC, n - c0)));
    return { lgs, hs };
  }
  _restoreDN(k) {   // recurrent state as it was after verify column k
    const enc = this.device.createCommandEncoder();
    for (const L of this.layers) if (!L.isFull) {
      enc.copyBufferToBuffer(L.S_shadow, k * L.S.size, L.S, 0, L.S.size);
      enc.copyBufferToBuffer(L.conv_shadow, k * L.convState.size, L.convState, 0, L.convState.size);
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
  // runTrunk(tokens, pos) -> hidden states for all columns (chain mode);
  // onReject(k) tells the other devices to roll their recurrent state back
  // to what it was after column k.
  async specStep(tNext, sample, K = 3, { runTrunk = null, onReject = null } = {}) {
    const pos = this.pos, M2 = this.mtp, { dim } = this.dims;
    K = Math.max(1, Math.min(7, K));
    const drafts = [];
    for (let k = 0; k < K; k++) {
      // after the first call this.x holds the MTP block's own output hidden,
      // which is what chained drafting feeds back in
      drafts.push(await this.mtpRun(null, k === 0 ? tNext : drafts[k - 1], pos + k, "argmax"));
    }
    const { lgs, hs } = await this.verifyN([tNext, ...drafts], pos, runTrunk);
    const out = [];
    let a = 0;   // accepted drafts
    for (let k = 0; k <= K; k++) {
      const t = sample(lgs[k]);
      out.push(t);
      if (k < K && t === drafts[k]) a++; else break;
    }
    M2.stats.drafts += K; M2.stats.accepted += a;
    if (a < K) { this._restoreDN(a); if (onReject) await onReject(a); }
    // re-fill the draft cache for the accepted positions with exact trunk hiddens
    for (let j = 1; j <= a; j++) {
      this.setHidden(hs.subarray((j - 1) * dim, j * dim));
      await this.mtpRun(null, out[j - 1], pos + j, false);
    }
    this.setHidden(hs.subarray(a * dim, (a + 1) * dim));
    this.pos = pos + a + 1;
    return out;
  }

  async prefillTokens(ids) {
    if (!this.B) this._initBatch();
    let i = 0, sinceSync = 0;
    const NC = this.NC;
    while (ids.length - i >= NC) {
      const basePos = this.pos;
      for (let c = 0; c < NC; c++) {
        this.device.queue.writeBuffer(this.frameBufsB[c], 0, new Uint32Array([basePos + c, basePos + c + 1, NC, 0]));
        this.device.queue.writeBuffer(this.B.x.buf, c * this.B.x.stride, this._embedRowF32(ids[i + c]));
      }
      const enc = this.device.createCommandEncoder();
      for (let l = 0; l < this.layers.length; l++) this._encodeLayerBatch(enc, l, basePos, NC);
      enc.copyBufferToBuffer(this.B.x.buf, (NC - 1) * this.B.x.stride, this.x, 0, this.dims.dim * 4);
      this.device.queue.submit([enc.finish()]);
      if (this.mtp && this.mtpFill !== false)
        for (let c = 0; c < NC; c++) if (i + c + 1 < ids.length) await this.mtpRun(c, ids[i + c + 1], basePos + c + 1, false);
      this.pos += NC;
      i += NC;
      if (++sinceSync >= 4) { await this.device.queue.onSubmittedWorkDone(); sinceSync = 0; }
    }
    // step the tail down through the narrower twins (16 -> 8 -> 4) before
    // falling back to single tokens: at NC=16 a 15-token remainder would
    // otherwise cost 15 full single-token passes (~1.7 s on a 27B).
    for (const W of [8, 4].filter((w) => w < NC)) {
      while (ids.length - i >= W) {
        const basePos = this.pos;
        for (let c = 0; c < W; c++) {
          this.device.queue.writeBuffer(this.frameBufsB[c], 0, new Uint32Array([basePos + c, basePos + c + 1, W, 0]));
          this.device.queue.writeBuffer(this.B.x.buf, c * this.B.x.stride, this._embedRowF32(ids[i + c]));
        }
        const enc = this.device.createCommandEncoder();
        for (let l = 0; l < this.layers.length; l++) this._encodeLayerBatch(enc, l, basePos, W);
        enc.copyBufferToBuffer(this.B.x.buf, (W - 1) * this.B.x.stride, this.x, 0, this.dims.dim * 4);
        this.device.queue.submit([enc.finish()]);
        if (this.mtp && this.mtpFill !== false)
          for (let c = 0; c < W; c++) if (i + c + 1 < ids.length) await this.mtpRun(c, ids[i + c + 1], basePos + c + 1, false);
        this.pos += W; i += W;
        await this.device.queue.onSubmittedWorkDone();
      }
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
