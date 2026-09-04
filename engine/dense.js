// DenseEngine: WebGPU inference for dense Llama-architecture models (Qwen3, SmolLM), layer-shardable.
import { weightsFromSafetensors } from "./safetensors.js";
import { WGSL } from "./wgsl/base.js";
import { probeUnpack, coopWGSL } from "./wgsl/coop.js";
import { f16ToF32 } from "./gguf.js";

export class DenseEngine {
  // opts: { device, cfg, tensors?|weights?, layerRange, hasEmbed, hasHead, maxSeq }
  reset() { this.pos = 0; }   // fresh context; the KV cache is overwritten from position 0

  static async create(opts) {
    const e = new DenseEngine();
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

    const mod = device.createShaderModule({ code: WGSL + coopWGSL(coopWG, coopRows, 64, 4, 4, await probeUnpack(device)) });
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
      rope: ["rw", "u"], attn_scores: ["ro", "ro", "rw"], attn_softmax: ["rw"],
      attn_out: ["ro", "ro", "rw"], silu_mul: ["rw", "ro"], add_res: ["rw", "ro"],
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
    // fused gate/up(+SiLU) op; null when kinds differ (fallback: unfused path)
    const guOp = (wg2, wu2, x, y, dOut, dIn, xB, yB) => {
      if (!coop || !wg2 || !wu2 || wg2.kind !== wu2.kind) return null;
      const base = wg2.kind === "q8" ? "matvec_q8_gu" : wg2.kind === "q4" ? "matvec_q4_gu" : "matvec_gu";
      const pipe = xB ? base + "_b" : base;
      const shp = xB ? this._shapeB(dOut, dIn, xB.stride / 16, yB.stride / 4) : this._shapeB(dOut, dIn, 0, 0);
      const bufs = wg2.kind === "f32" ? [wg2.buf, wu2.buf, x, y, shp] : [wg2.qs, wg2.sc, wu2.qs, wu2.sc, x, y, shp];
      return { pipe, wgs: Math.ceil(dOut / this.coopRows), bg: this._bg(this.pipes[pipe], 1, bufs) };
    };
    this._guOp = guOp;
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
      gu: guOp(L2.wgate, L2.wup, this.xn, this.g, inter, dim),
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
    if (op.wgs > 32768) pass.dispatchWorkgroups(32768, Math.ceil(op.wgs / 32768)); else pass.dispatchWorkgroups(op.wgs);   // > 65535 per dimension is silently dropped
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
      if (BG.gu) this._dispatchOp(pass, BG.gu);
      else {
        this._dispatchOp(pass, BG.gate);
        this._dispatchOp(pass, BG.up);
        this._dispatch(pass, "silu_mul", this.bgSilu, inter);
      }
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
    this.stageXB = dev.createBuffer({ size: 4 * dim * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
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
        gu: this._guOp(L.wgate, L.wup, B.xn.buf, B.g.buf, inter, dim, B.xn, B.g),
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
      if (LB.gu) this._dispatchOp(pass, LB.gu);
      else {
        for (const op of LB.gateUp) this._dispatchOp(pass, op);
        for (let c = 0; c < 4; c++) this._dCol(pass, "silu_mul", c, LB.cols[c].silu, inter);
      }
      this._dispatchOp(pass, LB.down);
      for (let c = 0; c < 4; c++) this._dCol(pass, "add_res", c, LB.cols[c].addTmp, dim);
      pass.end();
    }
  }

  _stageEmbedBatchCol(enc, id, c) {
    const { dim } = this.dims;
    if (this.embedGPU) enc.copyBufferToBuffer(this.embedGPU, id * dim * 4, this.B.x.buf, c * this.B.x.stride, dim * 4);
    else this.device.queue.writeBuffer(this.B.x.buf, c * this.B.x.stride, this._embedRowF32(id));
  }
  async _runBatchAndRead(basePos) {
    const { dim } = this.dims;
    const enc = this.device.createCommandEncoder();
    if (this._pendingEmbeds) { for (const [id, c] of this._pendingEmbeds) this._stageEmbedBatchCol(enc, id, c); this._pendingEmbeds = null; }
    for (let l = 0; l < this.layers.length; l++) this._encodeLayerBatch(enc, l, basePos);
    for (let c = 0; c < 4; c++) enc.copyBufferToBuffer(this.B.x.buf, c * this.B.x.stride, this.stageXB, c * dim * 4, dim * 4);
    this.device.queue.submit([enc.finish()]);
    await this.stageXB.mapAsync(GPUMapMode.READ);
    const out = Float32Array.from(new Float32Array(this.stageXB.getMappedRange(), 0, 4 * dim));
    this.stageXB.unmap();
    this.pos = basePos + 4;
    return out;
  }
  // host, split mode: 4 prompt tokens -> 4 hiddens for the next peer
  async embedRunBatch(ids, basePos) {
    if (!this.B) this._initBatch();
    this.pos = basePos;
    this._pendingEmbeds = ids.map((id, c) => [id, c]);
    for (let c = 0; c < 4; c++)
      this.device.queue.writeBuffer(this.frameBufsB[c], 0, new Uint32Array([basePos + c, basePos + c + 1]));
    return this._runBatchAndRead(basePos);
  }
  // worker, split mode: 4 hiddens in, my layers, 4 hiddens out
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

// ---- device autotune: pick the coop kernel shape this GPU actually likes ----
// Times the quantized GEMV (the hot kernel) at a few candidate workgroup
// shapes on synthetic buffers sized like a real layer. Wall-clock around
// onSubmittedWorkDone; never timestamp-query (enabling it alone has measured
// multi-x slowdowns). ~1s total at model load.
