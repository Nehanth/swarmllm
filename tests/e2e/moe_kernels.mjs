// engine/wgsl/moe.js on its own: random Q4_0 / Q8_0 experts in the engine's layout, random inputs,
// router -> gate/up -> down -> combine on the GPU vs a float64 JavaScript reference, and every
// column of a 4-column launch bit-identical to the same column launched alone.
//   NODE_PATH=... node tests/e2e/moe_kernels.mjs
import { loadPlaywright, chromiumPath, GPU_ARGS, serveRepo } from "./engine_synth.mjs";
const PORT = 18984;

async function pageMain() {
  const { moeWGSL } = await import("/engine/wgsl/moe.js");
  const { f16ToF32, f32ToF16 } = await import("/engine/gguf.js");
  const out = [], res = [];
  const check = (n, ok, d = "") => { res.push(ok); out.push(`${ok ? "PASS" : "FAIL"} ${n}${d ? "  " + d : ""}`); };
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const errs = []; device.addEventListener("uncapturederror", (e) => errs.push(e.error?.message));
  const HEAD = `struct Config { dim: u32, kvDim: u32, nH: u32, nKV: u32, headDim: u32, inter: u32, vocab: u32, maxSeq: u32, eps: f32, theta: f32, qDim: u32 };
struct Frame { pos: u32, seqLen: u32, nCols: u32, snap: u32 };
@group(0) @binding(0) var<uniform> cfg: Config;
@group(0) @binding(1) var<uniform> frame: Frame;
`;
  const mod = device.createShaderModule({ code: HEAD + moeWGSL() });
  const info = await mod.getCompilationInfo();
  const bad = info.messages.filter((m) => m.type === "error");
  if (bad.length) return { out: ["shader errors: " + bad.map((m) => m.lineNum + ": " + m.message).join(" | ")], ok: false };
  const U = GPUBufferUsage;
  const buf = (data, usage = U.STORAGE) => { const b = device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 16) * 16), usage: usage | U.COPY_DST | U.COPY_SRC }); device.queue.writeBuffer(b, 0, data); return b; };
  const empty = (bytes) => device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
  const uni = (a) => buf(new Uint32Array(a), U.UNIFORM);
  const g0 = device.createBindGroupLayout({ entries: [0, 1].map((b) => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } })) });
  const T = { u: "uniform", ro: "read-only-storage", rw: "storage" };
  const pipe = (name, spec) => {
    const l1 = device.createBindGroupLayout({ entries: spec.map((t, i) => ({ binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: T[t] } })) });
    return device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [g0, l1] }), compute: { module: mod, entryPoint: name } });
  };
  const P = {
    router: pipe("moe_router", ["ro", "rw", "rw", "u"]),
    gu_q4: pipe("moe_gu_q4", ["ro", "ro", "ro", "ro", "ro", "rw", "ro", "u"]), gu_q8: pipe("moe_gu_q8", ["ro", "ro", "ro", "ro", "ro", "rw", "ro", "u"]),
    dn_q4: pipe("moe_dn_q4", ["ro", "ro", "ro", "rw", "ro", "u"]), dn_q8: pipe("moe_dn_q8", ["ro", "ro", "ro", "rw", "ro", "u"]),
    combine: pipe("moe_combine", ["rw", "ro", "ro", "ro", "ro", "u"]),
  };
  const cfgB = uni(new Array(12).fill(0)), frameB = uni([0, 1, 1, 0]);
  const bg0 = (p) => device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: cfgB } }, { binding: 1, resource: { buffer: frameB } }] });
  const bg1 = (p, bufs) => device.createBindGroup({ layout: p.getBindGroupLayout(1), entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
  const read = async (b, n) => {
    const st = device.createBuffer({ size: Math.ceil(n * 4 / 16) * 16, usage: U.MAP_READ | U.COPY_DST });
    const e = device.createCommandEncoder(); e.copyBufferToBuffer(b, 0, st, 0, Math.ceil(n * 4 / 16) * 16); device.queue.submit([e.finish()]);
    await st.mapAsync(GPUMapMode.READ); const r = new Float32Array(st.getMappedRange().slice(0)); st.unmap(); return r.subarray(0, n);
  };
  let seed = 12345; const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
  // quantized matrix [rows][dIn] in engine layout; returns { qs, sc, deq(row, j) }
  const quant = (fmt, rows, dIn) => {
    const nb = dIn / 32, W = fmt === "q4" ? 4 : 8;
    const qs = new Uint32Array(rows * nb * W), sch = new Uint16Array(rows * nb + (rows * nb) % 2);
    for (let i = 0; i < qs.length; i++) qs[i] = (rnd() * 4294967296) >>> 0;
    for (let i = 0; i < rows * nb; i++) sch[i] = f32ToF16((rnd() - 0.5) * 0.05);
    const sc = new Uint32Array(sch.buffer);
    const deq = (r, j) => {
      const b = Math.floor(j / 32), k = j % 32, s = f16ToF32(sch[r * nb + b]);
      if (fmt === "q4") { const qt = (k % 16) >> 2, i = k % 4, w = qs[(r * nb + b) * 4 + qt]; const n = ((w >>> (8 * i + (k >= 16 ? 4 : 0))) & 15) - 8; return s * n; }
      const w = qs[(r * nb + b) * 8 + (k >> 2)]; let q = (w >>> (8 * (k % 4))) & 255; if (q > 127) q -= 256; return s * q;
    };
    return { qs, sc, deq };
  };
  const dim = 256, inter = 96, nExp = 16, K = 4, C = 4;
  for (const [gfmt, dfmt] of [["q4", "q8"], ["q8", "q4"]]) {
    const Wg = quant(gfmt, nExp * inter, dim), Wu = quant(gfmt, nExp * inter, dim), Wd = quant(dfmt, nExp * dim, inter);
    const x = Float32Array.from({ length: C * dim }, () => rnd() - 0.5);
    const logits = Float32Array.from({ length: C * nExp }, () => (rnd() - 0.5) * 4);
    const sh = Float32Array.from({ length: C * dim }, () => rnd() - 0.5);
    const sg = Float32Array.from({ length: C * 4 }, () => rnd() - 0.5);
    const x0 = Float32Array.from({ length: C * dim }, () => rnd() - 0.5);
    // GPU run for columns [c0, c0 + n)
    const run = async (c0, n) => {
      const bL = buf(logits.slice(c0 * nExp, (c0 + n) * nExp)), bSel = empty(n * K * 4), bW = empty(n * K * 4);
      const bX = buf(x.slice(c0 * dim, (c0 + n) * dim)), bH = empty(n * K * inter * 4), bY = empty(n * K * dim * 4);
      const bOut = buf(x0.slice(c0 * dim, (c0 + n) * dim)), bSh = buf(sh.slice(c0 * dim, (c0 + n) * dim)), bSg = buf(sg.slice(c0 * 4, (c0 + n) * 4));
      const enc = device.createCommandEncoder(), p = enc.beginComputePass();
      const go = (pp, bufs, x, y) => { p.setPipeline(pp); p.setBindGroup(0, bg0(pp)); p.setBindGroup(1, bg1(pp, bufs)); p.dispatchWorkgroups(x, y); };
      go(P.router, [bL, bSel, bW, uni([0, 0, K, nExp, nExp, 0, 1, 0])], n, 1);
      go(P["gu_" + gfmt], [buf(Wg.qs), buf(Wg.sc), buf(Wu.qs), buf(Wu.sc), bX, bH, bSel, uni([inter, dim, K, nExp, dim, inter, 0, 0])], Math.ceil(inter / 4), n * K);
      go(P["dn_" + dfmt], [buf(Wd.qs), buf(Wd.sc), bH, bY, bSel, uni([dim, inter, K, nExp, inter, dim, 0, 0])], Math.ceil(dim / 4), n * K);
      go(P.combine, [bOut, bY, bW, bSh, bSg, uni([dim, 0, K, dim, dim, dim, 1, 4])], Math.ceil(dim / 64), n);
      p.end(); device.queue.submit([enc.finish()]);
      return read(bOut, n * dim);
    };
    const all = await run(0, C);
    let same = true;
    for (let c = 0; c < C; c++) { const one = await run(c, 1); for (let i = 0; i < dim; i++) if (!Object.is(one[i], all[c * dim + i])) same = false; }
    check(`${gfmt} gate/up + ${dfmt} down: ${C} columns in one launch == one at a time (bit-identical)`, same);
    // reference in float64
    let maxErr = 0, maxRef = 0;
    for (let c = 0; c < C; c++) {
      const l = logits.subarray(c * nExp, (c + 1) * nExp), m = Math.max(...l);
      const pr = Array.from(l, (v) => Math.exp(v - m)); const z = pr.reduce((a, b) => a + b, 0);
      const order = pr.map((v, i) => [v / z, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, K);
      const tot = order.reduce((a, b) => a + b[0], 0);
      const o = new Float64Array(dim);
      for (const [pw, e] of order) {
        const h = new Float64Array(inter);
        for (let r = 0; r < inter; r++) {
          let g = 0, u = 0;
          for (let j = 0; j < dim; j++) { g += Wg.deq(e * inter + r, j) * x[c * dim + j]; u += Wu.deq(e * inter + r, j) * x[c * dim + j]; }
          h[r] = g / (1 + Math.exp(-g)) * u;
        }
        for (let r = 0; r < dim; r++) { let y = 0; for (let j = 0; j < inter; j++) y += Wd.deq(e * dim + r, j) * h[j]; o[r] += (pw / tot) * y; }
      }
      const s = 1 / (1 + Math.exp(-sg[c * 4]));
      for (let i = 0; i < dim; i++) {
        const ref = x0[c * dim + i] + o[i] + s * sh[c * dim + i];
        maxErr = Math.max(maxErr, Math.abs(ref - all[c * dim + i])); maxRef = Math.max(maxRef, Math.abs(ref));
      }
    }
    check(`${gfmt} gate/up + ${dfmt} down vs float64 reference`, maxErr / maxRef < 1e-4, `max err ${maxErr.toExponential(2)} (${(maxErr / maxRef).toExponential(1)} of max)`);
  }
  check("no GPU validation errors", !errs.length, errs.slice(0, 2).join(" | "));
  return { out, ok: res.every(Boolean) };
}
const srv = serveRepo(PORT, {});
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), args: GPU_ARGS });
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("[pageerror]", String(e).slice(0, 300)));
await page.goto(`http://127.0.0.1:${PORT}/favicon.svg`);
const r = await page.evaluate(pageMain);
for (const l of r.out) console.log(l);
console.log(r.ok ? "MOE KERNELS PASS" : "MOE KERNELS FAIL");
await browser.close(); srv.close();
process.exit(r.ok ? 0 : 1);
