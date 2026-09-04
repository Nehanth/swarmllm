// Row-stationary prefill GEMM vs the batched GEMV, inside the real shader module.
// Gates: compiles clean alongside every other kernel, and matches
// matvec_q4_coop_b (and _acc) within 5e-6 on every shape the engine runs.
import { WGSL, coopWGSL, probeUnpack } from "../engine/engine.js";
import { gemmWGSL, GEMM_S, GEMM_TILE } from "../engine/wgsl/gemm.js";
import { f32ToF16 } from "../engine/gguf.js";

const N = 16;
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: {
  maxBufferSize: adapter.limits.maxBufferSize,
  maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
const unpack = await probeUnpack(device);
const SH = ["17408x5120", "5120x17408", "5120x6144", "10240x5120", "6144x5120", "1024x5120"];
const dIns = [...new Set(SH.map((s) => +s.split("x")[1]))];
const splits = [...new Set(SH.map((s) => GEMM_S[s]))];

const code = WGSL + coopWGSL(256, 4, 64, N, 1, unpack) + gemmWGSL({ N, splits, dIns, UNPACK: unpack });
device.pushErrorScope("validation");
const mod = device.createShaderModule({ code });
const info = await mod.getCompilationInfo();
const errs = info.messages.filter((m) => m.type === "error");
const scopeErr = await device.popErrorScope();
for (const e of errs.slice(0, 3)) console.log(`WGSL L${e.lineNum}: ${e.message.slice(0, 200)}`);
if (errs.length || scopeErr) { console.log("GEMM COMPILE FAIL", scopeErr?.message?.slice(0, 200) ?? ""); Deno.exit(1); }
console.log(`module compiles: ${(code.length / 1024).toFixed(0)} KB, ${(code.match(/@compute/g) || []).length} entry points`);

const C = GPUShaderStage.COMPUTE, S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const l0 = device.createBindGroupLayout({ entries: [0, 1].map((b) => ({ binding: b, visibility: C, buffer: { type: "uniform" } })) });
const l1 = device.createBindGroupLayout({ entries: ["read-only-storage", "read-only-storage", "read-only-storage", "storage", "uniform"]
  .map((t, i) => ({ binding: i, visibility: C, buffer: { type: t } })) });
const layout = device.createPipelineLayout({ bindGroupLayouts: [l0, l1] });
const pipe = async (entryPoint) => device.createComputePipelineAsync({ layout, compute: { module: mod, entryPoint } });
const buf = (data, usage = S) => { const b = device.createBuffer({ size: data.byteLength, usage }); device.queue.writeBuffer(b, 0, data); return b; };
const U = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
const cfgB = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM }), frameB = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM });
const bg0 = device.createBindGroup({ layout: l0, entries: [{ binding: 0, resource: { buffer: cfgB } }, { binding: 1, resource: { buffer: frameB } }] });
const bg1 = (...bs) => device.createBindGroup({ layout: l1, entries: bs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
const read = async (b, n) => { const st = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e = device.createCommandEncoder(); e.copyBufferToBuffer(b, 0, st, 0, n * 4); device.queue.submit([e.finish()]);
  await st.mapAsync(GPUMapMode.READ); const out = Float32Array.from(new Float32Array(st.getMappedRange())); st.unmap(); st.destroy(); return out; };
const run = (p, bg, wgs) => { const e = device.createCommandEncoder(); const ps = e.beginComputePass();
  ps.setPipeline(p); ps.setBindGroup(0, bg0); ps.setBindGroup(1, bg); 
  if (wgs > 32768) ps.dispatchWorkgroups(32768, Math.ceil(wgs / 32768)); else ps.dispatchWorkgroups(wgs);
  ps.end(); device.queue.submit([e.finish()]); };

const align = (n) => Math.ceil(n * 4 / 256) * 256 / 4;   // engine's 256-byte column alignment, in floats
const pXpose = await pipe("gemm_xpose");
let worst = 0, fails = 0;
for (const shape of SH) {
  const [dOut, dIn] = shape.split("x").map(Number);
  const Sp = GEMM_S[shape], nb = dIn / 32;
  const xs = align(dIn), ys = align(dOut);
  const qs = buf(Uint32Array.from({ length: dOut * dIn / 8 }, () => (Math.random() * 2 ** 32) >>> 0));
  const sc = buf(Uint32Array.from({ length: Math.ceil(dOut * nb / 2) },
    () => f32ToF16(0.005 + Math.random() * 0.03) | (f32ToF16(0.005 + Math.random() * 0.03) << 16)));
  const xd = new Float32Array(N * xs); for (let c = 0; c < N; c++) for (let k = 0; k < dIn; k++) xd[c * xs + k] = Math.random() * 2 - 1;
  const x = buf(xd);
  const shp = buf(new Uint32Array([dOut, dIn, xs / 4, ys]), U);
  const xT = device.createBuffer({ size: dIn * N * 4, usage: S });
  const part = device.createBuffer({ size: Sp * N * dOut * 4, usage: S });
  const yRef = device.createBuffer({ size: N * ys * 4, usage: S });
  const yG = device.createBuffer({ size: N * ys * 4, usage: S });

  run(pXpose, bg1(x, sc, sc, xT, shp), Math.ceil(dIn * N / 64));
  const pG = await pipe(`gemm_q4_${dIn}_s${Sp}`);
  run(pG, bg1(qs, sc, xT, part, shp), Math.ceil(dOut / GEMM_TILE) * Sp);
  const pR = await pipe(`gemm_red_s${Sp}`);
  run(pR, bg1(part, sc, sc, yG, shp), Math.ceil(N * dOut / 64));
  const pRef = await pipe("matvec_q4_coop_b");
  run(pRef, bg1(qs, sc, x, yRef, shp), Math.ceil(dOut / 1));

  // _acc: pre-fill both outputs identically, then accumulate into them
  const seed = new Float32Array(N * ys); for (let i = 0; i < seed.length; i++) seed[i] = Math.random() - 0.5;
  const yA = buf(seed), yRA = buf(seed);
  run(await pipe(`gemm_red_s${Sp}_acc`), bg1(part, sc, sc, yA, shp), Math.ceil(N * dOut / 64));
  run(await pipe("matvec_q4_coop_b_acc"), bg1(qs, sc, x, yRA, shp), Math.ceil(dOut / 1));

  const [a, b, aa, ba] = await Promise.all([read(yG, N * ys), read(yRef, N * ys), read(yA, N * ys), read(yRA, N * ys)]);
  const rel = (p, q) => { let num = 0, den = 0;
    for (let c = 0; c < N; c++) for (let r = 0; r < dOut; r++) { const i = c * ys + r; num += (p[i] - q[i]) ** 2; den += q[i] ** 2; }
    return Math.sqrt(num / den); };
  const r1 = rel(a, b), r2 = rel(aa, ba);
  worst = Math.max(worst, r1, r2);
  const ok = r1 < 5e-6 && r2 < 5e-6;
  if (!ok) fails++;
  console.log(`${shape.padEnd(12)} S=${String(Sp).padEnd(2)} relDiff ${r1.toExponential(2)}  acc ${r2.toExponential(2)}  ${ok ? "ok" : "FAIL"}`);
  for (const bb of [qs, sc, x, xT, part, yRef, yG, yA, yRA]) bb.destroy();
}
console.log(fails ? `GEMM FAIL (${fails} shapes)` : `GEMM PASS ✓ (worst relDiff ${worst.toExponential(2)}, gate 5e-6)`);
if (fails) Deno.exit(1);
