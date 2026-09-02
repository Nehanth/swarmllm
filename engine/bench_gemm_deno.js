// Prototype: tiled Q4_0 GEMM for prefill (16 token columns) vs the batched
// GEMV kernels. Synthetic weights; correctness = agreement with coop_b.
import { WGSL, coopWGSL, probeUnpack } from "./engine.js";
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
const dOut = +(Deno.env.get("DOUT") || 17408), dIn = 5120, N = 16, nb = dIn / 32;
const TM = +(Deno.env.get("TM") || 32 * +(Deno.env.get("RT") || 2));          // rows per workgroup tile
const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const rnd = (n, f) => { const a = new Uint32Array(n); for (let i = 0; i < n; i++) a[i] = f(i); return a; };
const qsData = rnd(dOut * dIn / 8, () => (Math.random() * 2 ** 32) >>> 0);
const f16 = (v) => { const f = new Float32Array([v]); const u = new Uint32Array(f.buffer)[0]; const s = (u >> 16) & 0x8000, e = ((u >> 23) & 0xff) - 112, m = (u >> 13) & 0x3ff; return e <= 0 ? s : s | (e << 10) | m; };
const scData = rnd(Math.ceil(dOut * nb / 2), () => f16(0.01 + Math.random() * 0.02) | (f16(0.01 + Math.random() * 0.02) << 16));
const xData = new Float32Array(N * dIn); for (let i = 0; i < xData.length; i++) xData[i] = Math.random() * 2 - 1;
const mk = (data, usage = S) => { const b = device.createBuffer({ size: data.byteLength, usage }); device.queue.writeBuffer(b, 0, data); return b; };
const U = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
device.addEventListener("uncapturederror", (e) => console.log("GPU ERROR:", e.error.message.slice(0, 300)));
const qs = mk(qsData), sc = mk(scData), x = mk(xData);
const y = device.createBuffer({ size: N * dOut * 4, usage: S }), y2 = device.createBuffer({ size: N * dOut * 4, usage: S });
const shape = mk(new Uint32Array([dOut, dIn, dIn / 4, dOut]), U);   // BShape { dOut, dIn, xs4 (vec4 stride), ys }
const cfgB = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM }), frameB = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM });
const C = GPUShaderStage.COMPUTE;
const l0 = device.createBindGroupLayout({ entries: [0, 1].map((b) => ({ binding: b, visibility: C, buffer: { type: "uniform" } })) });
const l1 = device.createBindGroupLayout({ entries: ["read-only-storage", "read-only-storage", "read-only-storage", "storage", "uniform"].map((t, i) => ({ binding: i, visibility: C, buffer: { type: t } })) });
const bg0 = device.createBindGroup({ layout: l0, entries: [{ binding: 0, resource: { buffer: cfgB } }, { binding: 1, resource: { buffer: frameB } }] });
const bg1 = (yb) => device.createBindGroup({ layout: l1, entries: [qs, sc, x, yb, shape].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
const unpack = await probeUnpack(device);

// ---- the GEMM: WG = 128 threads, tile = TM rows x 16 cols, k in blocks of 32.
// Weights for the tile are dequantized ONCE into shared memory (each thread
// unpacks one u32 word = 8 nibbles); the 16 x-columns for the k-block are
// staged as vec4s; each thread owns 2 rows x 2 cols (4 accumulators).
const RT = +(Deno.env.get("RT") || 2), T = 128;    // RT rows x 4 cols per thread; TM = T/4*RT rows per WG
if (TM !== T / 4 * RT) throw new Error(`TM must be ${T / 4 * RT} for RT=${RT}`);
const R = Array.from({ length: RT }, (_, i) => i);
const WPT = TM * 8 / T;                             // weight words per thread per block pair (8)
const loads = Array.from({ length: WPT }, (_, j) => j);
const gemm = `
struct BShape { dOut: u32, dIn: u32, xs4: u32, ys: u32 };
@group(0) @binding(0) var<uniform> cfg0: vec4<u32>;
@group(0) @binding(1) var<uniform> frame0: vec4<u32>;
@group(1) @binding(0) var<storage, read> g_qs: array<u32>;
@group(1) @binding(1) var<storage, read> g_sc: array<u32>;
@group(1) @binding(2) var<storage, read> g_x: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> g_y: array<f32>;
@group(1) @binding(4) var<uniform> g_shape: BShape;
var<workgroup> Ws: array<vec4<f32>, ${TM * 17}>;   // [TM rows][17 = 16 k-quads + pad]
var<workgroup> Xs: array<vec4<f32>, ${16 * 16}>;   // [16 k-quads][16 cols]
fn g_sc_at(i: u32) -> f32 { return unpack2x16float(g_sc[i >> 1u])[i & 1u]; }
fn nib_lo(w: u32) -> vec4<f32> { return vec4<f32>(unpack4xU8(w & 0x0F0F0F0Fu)) - vec4<f32>(8.0); }
fn nib_hi(w: u32) -> vec4<f32> { return vec4<f32>(unpack4xU8((w >> 4u) & 0x0F0F0F0Fu)) - vec4<f32>(8.0); }
@compute @workgroup_size(${T})
fn gemm16(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let dIn = g_shape.dIn; let dOut = g_shape.dOut; let xs4 = g_shape.xs4; let ys = g_shape.ys;
  let nb = dIn / 32u; let rowWords = dIn / 8u; let nbp = nb / 2u;
  let row0 = (wg.y * 32768u + wg.x) * ${TM}u;
  let rq = t >> 2u; let cq = t & 3u;                 // row-group, col-quad (0..3)
  let rb = row0 + rq * ${RT}u; let c0 = cq * 4u;
  ${R.map((i) => `var a${i} = vec4<f32>(0.0);`).join(" ")}
  ${loads.map((j) => `let lr${j} = (t + ${j * T}u) >> 3u; let lw${j} = (t + ${j * T}u) & 7u; let ok${j} = row0 + lr${j} < dOut; let wb${j} = (row0 + lr${j}) * rowWords + lw${j};`).join("\n  ")}
  let xc0 = t & 15u; let xq0 = t >> 4u; let xq1 = xq0 + 8u;
  let xb0 = xc0 * xs4 + xq0; let xb1 = xc0 * xs4 + xq1;
  ${loads.map((j) => `var w${j}r = select(0u, g_qs[wb${j}], ok${j});`).join(" ")}
  var x0r = g_x[xb0]; var x1r = g_x[xb1];
  for (var bp: u32 = 0u; bp < nbp; bp++) {
    workgroupBarrier();
    ${loads.map((j) => `Ws[lr${j} * 17u + (lw${j} >> 2u) * 8u + (lw${j} & 3u)] = nib_lo(w${j}r); Ws[lr${j} * 17u + (lw${j} >> 2u) * 8u + 4u + (lw${j} & 3u)] = nib_hi(w${j}r);`).join("\n    ")}
    Xs[xq0 * 16u + xc0] = x0r; Xs[xq1 * 16u + xc0] = x1r;
    workgroupBarrier();
    if (bp + 1u < nbp) {
      let o = (bp + 1u) * 8u;
      ${loads.map((j) => `w${j}r = select(0u, g_qs[wb${j} + o], ok${j});`).join(" ")}
      x0r = g_x[xb0 + (bp + 1u) * 16u]; x1r = g_x[xb1 + (bp + 1u) * 16u];
    }
    for (var half: u32 = 0u; half < 2u; half++) {
      ${R.map((i) => `var p${i} = vec4<f32>(0.0);`).join(" ")}
      let q0 = half * 8u;
      for (var q: u32 = q0; q < q0 + 8u; q++) {
        let xa = Xs[q * 16u + c0]; let xb = Xs[q * 16u + c0 + 1u]; let xc2 = Xs[q * 16u + c0 + 2u]; let xd = Xs[q * 16u + c0 + 3u];
        ${R.map((i) => `let w${i} = Ws[(rq * ${RT}u + ${i}u) * 17u + q]; p${i} += vec4<f32>(dot(w${i}, xa), dot(w${i}, xb), dot(w${i}, xc2), dot(w${i}, xd));`).join("\n        ")}
      }
      let bi = 2u * bp + half;
      ${R.map((i) => `a${i} += g_sc_at(min(rb + ${i}u, dOut - 1u) * nb + bi) * p${i};`).join(" ")}
    }
  }
  ${R.map((i) => `if (rb + ${i}u < dOut) { g_y[c0 * ys + rb + ${i}u] = a${i}.x; g_y[(c0 + 1u) * ys + rb + ${i}u] = a${i}.y; g_y[(c0 + 2u) * ys + rb + ${i}u] = a${i}.z; g_y[(c0 + 3u) * ys + rb + ${i}u] = a${i}.w; }`).join("\n  ")}
}`;
const mkPipe = async (code, entry) => device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [l0, l1] }), compute: { module: device.createShaderModule({ code }), entryPoint: entry } });
const pG = await mkPipe(gemm, "gemm16");
const p4 = await mkPipe(WGSL + coopWGSL(256, 4, 64, 4, 4, unpack), "matvec_q4_coop_b");
const p8 = await mkPipe(WGSL + coopWGSL(256, 4, 64, 8, 2, unpack), "matvec_q4_coop_b");
const time = async (fn, reps = 30) => { const tw = performance.now(); while (performance.now() - tw < 300) { fn(10); await device.queue.onSubmittedWorkDone(); } const t0 = performance.now(); fn(reps); await device.queue.onSubmittedWorkDone(); return (performance.now() - t0) / reps; };
const wgsG = Math.ceil(dOut / TM);
const runG = (n, yb) => { const e = device.createCommandEncoder(); const p = e.beginComputePass(); p.setPipeline(pG); p.setBindGroup(0, bg0); p.setBindGroup(1, bg1(yb)); for (let i = 0; i < n; i++) { if (wgsG > 32768) p.dispatchWorkgroups(32768, Math.ceil(wgsG / 32768)); else p.dispatchWorkgroups(wgsG); } p.end(); device.queue.submit([e.finish()]); };
// batched GEMV over 16 columns = 4 dispatches of 4 cols (x/y column offsets via separate shapes)
const shapes4 = [0, 1, 2, 3].map((k) => mk(new Uint32Array([dOut, dIn, dIn / 4, dOut]), U));
const bgs4 = (yb, rows) => [0, 1, 2, 3].map((k) => device.createBindGroup({ layout: l1, entries: [
  { binding: 0, resource: { buffer: qs } }, { binding: 1, resource: { buffer: sc } },
  { binding: 2, resource: { buffer: x, offset: k * 4 * dIn * 4, size: 4 * dIn * 4 } },
  { binding: 3, resource: { buffer: yb, offset: k * 4 * dOut * 4, size: 4 * dOut * 4 } }, { binding: 4, resource: { buffer: shapes4[k] } }] }));
const run4 = (n, yb) => { const b = bgs4(yb); const e = device.createCommandEncoder(); const p = e.beginComputePass(); p.setPipeline(p4); p.setBindGroup(0, bg0); for (let i = 0; i < n; i++) for (let k = 0; k < 4; k++) { p.setBindGroup(1, b[k]); p.dispatchWorkgroups(Math.ceil(dOut / 4)); } p.end(); device.queue.submit([e.finish()]); };
const bgs8 = (yb) => [0, 1].map((k) => device.createBindGroup({ layout: l1, entries: [
  { binding: 0, resource: { buffer: qs } }, { binding: 1, resource: { buffer: sc } },
  { binding: 2, resource: { buffer: x, offset: k * 8 * dIn * 4, size: 8 * dIn * 4 } },
  { binding: 3, resource: { buffer: yb, offset: k * 8 * dOut * 4, size: 8 * dOut * 4 } }, { binding: 4, resource: { buffer: shapes4[k] } }] }));
const run8 = (n, yb) => { const b = bgs8(yb); const e = device.createCommandEncoder(); const p = e.beginComputePass(); p.setPipeline(p8); p.setBindGroup(0, bg0); for (let i = 0; i < n; i++) for (let k = 0; k < 2; k++) { p.setBindGroup(1, b[k]); p.dispatchWorkgroups(Math.ceil(dOut / 2)); } p.end(); device.queue.submit([e.finish()]); };
// correctness: gemm vs coop_b (4-col)
runG(1, y); run4(1, y2); await device.queue.onSubmittedWorkDone();
const rd = async (b) => { const st = device.createBuffer({ size: N * dOut * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }); const e = device.createCommandEncoder(); e.copyBufferToBuffer(b, 0, st, 0, N * dOut * 4); device.queue.submit([e.finish()]); await st.mapAsync(GPUMapMode.READ); const out = Float32Array.from(new Float32Array(st.getMappedRange())); st.unmap(); return out; };
const A = await rd(y), B = await rd(y2);
let num = 0, den = 0; for (let i = 0; i < A.length; i++) { num += (A[i] - B[i]) ** 2; den += B[i] ** 2; }
console.log(`correctness vs coop_b: relDiff=${Math.sqrt(num / den).toExponential(2)}  sample gemm=${A[12345].toFixed(4)} ref=${B[12345].toFixed(4)}`);
const tG = await time((n) => runG(n, y)), t4 = await time((n) => run4(n, y2)), t8 = await time((n) => run8(n, y2));
const gb = dOut * dIn * 0.5625 / 1e9;
console.log(`${dOut}x${dIn} Q4, 16 columns:  gemm16(TM=${TM}) ${tG.toFixed(3)} ms (${(gb / tG * 1e3).toFixed(0)} GB/s weights)  |  coop_b 4x4cols ${t4.toFixed(3)} ms  |  coop_b 2x8cols ${t8.toFixed(3)} ms  => gemm ${(t4 / tG).toFixed(2)}x vs 4-col path`);
