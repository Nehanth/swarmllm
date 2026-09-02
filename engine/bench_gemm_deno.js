// Prototype: tiled Q4_0 GEMM for prefill (16 token columns) vs the batched
// GEMV kernels. Synthetic weights; correctness = agreement with coop_b.
import { WGSL, coopWGSL, probeUnpack } from "./engine.js";
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
const dOut = +(Deno.env.get("DOUT") || 17408), dIn = 5120, N = 16, nb = dIn / 32;
const TM = +(Deno.env.get("TM") || 64);          // rows per workgroup tile
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
const RP = TM / 2;                                  // row-pairs; 4 col-quads -> threads = RP * 4
const T = RP * 4;
const gemm = `
struct BShape { dOut: u32, dIn: u32, xs4: u32, ys: u32 };
@group(0) @binding(0) var<uniform> cfg0: vec4<u32>;
@group(0) @binding(1) var<uniform> frame0: vec4<u32>;
@group(1) @binding(0) var<storage, read> g_qs: array<u32>;
@group(1) @binding(1) var<storage, read> g_sc: array<u32>;
@group(1) @binding(2) var<storage, read> g_x: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> g_y: array<f32>;
@group(1) @binding(4) var<uniform> g_shape: BShape;
var<workgroup> Ws: array<vec4<f32>, ${TM * 8}>;   // [TM rows][8 k-quads]
var<workgroup> Xs: array<vec4<f32>, ${8 * 16}>;   // [8 k-quads][16 cols]
fn g_sc_at(i: u32) -> f32 { return unpack2x16float(g_sc[i >> 1u])[i & 1u]; }
@compute @workgroup_size(${T})
fn gemm16(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let dIn = g_shape.dIn; let dOut = g_shape.dOut; let xs4 = g_shape.xs4; let ys = g_shape.ys;
  let nb = dIn / 32u; let rowWords = dIn / 8u;
  let row0 = (wg.y * 32768u + wg.x) * ${TM}u;
  let rp = t >> 2u; let cq = t & 3u;                 // 2 rows x 4 cols per thread
  let r0 = row0 + rp * 2u; let r1 = r0 + 1u;
  let c0 = cq * 4u;
  var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0);
  for (var b: u32 = 0u; b < nb; b++) {
    workgroupBarrier();
    // weights: TM*4 words per block, ${Math.ceil(TM * 4 / T)} per thread; word (row lr, quad lw)
    for (var i: u32 = t; i < ${TM * 4}u; i += ${T}u) {
      let lr = i >> 2u; let lw = i & 3u;
      let row = row0 + lr;
      let word = select(0u, g_qs[row * rowWords + b * 4u + lw], row < dOut);
      Ws[lr * 8u + lw] = vec4<f32>(unpack4xU8(word & 0x0F0F0F0Fu)) - vec4<f32>(8.0);          // k = lw*4 .. +3
      Ws[lr * 8u + 4u + lw] = vec4<f32>(unpack4xU8((word >> 4u) & 0x0F0F0F0Fu)) - vec4<f32>(8.0);   // k = 16 + lw*4 ..
    }
    for (var i: u32 = t; i < 128u; i += ${T}u) { let xc = i & 15u; let xq = i >> 4u; Xs[xq * 16u + xc] = g_x[xc * xs4 + b * 8u + xq]; }
    workgroupBarrier();
    var p0 = vec4<f32>(0.0); var p1 = vec4<f32>(0.0);
    for (var q: u32 = 0u; q < 8u; q++) {
      let w0 = Ws[rp * 16u + q]; let w1 = Ws[rp * 16u + 8u + q];
      let xa = Xs[q * 16u + c0]; let xb = Xs[q * 16u + c0 + 1u]; let xc2 = Xs[q * 16u + c0 + 2u]; let xd = Xs[q * 16u + c0 + 3u];
      p0 += vec4<f32>(dot(w0, xa), dot(w0, xb), dot(w0, xc2), dot(w0, xd));
      p1 += vec4<f32>(dot(w1, xa), dot(w1, xb), dot(w1, xc2), dot(w1, xd));
    }
    let s0 = g_sc_at(min(r0, dOut - 1u) * nb + b); let s1 = g_sc_at(min(r1, dOut - 1u) * nb + b);
    a0 += s0 * p0; a1 += s1 * p1;
  }
  if (r0 < dOut) { g_y[c0 * ys + r0] = a0.x; g_y[(c0 + 1u) * ys + r0] = a0.y; g_y[(c0 + 2u) * ys + r0] = a0.z; g_y[(c0 + 3u) * ys + r0] = a0.w; }
  if (r1 < dOut) { g_y[c0 * ys + r1] = a1.x; g_y[(c0 + 1u) * ys + r1] = a1.y; g_y[(c0 + 2u) * ys + r1] = a1.z; g_y[(c0 + 3u) * ys + r1] = a1.w; }
}`;
const mkPipe = async (code, entry) => device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [l0, l1] }), compute: { module: device.createShaderModule({ code }), entryPoint: entry } });
const pG = await mkPipe(gemm, "gemm16");
const p4 = await mkPipe(WGSL + coopWGSL(256, 4, 64, 4, 4, unpack), "matvec_q4_coop_b");
const p8 = await mkPipe(WGSL + coopWGSL(256, 4, 64, 8, 2, unpack), "matvec_q4_coop_b");
const time = async (fn, reps = 30) => { fn(3); await device.queue.onSubmittedWorkDone(); const t0 = performance.now(); fn(reps); await device.queue.onSubmittedWorkDone(); return (performance.now() - t0) / reps; };
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
