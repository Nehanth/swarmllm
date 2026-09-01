// Single-column q4 matvec variants: can wider (16-byte) weight loads per thread
// push closer to peak bandwidth than the current 4-byte-per-thread coop kernel?
import { WGSL, coopWGSL } from "./engine.js";
import { f32ToF16 } from "./gguf.js";
const dIn = +(Deno.env.get("DIN") || 5120), dOut = +(Deno.env.get("DOUT") || 17408);
const WG = +(Deno.env.get("WG") || 256), ROWS = 4, nb = dIn / 32;
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
const rnd = (() => { let s = 4242; return () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296; })();
const qsArr = new Uint32Array(dOut * dIn / 8); for (let i = 0; i < qsArr.length; i++) qsArr[i] = (rnd() * 4294967296) >>> 0;
const scArr = new Uint32Array(Math.ceil(dOut * nb / 2)); const sc16 = new Uint16Array(scArr.buffer); for (let i = 0; i < sc16.length; i++) sc16[i] = f32ToF16((rnd() - 0.5) * 0.02);
const xArr = new Float32Array(dIn); for (let i = 0; i < dIn; i++) xArr[i] = rnd() - 0.5;
const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const mk = (arr) => { const b = device.createBuffer({ size: arr.byteLength, usage: S }); device.queue.writeBuffer(b, 0, arr); return b; };
const qs = mk(qsArr), sc = mk(scArr), x = mk(xArr), y = device.createBuffer({ size: dOut * 4, usage: S });
const shape = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(shape, 0, new Uint32Array([dOut, dIn, 0, 0]));
const cfgB = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM }), frameB = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM });
const stage = device.createBuffer({ size: dOut * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
const dec = (w, v) => `let ${v}lo = vec4<f32>(f32(${w} & 0xFu), f32((${w} >> 8u) & 0xFu), f32((${w} >> 16u) & 0xFu), f32((${w} >> 24u) & 0xFu)) - vec4<f32>(8.0);
      let ${v}hi = vec4<f32>(f32((${w} >> 4u) & 0xFu), f32((${w} >> 12u) & 0xFu), f32((${w} >> 20u) & 0xFu), f32((${w} >> 28u) & 0xFu)) - vec4<f32>(8.0);`;
// variant W: each thread loads a whole 16-byte q4 block (vec4<u32>) per row; WG threads = WG block lanes
const EXTRA = (wg, rows) => `
@group(1) @binding(0) var<storage, read> q4_qs4: array<vec4<u32>>;
var<workgroup> w_part_${wg}_${rows}: array<f32, ${rows * wg}>;
@compute @workgroup_size(${wg})
fn xs_w${wg}_${rows}(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let dIn = q4_shape.dIn; let dOut = q4_shape.dOut;
  let nb = dIn / 32u; let rowBlocks = nb;   // one vec4<u32> per block
  let row0 = wgid.x * ${rows}u;
  let full = row0 + ${rows - 1}u < dOut;
  ${Array.from({ length: rows }, (_, r) => `var a${r} = 0.0;`).join(" ")}
  for (var b: u32 = t; b < nb; b += ${wg}u) {
    let x0 = q4_x4[b * 8u]; let x1 = q4_x4[b * 8u + 1u]; let x2 = q4_x4[b * 8u + 2u]; let x3 = q4_x4[b * 8u + 3u];
    let x4 = q4_x4[b * 8u + 4u]; let x5 = q4_x4[b * 8u + 5u]; let x6 = q4_x4[b * 8u + 6u]; let x7 = q4_x4[b * 8u + 7u];
    ${Array.from({ length: rows }, (_, r) => `if (full || row0 + ${r}u < dOut) {
      let w = q4_qs4[(row0 + ${r}u) * rowBlocks + b];
      ${dec("w.x", "p")} ${dec("w.y", "q")} ${dec("w.z", "r")} ${dec("w.w", "s")}
      a${r} += q4s((row0 + ${r}u) * nb + b) * (dot(plo, x0) + dot(qlo, x1) + dot(rlo, x2) + dot(slo, x3) + dot(phi, x4) + dot(qhi, x5) + dot(rhi, x6) + dot(shi, x7));
    }`).join("\n    ")}
  }
  ${Array.from({ length: rows }, (_, r) => `w_part_${wg}_${rows}[${r * wg}u + t] = a${r};`).join(" ")}
  workgroupBarrier();
  var stride: u32 = ${wg / 2}u;
  while (stride > 0u) {
    if (t < stride) { ${Array.from({ length: rows }, (_, r) => `w_part_${wg}_${rows}[${r * wg}u + t] += w_part_${wg}_${rows}[${r * wg}u + t + stride];`).join(" ")} }
    workgroupBarrier(); stride = stride >> 1u;
  }
  if (t < ${rows}u) { let row = row0 + t; if (row < dOut) { q4_y[row] = w_part_${wg}_${rows}[t * ${wg}u]; } }
}`;
const variants = [[64, 4], [32, 4], [64, 8], [128, 4], [32, 8]];
const mod = device.createShaderModule({ code: WGSL + coopWGSL(WG, ROWS) + variants.map(([w, r]) => EXTRA(w, r)).join("\n").replace(/@group\(1\) @binding\(0\) var<storage, read> q4_qs4: array<vec4<u32>>;/g, (m, off, s) => s.indexOf(m) === off ? m : "") });
const info = await mod.getCompilationInfo(); for (const m of info.messages) if (m.type === "error") { console.log("WGSL error:", m.message, "line", m.lineNum); Deno.exit(1); }
const C = GPUShaderStage.COMPUTE;
const l0 = device.createBindGroupLayout({ entries: [0, 1].map((b) => ({ binding: b, visibility: C, buffer: { type: "uniform" } })) });
const l1 = device.createBindGroupLayout({ entries: ["read-only-storage", "read-only-storage", "read-only-storage", "storage", "uniform"].map((t, i) => ({ binding: i, visibility: C, buffer: { type: t } })) });
const bg0 = device.createBindGroup({ layout: l0, entries: [{ binding: 0, resource: { buffer: cfgB } }, { binding: 1, resource: { buffer: frameB } }] });
const bg1 = device.createBindGroup({ layout: l1, entries: [qs, sc, x, y, shape].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
async function timeIt(entry, rows, n = 200) {
  const pipe = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [l0, l1] }), compute: { module: mod, entryPoint: entry } });
  const run = (k) => { const enc = device.createCommandEncoder(); const p = enc.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0, bg0); p.setBindGroup(1, bg1); for (let i = 0; i < k; i++) p.dispatchWorkgroups(Math.ceil(dOut / rows)); p.end(); device.queue.submit([enc.finish()]); return device.queue.onSubmittedWorkDone(); };
  await run(60); const t0 = performance.now(); await run(n); const ms = (performance.now() - t0) / n;
  const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(y, 0, stage, 0, dOut * 4); device.queue.submit([enc.finish()]);
  await stage.mapAsync(GPUMapMode.READ); const out = new Float32Array(stage.getMappedRange()).slice(); stage.unmap();
  return { ms, out };
}
const base = await timeIt("matvec_q4_coop", ROWS);
const gbs = (ms) => (dOut * dIn * 0.5625 / 1e6 / ms).toFixed(0);
let line = `${dOut}x${dIn}: coop(WG${WG})=${base.ms.toFixed(3)}ms ${gbs(base.ms)}GB/s`;
for (const [w, r] of variants) { const v = await timeIt(`xs_w${w}_${r}`, r); let md = 0; for (let i = 0; i < dOut; i++) md = Math.max(md, Math.abs(v.out[i] - base.out[i])); line += `  w${w}x${r}=${v.ms.toFixed(3)}ms ${gbs(v.ms)}GB/s(diff ${md.toExponential(1)})`; }
console.log(line);
