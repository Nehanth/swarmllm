// Micro-benchmark for batched (4-column) q4 matvec variants on synthetic data.
// Compares against the current matvec_q4_coop_b for correctness and time.
import { WGSL, coopWGSL } from "../engine/engine.js";
import { f32ToF16 } from "../engine/gguf.js";
const dIn = +(Deno.env.get("DIN") || 5120), dOut = +(Deno.env.get("DOUT") || 17408);
const WG = +(Deno.env.get("WG") || 256), ROWS = +(Deno.env.get("ROWS") || 4), nb = dIn / 32;
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
const rnd = (() => { let s = 12345; return () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296; })();
const qsArr = new Uint32Array(dOut * dIn / 8); for (let i = 0; i < qsArr.length; i++) qsArr[i] = (rnd() * 4294967296) >>> 0;
const scArr = new Uint32Array(Math.ceil(dOut * nb / 2)); const sc16 = new Uint16Array(scArr.buffer); for (let i = 0; i < sc16.length; i++) sc16[i] = f32ToF16((rnd() - 0.5) * 0.02);
const xs4 = dIn / 4, ys = dOut;   // dense strides (dIn*4 bytes is 256-aligned for these shapes)
const xArr = new Float32Array(4 * dIn); for (let i = 0; i < xArr.length; i++) xArr[i] = rnd() - 0.5;
const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const mk = (arr) => { const b = device.createBuffer({ size: arr.byteLength, usage: S }); device.queue.writeBuffer(b, 0, arr); return b; };
const qs = mk(qsArr), sc = mk(scArr), x = mk(xArr);
const y = device.createBuffer({ size: 4 * dOut * 4, usage: S });
const shape = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(shape, 0, new Uint32Array([dOut, dIn, xs4, ys]));
const shape1 = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(shape1, 0, new Uint32Array([dOut, dIn, 0, 0]));
const cfgB = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM });
const frameB = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM });
const stage = device.createBuffer({ size: 4 * dOut * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

const decode = `
  let lo = vec4<f32>(f32(word & 0xFu), f32((word >> 8u) & 0xFu), f32((word >> 16u) & 0xFu), f32((word >> 24u) & 0xFu)) - vec4<f32>(8.0);
  let hi = vec4<f32>(f32((word >> 4u) & 0xFu), f32((word >> 12u) & 0xFu), f32((word >> 20u) & 0xFu), f32((word >> 28u) & 0xFu)) - vec4<f32>(8.0);`;
const EXTRA = `
// V1: same lane layout as coop (qt = word-in-block, 64 block lanes); each weight
// word decoded once and applied to all 4 columns; 4 sequential reductions.
var<workgroup> v1_part: array<f32, ${8 * WG}>;
@compute @workgroup_size(${WG})
fn xb_v1(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x; let qt = t & 3u; let bl = t >> 2u;
  let dIn = qb_shape.dIn; let dOut = qb_shape.dOut; let xs4 = qb_shape.xs4; let ys = qb_shape.ys;
  let nb = dIn / 32u; let rowWords = dIn / 8u; let row0 = wg.x * 4u;
  ${[0,1,2,3].map(r => [0,1,2,3].map(m => `var a${r}${m}: f32 = 0.0;`).join(" ")).join("\n  ")}
  for (var b: u32 = bl; b < nb; b += ${WG / 4}u) {
    ${[0,1,2,3].map(m => `let xlo${m} = q4_x4[${m}u * xs4 + b * 8u + qt]; let xhi${m} = q4_x4[${m}u * xs4 + b * 8u + qt + 4u];`).join("\n    ")}
    let wIdx = row0 * rowWords + b * 4u + qt; let scBase = row0 * nb + b;
    ${[0,1,2,3].map(r => `if (row0 + ${r}u < dOut) { let word = q4_qs[wIdx + ${r}u * rowWords]; ${decode}
      let s = q4s(scBase + ${r}u * nb);
      ${[0,1,2,3].map(m => `a${r}${m} += s * (dot(lo, xlo${m}) + dot(hi, xhi${m}));`).join(" ")} }`).join("\n    ")}
  }
  // two reduction rounds of 8 values (rows x 2 cols)
  ${[0,1].map(h => `
  ${[0,1,2,3].map(r => [0,1].map(mm => `v1_part[${(r*2+mm) * WG}u + t] = a${r}${2*h+mm};`).join(" ")).join("\n  ")}
  workgroupBarrier();
  { var stride: u32 = ${WG / 2}u;
    while (stride > 0u) {
      if (t < stride) { ${[0,1,2,3,4,5,6,7].map(k => `v1_part[${k * WG}u + t] += v1_part[${k * WG}u + t + stride];`).join(" ")} }
      workgroupBarrier(); stride = stride >> 1u; } }
  if (t < 8u) { let r = t >> 1u; let mm = t & 1u; let row = row0 + r;
    if (row < dOut) { q4_y[(${2*h}u + mm) * ys + row] = v1_part[t * ${WG}u]; } }
  workgroupBarrier();`).join("\n")}
}

// V2: lane owns a column (m = t & 3) and walks whole blocks (64 block lanes);
// reduction is over 64 lanes for all 16 (row, col) sums at once.
var<workgroup> v2_part: array<f32, ${16 * (WG / 4)}>;
@compute @workgroup_size(${WG})
fn xb_v2(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x; let m = t & 3u; let bl = t >> 2u;
  let dIn = qb_shape.dIn; let dOut = qb_shape.dOut; let xs4 = qb_shape.xs4; let ys = qb_shape.ys;
  let nb = dIn / 32u; let rowWords = dIn / 8u; let row0 = wg.x * 4u;
  var a0: f32 = 0.0; var a1: f32 = 0.0; var a2: f32 = 0.0; var a3: f32 = 0.0;
  let xbase = m * xs4;
  for (var b: u32 = bl; b < nb; b += ${WG / 4}u) {
    ${[0,1,2,3].map(q => `let xlo${q} = q4_x4[xbase + b * 8u + ${q}u]; let xhi${q} = q4_x4[xbase + b * 8u + ${q + 4}u];`).join("\n    ")}
    let wIdx = row0 * rowWords + b * 4u; let scBase = row0 * nb + b;
    ${[0,1,2,3].map(r => `if (row0 + ${r}u < dOut) { var acc: f32 = 0.0;
      ${[0,1,2,3].map(q => `{ let word = q4_qs[wIdx + ${r}u * rowWords + ${q}u]; ${decode} acc += dot(lo, xlo${q}) + dot(hi, xhi${q}); }`).join("\n      ")}
      a${r} += q4s(scBase + ${r}u * nb) * acc; }`).join("\n    ")}
  }
  ${[0,1,2,3].map(r => `v2_part[(${r}u * 4u + m) * ${WG / 4}u + bl] = a${r};`).join(" ")}
  workgroupBarrier();
  { var stride: u32 = ${WG / 8}u;
    while (stride > 0u) {
      if (bl < stride) { ${[0,1,2,3].map(r => `v2_part[(${r}u * 4u + m) * ${WG / 4}u + bl] += v2_part[(${r}u * 4u + m) * ${WG / 4}u + bl + stride];`).join(" ")} }
      workgroupBarrier(); stride = stride >> 1u; } }
  if (bl == 0u) { ${[0,1,2,3].map(r => `if (row0 + ${r}u < dOut) { q4_y[m * ys + row0 + ${r}u] = v2_part[(${r}u * 4u + m) * ${WG / 4}u]; }`).join(" ")} }
}
`;
const mod = device.createShaderModule({ code: WGSL + coopWGSL(WG, ROWS) + EXTRA });
const info = await mod.getCompilationInfo(); for (const m of info.messages) if (m.type === "error") { console.log("WGSL error:", m.message, "line", m.lineNum); Deno.exit(1); }
const C = GPUShaderStage.COMPUTE;
const l0 = device.createBindGroupLayout({ entries: [0, 1].map((b) => ({ binding: b, visibility: C, buffer: { type: "uniform" } })) });
const l1 = device.createBindGroupLayout({ entries: ["read-only-storage", "read-only-storage", "read-only-storage", "storage", "uniform"].map((t, i) => ({ binding: i, visibility: C, buffer: { type: t } })) });
const bg0 = device.createBindGroup({ layout: l0, entries: [{ binding: 0, resource: { buffer: cfgB } }, { binding: 1, resource: { buffer: frameB } }] });
const bgB = device.createBindGroup({ layout: l1, entries: [qs, sc, x, y, shape].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
const bg1 = device.createBindGroup({ layout: l1, entries: [qs, sc, x, y, shape1].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
async function timeIt(entry, bg, n = 200) {
  const pipe = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [l0, l1] }), compute: { module: mod, entryPoint: entry } });
  const run = (k) => { const enc = device.createCommandEncoder(); const p = enc.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0, bg0); p.setBindGroup(1, bg); for (let i = 0; i < k; i++) p.dispatchWorkgroups(Math.ceil(dOut / ROWS)); p.end(); device.queue.submit([enc.finish()]); return device.queue.onSubmittedWorkDone(); };
  await run(30); const t0 = performance.now(); await run(n); const ms = (performance.now() - t0) / n;
  const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(y, 0, stage, 0, 4 * dOut * 4); device.queue.submit([enc.finish()]);
  await stage.mapAsync(GPUMapMode.READ); const out = new Float32Array(stage.getMappedRange()).slice(); stage.unmap();
  return { ms, out };
}
const single = await timeIt("matvec_q4_coop", bg1);
const cur = await timeIt("matvec_q4_coop_b", bgB);
const res = { single: single.ms, cur_b: cur.ms };
const outs = { cur_b: cur.out };
for (const v of ["xb_v1", "xb_v2"]) { device.queue.writeBuffer(y, 0, new Float32Array(4 * dOut)); const r = await timeIt(v, bgB); res[v] = r.ms; outs[v] = r.out; }
console.log(`${dOut}x${dIn}: ` + Object.entries(res).map(([k, v]) => `${k}=${v.toFixed(3)}ms`).join("  ") + `  (batched/single: cur ${(cur.ms / single.ms).toFixed(2)}x, v1 ${(res.xb_v1 / single.ms).toFixed(2)}x, v2 ${(res.xb_v2 / single.ms).toFixed(2)}x)`);
for (const v of ["xb_v1", "xb_v2"]) { let md = 0, mref = 0; for (let i = 0; i < 4 * dOut; i++) { md = Math.max(md, Math.abs(outs[v][i] - outs.cur_b[i])); mref = Math.max(mref, Math.abs(outs.cur_b[i])); } console.log(`${v} max|diff| vs cur_b = ${md.toExponential(2)} (max|ref| ${mref.toFixed(3)})`); }
