// gate/up fused kernel: single vs batched cost at the model's shape.
import { WGSL, coopWGSL } from "../engine/engine.js";
import { f32ToF16 } from "../engine/gguf.js";
const dIn = 5120, dOut = 17408, WG = 256, ROWS = 4, nb = dIn / 32; const WARM = +(Deno.env.get("WARM") || 3);
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
const rnd = (() => { let s = 777; return () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296; })();
const mkQ = () => { const a = new Uint32Array(dOut * dIn / 8); for (let i = 0; i < a.length; i++) a[i] = (rnd() * 4294967296) >>> 0; return a; };
const mkS = () => { const a = new Uint32Array(Math.ceil(dOut * nb / 2)); const s16 = new Uint16Array(a.buffer); for (let i = 0; i < s16.length; i++) s16[i] = f32ToF16((rnd() - 0.5) * 0.02); return a; };
const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const mk = (arr) => { const b = device.createBuffer({ size: arr.byteLength, usage: S }); device.queue.writeBuffer(b, 0, arr); return b; };
const gqs = mk(mkQ()), gsc = mk(mkS()), uqs = mk(mkQ()), usc = mk(mkS());
const xArr = new Float32Array(4 * dIn); for (let i = 0; i < xArr.length; i++) xArr[i] = rnd() - 0.5;
const x = mk(xArr), y = device.createBuffer({ size: 4 * dOut * 4, usage: S });
const shapeB = mk(new Uint32Array([dOut, dIn, dIn / 4, dOut])); const shape1 = mk(new Uint32Array([dOut, dIn, 0, 0]));
const u = (b) => { const ub = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(b, 0, ub, 0, 16); device.queue.submit([enc.finish()]); return ub; };
const shB = u(shapeB), sh1 = u(shape1);
const cfgB = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM }), frameB = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM });
const mod = device.createShaderModule({ code: WGSL + coopWGSL(WG, ROWS) });
const C = GPUShaderStage.COMPUTE;
const l0 = device.createBindGroupLayout({ entries: [0, 1].map((b) => ({ binding: b, visibility: C, buffer: { type: "uniform" } })) });
const mkL = (types) => device.createBindGroupLayout({ entries: types.map((t, i) => ({ binding: i, visibility: C, buffer: { type: t } })) });
const lGU = mkL(["read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "read-only-storage", "storage", "uniform"]);
const lMV = mkL(["read-only-storage", "read-only-storage", "read-only-storage", "storage", "uniform"]);
const bg0 = device.createBindGroup({ layout: l0, entries: [{ binding: 0, resource: { buffer: cfgB } }, { binding: 1, resource: { buffer: frameB } }] });
const bgOf = (l, bufs) => device.createBindGroup({ layout: l, entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
async function timeIt(entry, l, bg, n = +(Deno.env.get("N") || 20)) {
  const pipe = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [l0, l] }), compute: { module: mod, entryPoint: entry } });
  const run = (k) => { const enc = device.createCommandEncoder(); const p = enc.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0, bg0); p.setBindGroup(1, bg); for (let i = 0; i < k; i++) p.dispatchWorkgroups(Math.ceil(dOut / ROWS)); p.end(); device.queue.submit([enc.finish()]); return device.queue.onSubmittedWorkDone(); };
  await run(WARM); const t0 = performance.now(); await run(n); return (performance.now() - t0) / n;
}
const r = {
  mv1: await timeIt("matvec_q4_coop", lMV, bgOf(lMV, [gqs, gsc, x, y, sh1])),
  mvB: await timeIt("matvec_q4_coop_b", lMV, bgOf(lMV, [gqs, gsc, x, y, shB])),
  gu1: await timeIt("matvec_q4_gu", lGU, bgOf(lGU, [gqs, gsc, uqs, usc, x, y, sh1])),
  guB: await timeIt("matvec_q4_gu_b", lGU, bgOf(lGU, [gqs, gsc, uqs, usc, x, y, shB])),
};
console.log(Object.entries(r).map(([k, v]) => `${k}=${v.toFixed(3)}ms`).join("  "), ` | gu_b/gu1 = ${(r.guB / r.gu1).toFixed(2)}x, gu1 vs 2x mv1 = ${(r.gu1 / (2 * r.mv1)).toFixed(2)}x, gu_b vs 2x mvB = ${(r.guB / (2 * r.mvB)).toFixed(2)}x`);
