// Row-stationary Q4_0 GEMM experiment (16 columns) + the old prototype, timed in one process.
// Run: deno run --unstable-webgpu --allow-env --allow-read benchmarks/bench_gemm_rowstat.js
// Multi-config, single-process GEMM harness (all candidates timed round-robin under continuous queue
// pressure, so the GB10 clock governor state is the same for every row; results reported relative to
// the in-run streaming probe). Env: CFGS="T=64,R=2,KB=2,S=4;T=64,R=4,KB=2,S=8" DOUT=17408 ROUNDS=4 REPS=60
import { WGSL, coopWGSL, probeUnpack } from "/home/nehanth/bello/engine/engine.js";
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize } });
console.log("maxComputeWorkgroupStorageSize", adapter.limits.maxComputeWorkgroupStorageSize);
const env = (k, d) => Deno.env.get(k) ?? d;
const dOut = +env("DOUT", 17408), dIn = 5120, N = 16, nb = dIn / 32;
const ROUNDS = +env("ROUNDS", 4), REPS = +env("REPS", 60);
const Sx = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const rnd = (n, f) => { const a = new Uint32Array(n); for (let i = 0; i < n; i++) a[i] = f(i); return a; };
const qsData = rnd(dOut * dIn / 8, () => (Math.random() * 2 ** 32) >>> 0);
const f16 = (v) => { const f = new Float32Array([v]); const u = new Uint32Array(f.buffer)[0]; const s = (u >> 16) & 0x8000, e = ((u >> 23) & 0xff) - 112, m = (u >> 13) & 0x3ff; return e <= 0 ? s : s | (e << 10) | m; };
const scData = rnd(Math.ceil(dOut * nb / 2), () => f16(0.01 + Math.random() * 0.02) | (f16(0.01 + Math.random() * 0.02) << 16));
const xData = new Float32Array(N * dIn); for (let i = 0; i < xData.length; i++) xData[i] = Math.random() * 2 - 1;
const mk = (data, usage = Sx) => { const b = device.createBuffer({ size: data.byteLength, usage }); device.queue.writeBuffer(b, 0, data); return b; };
const U = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
device.addEventListener("uncapturederror", (e) => console.log("GPU ERROR:", e.error.message.slice(0, 400)));
const qs = mk(qsData), sc = mk(scData), x = mk(xData);
const xT = device.createBuffer({ size: N * dIn * 4, usage: Sx });
const y2 = device.createBuffer({ size: N * dOut * 4, usage: Sx });
const shape = mk(new Uint32Array([dOut, dIn, dIn / 4, dOut]), U);
const cfgB = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM }), frameB = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM });
const C = GPUShaderStage.COMPUTE;
const l0 = device.createBindGroupLayout({ entries: [0, 1].map((b) => ({ binding: b, visibility: C, buffer: { type: "uniform" } })) });
const l1 = device.createBindGroupLayout({ entries: ["read-only-storage", "read-only-storage", "read-only-storage", "storage", "uniform"].map((t, i) => ({ binding: i, visibility: C, buffer: { type: t } })) });
const bg0 = device.createBindGroup({ layout: l0, entries: [{ binding: 0, resource: { buffer: cfgB } }, { binding: 1, resource: { buffer: frameB } }] });
const bg1 = (a, b, c, d) => device.createBindGroup({ layout: l1, entries: [a, b, c, d, shape].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
const unpack = await probeUnpack(device);
const rng = (n) => Array.from({ length: n }, (_, i) => i);
const HDR = `
struct BShape { dOut: u32, dIn: u32, xs4: u32, ys: u32 };
@group(0) @binding(0) var<uniform> cfg0: vec4<u32>;
@group(0) @binding(1) var<uniform> frame0: vec4<u32>;`;
const mkPipe = async (code, entry) => { const m = device.createShaderModule({ code }); const info = await m.getCompilationInfo(); for (const msg of info.messages) if (msg.type === "error") console.log("WGSL", entry, msg.lineNum, msg.message.slice(0, 300)); return device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [l0, l1] }), compute: { module: m, entryPoint: entry } }); };

// ---------------- v4 "row-stationary" kernel generator ----------------
function genV4(c) {
  const { T, R, KB, S, DQ = "unpack", NQ = 4 } = c; const PR = new Set(String(c.PROBE || "").split("+"));
  const TM = T * R, nStages = nb / KB; if (nStages % S) throw new Error("S must divide " + nStages); const stPerWG = nStages / S;
  const WV = TM * KB, WPT = WV / T, XV = 32 * KB * 4, XPT = XV / T; if (WV % T || XV % T) throw new Error("T must divide stage sizes");
  const RR = rng(R), Q4 = rng(4), QN = rng(NQ);
  const dq = PR.has("nodq") ? (m, s) => `bitcast<vec4<f32>>(vec4<u32>(${m}, ${m} >> 1u, ${m} >> 2u, ${m} >> 3u)) * ${s}` : DQ === "magic"
    ? (m, s) => `(vec4<f32>(bitcast<f32>((${m} & 0xFu) | 0x4B000000u), bitcast<f32>(((${m} >> 8u) & 0xFu) | 0x4B000000u), bitcast<f32>(((${m} >> 16u) & 0xFu) | 0x4B000000u), bitcast<f32>((${m} >> 24u) | 0x4B000000u)) - vec4<f32>(8388616.0)) * ${s}`
    : (m, s) => `(vec4<f32>(unpack4xU8(${m})) - vec4<f32>(8.0)) * ${s}`;
  const code = `${HDR}
@group(1) @binding(0) var<storage, read> g_qs: array<vec4<u32>>;
@group(1) @binding(1) var<storage, read> g_sc: array<u32>;
@group(1) @binding(2) var<storage, read> g_xT: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> g_p: array<f32>;
@group(1) @binding(4) var<uniform> g_shape: BShape;
var<workgroup> Wq: array<vec4<u32>, ${WV}>;
var<workgroup> Xs: array<vec4<f32>, ${XV}>;
@compute @workgroup_size(${T})
fn gemm16(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let dOut = g_shape.dOut; let dIn = g_shape.dIn; let nb = dIn / 32u;
  let wgl = wg.y * 32768u + wg.x;
  let tile = wgl / ${S}u; let split = wgl % ${S}u;
  let row0 = tile * ${TM}u; let st0 = split * ${stPerWG}u; let rb = row0 + t * ${R}u;
  ${RR.map((r) => Q4.map((q) => `var a${r}_${q} = vec4<f32>(0.0);`).join(" ")).join("\n  ")}
  ${rng(WPT).map((j) => `let li${j} = t + ${j * T}u; let lr${j} = min(row0 + li${j} / ${KB}u, dOut - 1u); let lb${j} = li${j} % ${KB}u;`).join("\n  ")}
  ${rng(WPT).map((j) => `var w${j} = g_qs[lr${j} * nb + st0 * ${KB}u + lb${j}];`).join("\n  ")}
  ${rng(XPT).map((j) => `var xv${j} = g_xT[st0 * ${XV}u + t + ${j * T}u];`).join("\n  ")}
  for (var s: u32 = 0u; s < ${stPerWG}u; s++) {
    ${PR.has("nobarrier") ? "" : "workgroupBarrier();"}
    ${PR.has("nostore") ? "" : rng(WPT).map((j) => `Wq[li${j}] = w${j};`).join(" ")}
    ${PR.has("nostore") ? "" : rng(XPT).map((j) => `Xs[t + ${j * T}u] = xv${j};`).join(" ")}
    ${PR.has("nobarrier") ? "" : "workgroupBarrier();"}
    let bs = (st0 + s) * ${KB}u;
    if (s + 1u < ${stPerWG}u && ${PR.has("noload") ? "false" : "true"}) {
      ${rng(WPT).map((j) => PR.has("contig") ? `w${j} = g_qs[(tile * ${stPerWG}u + s + 1u) * ${WV}u + li${j}];` : `w${j} = g_qs[lr${j} * nb + bs + ${KB}u + lb${j}];`).join(" ")}
      ${rng(XPT).map((j) => `xv${j} = g_xT[(st0 + s + 1u) * ${XV}u + t + ${j * T}u];`).join(" ")}
    }
    ${RR.map((r) => rng(Math.max(1, KB >> 1)).map((pp) => PR.has("noscale") ? `let sw${r}_${pp} = 0x3C003C00u + s;` : `let sw${r}_${pp} = g_sc[((min(rb + ${r}u, dOut - 1u) * nb + bs) >> 1u) + ${pp}u];`).join(" ")).join(" ")}
    ${PR.has("nofma") ? "" : rng(KB).map((b) => `
    {
      ${RR.map((r) => `let s${r} = unpack2x16float(sw${r}_${b >> 1})[${KB === 1 ? "(bs & 1u)" : String(b & 1) + "u"}];`).join(" ")}
      ${RR.map((r) => `let wa${r} = Wq[(t * ${R}u + ${r}u) * ${KB}u + ${b}u];`).join("\n      ")}
      ${rng(4).map((j) => `
      { ${RR.map((r) => `let lo${r} = ${dq(`(wa${r}[${j}] & 0x0F0F0F0Fu)`, `s${r}`)}; let hi${r} = ${dq(`((wa${r}[${j}] >> 4u) & 0x0F0F0F0Fu)`, `s${r}`)};`).join(" ")}
        ${rng(4).map((i) => { const kl = 32 * b + 4 * j + i, kh = kl + 16; return `
        { ${QN.map((q) => `let xl${q} = Xs[${kl * 4 + q}u];`).join(" ")} ${RR.map((r) => QN.map((q) => `a${r}_${q} += lo${r}[${i}] * xl${q};`).join(" ")).join(" ")} }
        { ${QN.map((q) => `let xh${q} = Xs[${kh * 4 + q}u];`).join(" ")} ${RR.map((r) => QN.map((q) => `a${r}_${q} += hi${r}[${i}] * xh${q};`).join(" ")).join(" ")} }`; }).join("")}
      }`).join("")}
    }`).join("")}
  }
  let pb = split * ${N}u * dOut;
  ${PR.has("noepi") ? `if (a0_0.x == 123456.0) { g_p[pb + rb] = ${PR.has("nostore") ? "1.0" : "f32(Wq[t].x)"}; }` : RR.map((r) => `if (rb + ${r}u < dOut) { ${Q4.map((q) => rng(4).map((cc) => `g_p[pb + ${4 * q + cc}u * dOut + rb + ${r}u] = a${r}_${q}[${cc}];`).join(" ")).join(" ")} }`).join("\n  ")}
}`;
  return { code, entry: "gemm16", TM, S, wgs: Math.ceil(dOut / TM) * S, needsXT: true };
}
// ---------------- the old prototype (benchmarks/bench_gemm.js), verbatim structure ----------------
function genProto(c) {
  const RT = c.RT || 2, T = 128, TM = T / 4 * RT, R = rng(RT), WPT = TM * 8 / T, loads = rng(WPT);
  const code = `${HDR}
@group(1) @binding(0) var<storage, read> g_qs: array<u32>;
@group(1) @binding(1) var<storage, read> g_sc: array<u32>;
@group(1) @binding(2) var<storage, read> g_x: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> g_y: array<f32>;
@group(1) @binding(4) var<uniform> g_shape: BShape;
var<workgroup> Ws: array<vec4<f32>, ${TM * 17}>;
var<workgroup> Xs: array<vec4<f32>, ${16 * 16}>;
fn g_sc_at(i: u32) -> f32 { return unpack2x16float(g_sc[i >> 1u])[i & 1u]; }
fn nib_lo(w: u32) -> vec4<f32> { return vec4<f32>(unpack4xU8(w & 0x0F0F0F0Fu)) - vec4<f32>(8.0); }
fn nib_hi(w: u32) -> vec4<f32> { return vec4<f32>(unpack4xU8((w >> 4u) & 0x0F0F0F0Fu)) - vec4<f32>(8.0); }
@compute @workgroup_size(${T})
fn gemm16(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let dIn = g_shape.dIn; let dOut = g_shape.dOut; let xs4 = g_shape.xs4; let ys = g_shape.ys;
  let nb = dIn / 32u; let rowWords = dIn / 8u; let nbp = nb / 2u;
  let row0 = (wg.y * 32768u + wg.x) * ${TM}u;
  let rq = t >> 2u; let cq = t & 3u;
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
  return { code, entry: "gemm16", TM, S: 1, wgs: Math.ceil(dOut / TM), needsXT: false };
}
const xtransCode = `${HDR}
@group(1) @binding(0) var<storage, read> g_x: array<f32>;
@group(1) @binding(1) var<storage, read> g_u1: array<u32>;
@group(1) @binding(2) var<storage, read> g_u2: array<u32>;
@group(1) @binding(3) var<storage, read_write> g_xT: array<f32>;
@group(1) @binding(4) var<uniform> g_shape: BShape;
@compute @workgroup_size(64)
fn xtrans(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; let dIn = g_shape.dIn; if (i >= dIn * ${N}u) { return; }
  g_xT[i] = g_x[(i % ${N}u) * dIn + i / ${N}u];
}`;
const reduceCode = (S) => `${HDR}
@group(1) @binding(0) var<storage, read> g_p: array<f32>;
@group(1) @binding(1) var<storage, read> g_u1: array<u32>;
@group(1) @binding(2) var<storage, read> g_u2: array<u32>;
@group(1) @binding(3) var<storage, read_write> g_y: array<f32>;
@group(1) @binding(4) var<uniform> g_shape: BShape;
@compute @workgroup_size(64)
fn reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; let n = ${N}u * g_shape.dOut; if (i >= n) { return; }
  var acc = 0.0; ${rng(S).map((s) => `acc += g_p[${s}u * n + i];`).join(" ")} g_y[i] = acc;
}`;
const bwCode = `${HDR}
@group(1) @binding(0) var<storage, read> g_qs: array<vec4<u32>>;
@group(1) @binding(1) var<storage, read> g_u1: array<u32>;
@group(1) @binding(2) var<storage, read> g_u2: array<u32>;
@group(1) @binding(3) var<storage, read_write> g_o: array<u32>;
@group(1) @binding(4) var<uniform> g_shape: BShape;
@compute @workgroup_size(256)
fn bw(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let base = wg.x * 4096u; var acc = vec4<u32>(0u);
  ${rng(16).map((i) => `acc ^= g_qs[base + lid.x + ${i * 256}u];`).join(" ")}
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678u) { g_o[wg.x] = 1u; }
}`;
// ---------------- build candidates ----------------
const parse = (s) => Object.fromEntries(s.split(",").map((kv) => { const [k, v] = kv.split("="); return [k, isNaN(+v) ? v : +v]; }));
const cfgs = env("CFGS", "proto:RT=2;v4:T=64,R=2,KB=2,S=4").split(";").map((s) => { const [name, rest] = s.split(":"); return { name, kind: name.startsWith("proto") ? "proto" : "v4", ...parse(rest) }; });
const pX = await mkPipe(xtransCode, "xtrans"), pBW = await mkPipe(bwCode, "bw");
const p4 = await mkPipe(WGSL + coopWGSL(256, 4, 64, 4, 4, unpack), "matvec_q4_coop_b");
const p1 = await mkPipe(WGSL + coopWGSL(64, 4, 64, 4, 4, unpack), "matvec_q4_coop");
const bgX = bg1(x, sc, sc, xT);
const cands = [];
for (const c of cfgs) {
  const g = c.kind === "proto" ? genProto(c) : genV4(c);
  const pipe = await mkPipe(g.code, g.entry);
  const y = device.createBuffer({ size: N * dOut * 4, usage: Sx });
  const part = g.S > 1 ? device.createBuffer({ size: g.S * N * dOut * 4, usage: Sx }) : null;
  const pR = g.S > 1 ? await mkPipe(reduceCode(g.S), "reduce") : null;
  const bgG = bg1(qs, sc, g.needsXT ? xT : x, part || y), bgR = part ? bg1(part, sc, sc, y) : null;
  const run = (n) => { const e = device.createCommandEncoder(); const p = e.beginComputePass(); p.setBindGroup(0, bg0);
    for (let i = 0; i < n; i++) {
      if (g.needsXT) { p.setPipeline(pX); p.setBindGroup(1, bgX); p.dispatchWorkgroups(Math.ceil(N * dIn / 64)); }
      p.setPipeline(pipe); p.setBindGroup(1, bgG); if (g.wgs > 32768) p.dispatchWorkgroups(32768, Math.ceil(g.wgs / 32768)); else p.dispatchWorkgroups(g.wgs);
      if (pR) { p.setPipeline(pR); p.setBindGroup(1, bgR); p.dispatchWorkgroups(Math.ceil(N * dOut / 64)); }
    } p.end(); device.queue.submit([e.finish()]); };
  cands.push({ name: c.name + "(" + Object.entries(c).filter(([k]) => !["name", "kind"].includes(k)).map(([k, v]) => k + "=" + v).join(",") + ")", run, y, wgs: g.wgs, TM: g.TM });
}
const shapes4 = [0, 1, 2, 3].map(() => mk(new Uint32Array([dOut, dIn, dIn / 4, dOut]), U));
const bgs4 = [0, 1, 2, 3].map((k) => device.createBindGroup({ layout: l1, entries: [
  { binding: 0, resource: { buffer: qs } }, { binding: 1, resource: { buffer: sc } },
  { binding: 2, resource: { buffer: x, offset: k * 4 * dIn * 4, size: 4 * dIn * 4 } },
  { binding: 3, resource: { buffer: y2, offset: k * 4 * dOut * 4, size: 4 * dOut * 4 } }, { binding: 4, resource: { buffer: shapes4[k] } }] }));
const run4 = (n) => { const e = device.createCommandEncoder(); const p = e.beginComputePass(); p.setPipeline(p4); p.setBindGroup(0, bg0); for (let i = 0; i < n; i++) for (let k = 0; k < 4; k++) { p.setBindGroup(1, bgs4[k]); p.dispatchWorkgroups(Math.ceil(dOut / 4)); } p.end(); device.queue.submit([e.finish()]); };
const bgBW = bg1(qs, sc, sc, y2), bg1c = bg1(qs, sc, x, y2);
const runBW = (n) => { const e = device.createCommandEncoder(); const p = e.beginComputePass(); p.setBindGroup(0, bg0); p.setPipeline(pBW); p.setBindGroup(1, bgBW); for (let i = 0; i < n; i++) p.dispatchWorkgroups(dOut * dIn / 32 / 4096); p.end(); device.queue.submit([e.finish()]); };
const run1 = (n) => { const e = device.createCommandEncoder(); const p = e.beginComputePass(); p.setBindGroup(0, bg0); p.setPipeline(p1); p.setBindGroup(1, bg1c); for (let i = 0; i < n; i++) p.dispatchWorkgroups(Math.ceil(dOut / 4)); p.end(); device.queue.submit([e.finish()]); };
// correctness
run4(1); for (const c of cands) c.run(1); await device.queue.onSubmittedWorkDone();
const rd = async (b) => { const st = device.createBuffer({ size: N * dOut * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }); const e = device.createCommandEncoder(); e.copyBufferToBuffer(b, 0, st, 0, N * dOut * 4); device.queue.submit([e.finish()]); await st.mapAsync(GPUMapMode.READ); const out = Float32Array.from(new Float32Array(st.getMappedRange())); st.unmap(); return out; };
const B = await rd(y2);
for (const c of cands) { const A = await rd(c.y); let num = 0, den = 0; for (let i = 0; i < A.length; i++) { num += (A[i] - B[i]) ** 2; den += B[i] ** 2; } c.rel = Math.sqrt(num / den); }
// timing: continuous pressure, round-robin, min of rounds
const all = [{ name: "bw-probe(stream)", run: runBW }, { name: "gemv coop 1-col", run: run1 }, { name: "coop_b 4x4cols", run: run4 }, ...cands];
for (const a of all) a.best = Infinity;
const warm = async () => { for (let k = 0; k < 3; k++) { for (const a of all) a.run(10); } await device.queue.onSubmittedWorkDone(); };
const tw = performance.now(); while (performance.now() - tw < 2500) await warm();
for (let r = 0; r < ROUNDS; r++) for (const a of all) { a.run(10); const t0 = performance.now(); a.run(REPS); await device.queue.onSubmittedWorkDone(); a.best = Math.min(a.best, (performance.now() - t0) / REPS); }
const gb = dOut * dIn * 0.5625 / 1e9, flop = 2 * dOut * dIn * N / 1e12, tbw = all[0].best;
console.log(`\n${dOut}x${dIn} Q4_0 x 16 cols  (${(gb * 1e3).toFixed(0)} MB weights; time = min over ${ROUNDS} rounds of ${REPS} back-to-back reps, all kernels interleaved)`);
for (const a of all) console.log(`${a.name.padEnd(46)} ${a.best.toFixed(3)} ms  ${(gb / a.best * 1e3).toFixed(0).padStart(4)} GB/s  ${(flop / a.best * 1e3).toFixed(1).padStart(5)} TFLOPS  x${(tbw / a.best).toFixed(2)} of stream-probe${a.rel !== undefined ? `  relDiff=${a.rel.toExponential(1)}` : ""}${a.wgs ? `  wgs=${a.wgs} TM=${a.TM}` : ""}`);
