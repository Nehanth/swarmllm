// dn_delta_mc at the 27B's DeltaNet shape (48 value heads, 16 key heads, dState 128): the shipped
// register-resident kernel vs the previous global-memory kernel (kept here verbatim as the
// reference), on the same random state and inputs, 1-16 columns, snapshots on and replay mode.
// S, outputs and snapshot slots must be bit-identical.
//   NODE_PATH=... node tests/e2e/dn_delta_synth.mjs
import { loadPlaywright, chromiumPath, GPU_ARGS, serveRepo } from "./engine_synth.mjs";
const PORT = 18998;
const REF = "@compute @workgroup_size(128)\nfn dn_delta_mc_ref(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {\n  let h = wg.x; let j = lid.x; let dS = dlm_dn.dState;\n  if (h >= dlm_dn.nVH || j >= dS) { return; }\n  let kh = h % dlm_dn.nKH;\n  let kOff = kh * dS; let vOff = h * dS; let Sb = h * dS * dS;\n  let scale = inverseSqrt(f32(dS));\n  let nCols = max(frame.nCols, 1u);\n  let sSize = dlm_dn.nVH * dS * dS;\n  for (var col: u32 = 0u; col < nCols; col++) {\n    let qo = col * dlm_mc.s0 + kOff;\n    let ko = col * dlm_mc.s0 + dlm_dn.keyDim + kOff;\n    let vo = col * dlm_mc.s0 + 2u * dlm_dn.keyDim + vOff;\n    let decay = dlm_decay[col * dlm_mc.s1 + h];\n    var vhat: f32 = 0.0;\n    var sq: f32 = 0.0;\n    var kq: f32 = 0.0;\n    for (var i: u32 = 0u; i < dS; i++) {\n      let idx = Sb + i * dS + j;\n      let sdec = dlm_s[idx] * decay;\n      dlm_s[idx] = sdec;\n      let ki = dlm_c[ko + i];\n      let qi = dlm_c[qo + i];\n      vhat += sdec * ki;\n      sq += sdec * qi;\n      kq += ki * qi;\n    }\n    let d = (dlm_c[vo + j] - vhat) * dlm_beta[col * dlm_mc.s1 + h];\n    for (var i: u32 = 0u; i < dS; i++) {\n      let idx = Sb + i * dS + j;\n      dlm_s[idx] += dlm_c[ko + i] * d;\n    }\n    dlm_o[col * dlm_mc.s2 + vOff + j] = (sq + d * kq) * scale;\n    let dlSB = frame.snap & 0xffu;     // snapshot slot base + 1 (0 = off)\n    // bit 31: replay rollback (the engine keeps one pre-verify state and re-runs this kernel\n    // on rejection), so no per-column state snapshots; the conv snapshots stay (they are tiny)\n    if (dlSB != 0u && (frame.snap & 0x80000000u) == 0u && dlSB + col < ((frame.snap >> 8u) & 0xffu)) {\n      let slot = dlSB - 1u + col;\n      for (var i: u32 = 0u; i < dS; i++) { dlm_shadow[slot * sSize + Sb + i * dS + j] = dlm_s[Sb + i * dS + j]; }\n    }\n  }\n}\n\n";

async function pageMain({ REF }) {
  const { WGSL } = await import("/engine/wgsl/base.js");
  const { WGSL2 } = await import("/engine/wgsl/qwen35.js");
  const out = [], say = (s) => out.push(s);
  const adapter = await navigator.gpu.requestAdapter();
  const dev = await adapter.requestDevice();
  const errs = []; dev.addEventListener("uncapturederror", (e) => errs.push(e.error?.message));
  const mod = dev.createShaderModule({ code: WGSL + WGSL2 + REF });
  const info = await mod.getCompilationInfo(); for (const m of info.messages) if (m.type === "error") say("WGSL error: " + m.message + " line " + m.lineNum);
  const C = GPUShaderStage.COMPUTE, S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, U = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
  const l0 = dev.createBindGroupLayout({ entries: [0, 1].map((b) => ({ binding: b, visibility: C, buffer: { type: "uniform" } })) });
  const t = { u: "uniform", ro: "read-only-storage", rw: "storage" };
  const l1 = dev.createBindGroupLayout({ entries: ["ro", "ro", "ro", "rw", "rw", "u", "u", "rw"].map((x, i) => ({ binding: i, visibility: C, buffer: { type: t[x] } })) });
  const layout = dev.createPipelineLayout({ bindGroupLayouts: [l0, l1] });
  const pipe = (e) => dev.createComputePipeline({ layout, compute: { module: mod, entryPoint: e } });
  const pNew = pipe("dn_delta_mc"), pRef = pipe("dn_delta_mc_ref");
  const nVH = 48, nKH = 16, dS = 128, keyDim = nKH * dS, dInner = nVH * dS, convDim = 2 * keyDim + dInner, NCmax = 16;
  const al = (n) => Math.ceil(n * 4 / 256) * 256 / 4;
  const s0 = al(convDim), s1 = al(nVH), s2 = al(dInner);
  let seed = 7; const rnd = () => { seed = (seed * 1103515245 + 12345) >>> 0; return seed / 4294967296; };
  const conv = new Float32Array(NCmax * s0).map(() => (rnd() - 0.5) * 0.2);
  const beta = new Float32Array(NCmax * s1).map(() => rnd());
  const decay = new Float32Array(NCmax * s1).map(() => 0.5 + 0.5 * rnd());
  const S0 = new Float32Array(nVH * dS * dS).map(() => (rnd() - 0.5) * 0.1);
  const mk = (a) => { const b = dev.createBuffer({ size: a.byteLength, usage: S }); dev.queue.writeBuffer(b, 0, a); return b; };
  const bConv = mk(conv), bBeta = mk(beta), bDecay = mk(decay);
  const cfg = dev.createBuffer({ size: 48, usage: U });
  const dnData = new ArrayBuffer(48); new Uint32Array(dnData).set([convDim, dS, nKH, nVH, keyDim, 64, 256, dInner]); new Float32Array(dnData)[8] = 1e7; new Float32Array(dnData)[9] = 1e-6;
  const dn = dev.createBuffer({ size: 48, usage: U }); dev.queue.writeBuffer(dn, 0, dnData);
  const mc = dev.createBuffer({ size: 16, usage: U }); dev.queue.writeBuffer(mc, 0, new Uint32Array([0, s0, s1, s2]));
  const frame = dev.createBuffer({ size: 16, usage: U });
  const g0 = dev.createBindGroup({ layout: l0, entries: [{ binding: 0, resource: { buffer: cfg } }, { binding: 1, resource: { buffer: frame } }] });
  const read = async (b, n) => { const st = dev.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }); const e = dev.createCommandEncoder(); e.copyBufferToBuffer(b, 0, st, 0, n * 4); dev.queue.submit([e.finish()]); await st.mapAsync(GPUMapMode.READ); const a = new Float32Array(st.getMappedRange()).slice(); st.unmap(); st.destroy(); return a; };
  const run = async (p, nCols, snap) => {
    const bS = mk(S0), bO = dev.createBuffer({ size: NCmax * s2 * 4, usage: S }), bSh = dev.createBuffer({ size: 7 * S0.byteLength, usage: S });
    dev.queue.writeBuffer(frame, 0, new Uint32Array([0, 1, nCols, snap >>> 0]));
    const g1 = dev.createBindGroup({ layout: l1, entries: [bConv, bBeta, bDecay, bS, bO, mc, dn, bSh].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const e = dev.createCommandEncoder(); const ps = e.beginComputePass(); ps.setPipeline(p); ps.setBindGroup(0, g0); ps.setBindGroup(1, g1); ps.dispatchWorkgroups(nVH); ps.end(); dev.queue.submit([e.finish()]);
    return { S: await read(bS, S0.length), O: await read(bO, NCmax * s2), Sh: await read(bSh, 7 * S0.length) };
  };
  let allOk = true;
  for (const [nCols, snap, tag] of [[1, 0, "no snapshots"], [5, (5 << 8) | 1, "snapshots"], [8, (8 << 8) | 1, "snapshots"], [8, ((8 << 8) | 1) | 0x80000000, "replay (no state snapshots)"], [16, 0, "no snapshots"]]) {
    const a = await run(pRef, nCols, snap), b = await run(pNew, nCols, snap);
    const cnt = (x, y) => { let d = 0; for (let i = 0; i < x.length; i++) if (!Object.is(x[i], y[i])) d++; return d; };
    // the reference kernel predates replay mode: with bit 31 it still writes snapshots, the new one must not
    const dS_ = cnt(a.S, b.S), dO = cnt(a.O, b.O), dSh = snap & 0x80000000 ? cnt(new Float32Array(b.Sh.length), b.Sh) : cnt(a.Sh, b.Sh);
    const ok = !dS_ && !dO && !dSh; if (!ok) allOk = false;
    say(`${ok ? "PASS" : "FAIL"} nCols ${nCols}, ${tag}: differing S ${dS_}/${a.S.length}, out ${dO}/${a.O.length}, snapshot slots ${dSh}${snap & 0x80000000 ? " (must stay empty)" : ""}`);
  }
  if (errs.length) { allOk = false; say("GPU errors: " + errs.slice(0, 2).join(" | ")); }
  return { out, allOk };
}
const srv = serveRepo(PORT, {});
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), args: GPU_ARGS });
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${PORT}/favicon.svg`);
const res = await page.evaluate(pageMain, { REF });
for (const l of res.out) console.log(l);
console.log(res.allOk ? "DN_DELTA PASS" : "DN_DELTA FAIL");
await browser.close(); srv.close();
process.exit(res.allOk ? 0 : 1);
