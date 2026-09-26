// MoE model end to end in the engine (synthetic qwen35moe file: 16 experts, top-4, shared expert):
// every layer's FFN block (router -> experts -> shared expert -> residual) against a float64
// reference computed from the GGUF's own weights, then greedy decoding, speculative == plain.
//   NODE_PATH=... node tests/e2e/moe_synth.mjs [--model f.gguf]
import fs from "fs"; import os from "os"; import path from "path";
import { loadPlaywright, chromiumPath, GPU_ARGS, serveRepo } from "./engine_synth.mjs";
import { writeSynth, SYNTH_MOE } from "./synth.mjs";
const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const PORT = +arg("port", 18983);

async function pageMain() {
  const { Qwen35Engine } = await import("/engine/qwen35.js");
  const { parseGGUFHeader, qwen35Weights, GGML_EMBED, dequantF32 } = await import("/engine/gguf.js");
  const { argmax } = await import("/engine/engine.js");
  const out = [], res = [];
  const check = (n, ok, d = "") => { res.push(ok); out.push(`${ok ? "PASS" : "FAIL"} ${n}${d ? "  " + d : ""}`); };
  const buf = await (await fetch("/__m.gguf")).arrayBuffer();
  const G = parseGGUFHeader(buf);
  const M = G.meta;
  check("qwen35moe keys aliased", M["general.architecture"] === "qwen35moe" && M["qwen35.expert_count"] > 0, `arch ${M["general.architecture"]}, ${M["qwen35.expert_count"]} experts, top-${M["qwen35.expert_used_count"]}`);
  const bytesOf = async (i) => new Uint8Array(buf, i.byteOffset, i.byteLength).slice();
  const W = (name) => { const i = G.tensors[name]; return dequantF32(i, new Uint8Array(buf, i.byteOffset, i.byteLength)); };
  const L = M["qwen35.block_count"] - 1;
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  const errs = []; device.addEventListener("uncapturederror", (e) => errs.push(e.error?.message));
  const eng = await Qwen35Engine.create({ device, meta: M, layerRange: [0, L], hasEmbed: true, hasHead: true, vocab: G.tensors[GGML_EMBED].shape[0],
    maxSeq: 512, batchCols: 16, coopRowsB: 1, coopWG: 64,
    weights: await qwen35Weights(G, bytesOf, { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true }) });
  const { dim } = eng.dims, { nExp, K, inter: ei, shInter } = eng.moe;
  const eps = M["qwen35.attention.layer_norm_rms_epsilon"];
  let seed = 7; const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
  const mv = (w, rows, cols, x, r0 = 0) => { const y = new Float64Array(rows); for (let r = 0; r < rows; r++) { let s = 0; for (let c = 0; c < cols; c++) s += w[(r0 + r) * cols + c] * x[c]; y[r] = s; } return y; };
  const silu = (v) => v / (1 + Math.exp(-v));
  for (const li of [0, 3, L]) {   // a DeltaNet layer, a full-attention layer, the draft block
    const p = `blk.${li}.`;
    const x = Float32Array.from({ length: dim }, () => rnd() * 2 - 1);
    const got = await eng.ffnOnly(li, x);
    const nw = W(p + "post_attention_norm.weight");
    let ss = 0; for (const v of x) ss += v * v;
    const inv = 1 / Math.sqrt(ss / dim + eps);
    const xn = Float64Array.from(x, (v, i) => v * inv * nw[i]);
    const lg = mv(W(p + "ffn_gate_inp.weight"), nExp, dim, xn);
    const m = Math.max(...lg), pr = Array.from(lg, (v) => Math.exp(v - m)), z = pr.reduce((a, b) => a + b, 0);
    const top = pr.map((v, i) => [v / z, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, K);
    const tot = top.reduce((a, b) => a + b[0], 0);
    const wg = W(p + "ffn_gate_exps.weight"), wu = W(p + "ffn_up_exps.weight"), wd = W(p + "ffn_down_exps.weight");
    const o = new Float64Array(dim);
    for (const [pw, e] of top) {
      const g = mv(wg, ei, dim, xn, e * ei), u = mv(wu, ei, dim, xn, e * ei);
      const h = Float64Array.from(g, (v, r) => silu(v) * u[r]);
      const y = mv(wd, dim, ei, h, e * dim);
      for (let i = 0; i < dim; i++) o[i] += (pw / tot) * y[i];
    }
    if (shInter) {
      const g = mv(W(p + "ffn_gate_shexp.weight"), shInter, dim, xn), u = mv(W(p + "ffn_up_shexp.weight"), shInter, dim, xn);
      const h = Float64Array.from(g, (v, r) => silu(v) * u[r]);
      const y = mv(W(p + "ffn_down_shexp.weight"), dim, shInter, h);
      const sgw = W(p + "ffn_gate_inp_shexp.weight"); let sg = 0; for (let i = 0; i < dim; i++) sg += sgw[i] * xn[i];
      const s = 1 / (1 + Math.exp(-sg));
      for (let i = 0; i < dim; i++) o[i] += s * y[i];
    }
    let maxErr = 0, maxRef = 0;
    for (let i = 0; i < dim; i++) { const ref = x[i] + o[i]; maxErr = Math.max(maxErr, Math.abs(ref - got[i])); maxRef = Math.max(maxRef, Math.abs(o[i])); }
    check(`layer ${li}${li === L ? " (draft block)" : ""} MoE FFN vs float64 reference`, maxErr / maxRef < 1e-3, `experts [${top.map((t) => t[1]).join(",")}], max err ${maxErr.toExponential(2)} (${(maxErr / maxRef).toExponential(1)} of the FFN output)`);
  }
  // decoding: plain greedy, then speculative must give exactly the same tokens
  eng.mtpFill = true;
  const prompt = Array.from({ length: 40 }, (_, i) => 33 + ((i * 7919) % 90));
  const start = async () => { eng.reset(); await eng.prefillTokens(prompt.slice(0, -1)); return argmax(await eng.forwardToken(prompt[prompt.length - 1])); };
  let t = await start(); const plain = [t];
  for (let i = 1; i < 24; i++) { t = argmax(await eng.forwardToken(t)); plain.push(t); }
  let next = await start(); const spec = [next];
  while (spec.length < 24) { const o2 = await eng.specStep(next, argmax, 3); spec.push(...o2); next = o2[o2.length - 1]; }
  check("MoE model: speculative decoding == plain (24 tokens, batched verify through the MoE)", spec.slice(0, 24).every((v, i) => v === plain[i]), `${eng.mtp.stats.accepted}/${eng.mtp.stats.drafts} drafts accepted`);
  // the batched MoE path with many columns accepted: perfect drafts (plain's own next tokens)
  let nx = await start(); const viaDrafts = [nx]; let verifies = 0;
  while (viaDrafts.length < 24) { const i = viaDrafts.length - 1; const o3 = await eng.specStepDrafts(nx, argmax, plain.slice(i + 1, i + 8)); viaDrafts.push(...o3); nx = o3[o3.length - 1]; verifies++; }
  check("MoE model: 7-token draft runs verified in batched passes == plain", viaDrafts.slice(0, 24).every((v, i) => v === plain[i]), `${verifies} verifies for 24 tokens`);
  // batched prefill (MoE through the multi-column kernels) == one token at a time, bit for bit
  eng.reset(); await eng.prefillTokens(prompt.slice(0, -1)); const lgA = await eng.forwardToken(prompt[prompt.length - 1]);
  eng.reset(); let lgB; for (const tk of prompt) lgB = await eng.forwardToken(tk);
  let nd = 0; for (let i = 0; i < lgA.length; i++) if (!Object.is(lgA[i], lgB[i])) nd++;
  check("MoE model: batched prefill == one token at a time (logits bit-identical)", nd === 0, `${nd} logits differ`);
  check("no GPU validation errors", !errs.length, errs.slice(0, 2).join(" | "));
  return { out, ok: res.every(Boolean) };
}
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "swarm-moe-"));
const model = arg("model") || writeSynth(path.join(tmp, "m.gguf"), { moe: SYNTH_MOE }).file;
const srv = serveRepo(PORT, { "/__m.gguf": path.resolve(model) });
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), args: GPU_ARGS });
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("[pageerror]", String(e).slice(0, 300)));
await page.goto(`http://127.0.0.1:${PORT}/favicon.svg`);
const r = await page.evaluate(pageMain);
for (const l of r.out) console.log(l);
console.log(r.ok ? "MOE PASS" : "MOE FAIL");
await browser.close(); srv.close(); fs.rmSync(tmp, { recursive: true, force: true });
process.exit(r.ok ? 0 : 1);
