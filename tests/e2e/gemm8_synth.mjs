// Q8_0 prefill GEMM vs the GEMV it replaces: 16-column batched passes through every layer of a
// 27B-shaped synthetic model (tests/e2e/synth.mjs --shape 27b, with the real file's Q8_0 tensors),
// hidden states compared GEMM-q8 vs GEMV-q8 (engine.gemm8 = false) vs all-GEMV (engine.gemm = false).
//   NODE_PATH=... node tests/e2e/gemm8_synth.mjs --model synth27q8.gguf
import path from "path";
import { loadPlaywright, chromiumPath, GPU_ARGS, serveRepo } from "./engine_synth.mjs";
const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const PORT = +arg("port", 18995);

async function pageMain({ rounds }) {
  const { Qwen35Engine } = await import("/engine/qwen35.js");
  const { parseGGUFHeader, qwen35Weights, GGML_EMBED } = await import("/engine/gguf.js");
  const out = [], say = (s) => out.push(s);
  const buf = await (await fetch("/__m.gguf")).arrayBuffer();
  const G = parseGGUFHeader(buf);
  const bytesOf = async (i) => new Uint8Array(buf, i.byteOffset, i.byteLength).slice();
  const L = G.meta["qwen35.block_count"] - 1;
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  const errs = []; device.addEventListener("uncapturederror", (e) => errs.push(e.error?.message));
  const eng = await Qwen35Engine.create({ device, meta: G.meta, layerRange: [0, L], hasEmbed: true, hasHead: true, vocab: G.tensors[GGML_EMBED].shape[0],
    maxSeq: 256, batchCols: 16, coopRowsB: 1, coopWG: 64, weights: await qwen35Weights(G, bytesOf, { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true }) });
  say(`gemm on: ${eng.gemmOn}; q4 pairs ${JSON.stringify(eng._gemmPairs)}; q8 pairs ${JSON.stringify(eng._gemm8Pairs)}`);
  const ids = Array.from({ length: 16 * rounds }, (_, i) => 33 + ((i * 7919) % 90));
  const pass = async (mode) => {
    eng.reset(); eng.gemm = mode !== "gemv"; eng.gemm8 = mode === "gemm8" ? true : false;
    const hs = [];
    const t0 = performance.now();
    for (let r = 0; r < rounds; r++) hs.push(await eng.embedRunBatch(ids.slice(r * 16, r * 16 + 16), r * 16));
    return { hs, ms: performance.now() - t0 };
  };
  const g8 = await pass("gemm8"), g4 = await pass("gemm4"), gv = await pass("gemv");
  const rel = (a, b) => { let n = 0, d = 0, bad = 0; for (let r = 0; r < a.length; r++) for (let i = 0; i < a[r].length; i++) { const x = a[r][i], y = b[r][i]; if (!Number.isFinite(x)) bad++; d += (x - y) ** 2; n += y * y; } return { rel: Math.sqrt(d / Math.max(n, 1e-30)), bad }; };
  const a = rel(g8.hs, gv.hs), b = rel(g4.hs, gv.hs), c = rel(g8.hs, g4.hs);
  const ok = a.bad === 0 && a.rel < 2e-3 && !errs.length;
  say(`${ok ? "PASS" : "FAIL"} ${rounds} x 16-column passes: rel L2 vs all-GEMV: q8 GEMM ${a.rel.toExponential(2)}, q4-only GEMM ${b.rel.toExponential(2)}; q8 GEMM vs q4-only ${c.rel.toExponential(2)}; non-finite ${a.bad}`);
  say(`time (SwiftShader, not a GPU number): q8 GEMM ${g8.ms.toFixed(0)} ms, q4-only GEMM ${g4.ms.toFixed(0)} ms, all-GEMV ${gv.ms.toFixed(0)} ms`);
  if (errs.length) say("GPU errors: " + errs.slice(0, 2).join(" | "));
  return { out, ok };
}
const srv = serveRepo(PORT, { "/__m.gguf": path.resolve(arg("model")) });
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), args: GPU_ARGS });
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${PORT}/favicon.svg`);
const res = await page.evaluate(pageMain, { rounds: +arg("rounds", 2) });
for (const l of res.out) console.log(l);
console.log(res.ok ? "GEMM8 PASS" : "GEMM8 FAIL");
await browser.close(); srv.close();
process.exit(res.ok ? 0 : 1);
