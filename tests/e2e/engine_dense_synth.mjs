// DenseEngine (engine/dense.js) on the synthetic Qwen3 dense model (tests/e2e/synth_dense.mjs), in
// headless Chromium + WebGPU (SwiftShader is fine), no room: for every cooperative-GEMV config the
// autotune can pick (engine/autotune.js: WG 256|128|64 x ROWS 4|8) it checks that the batched
// 4-column paths the room uses (prefillTokens solo, embedRunBatch on the host and runHiddenBatch on
// a worker in split mode) give the same hidden states as the one-token path (embedRun / runHidden).
//
//   NODE_PATH=<node_modules> node tests/e2e/engine_dense_synth.mjs [--port 18920] [--configs 64x4,64x8,256x4,256x8]
//        [--dense-js patched/dense.js]
import fs from "fs";
import os from "os";
import path from "path";
import { writeDense } from "./synth_dense.mjs";
import { loadPlaywright, chromiumPath, GPU_ARGS, serveRepo } from "./engine_synth.mjs";

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const PORT = +arg("port", 18920);
const CONFIGS = arg("configs", "64x4,64x8,256x4,256x8").split(",").map((s) => s.split("x").map(Number));

async function pageMain({ configs }) {
  const { DenseEngine } = await import("/engine/dense.js");
  const { parseGGUFHeader, ggufWeights } = await import("/engine/gguf.js");
  const buf = await (await fetch("/__model.gguf")).arrayBuffer();
  const cfg = await (await fetch("/__config.json")).json();
  const G = parseGGUFHeader(buf);
  const bytesOf = async (i) => new Uint8Array(buf, i.byteOffset, i.byteLength).slice();
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const errs = [];
  device.addEventListener("uncapturederror", (e) => errs.push(e.error?.message));
  const L = cfg.num_hidden_layers, dim = cfg.hidden_size, half = L >> 1;
  const ids = [300, 17, 42, 99, 7, 256, 311, 12];
  const maxAbs = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
  const out = [];
  for (const [wg, rows] of configs) {
    const mk = async (lo, hi, hasEmbed, hasHead) => DenseEngine.create({ device, cfg, weights: await ggufWeights(G, bytesOf, { lo, hi, hasEmbed, hasHead }),
      layerRange: [lo, hi], hasEmbed, hasHead, maxSeq: 64, coopWG: wg, coopRows: rows });
    const host = await mk(0, half, true, true), work = await mk(half, L, false, false);
    // reference: one token at a time through host layers then worker layers
    const ref = [];
    for (let p = 0; p < ids.length; p++) ref.push(await work.runHidden(await host.embedRun(ids[p], p), p));
    host.reset(); work.reset();
    // split-mode batched prefill (room.js aiPrefill / workerFrame), 4 columns per pass
    let dBatch = 0;
    for (let p = 0; p < ids.length; p += 4) {
      const hb = await host.embedRunBatch(ids.slice(p, p + 4), p);
      const wb = await work.runHiddenBatch(hb, p);
      for (let c = 0; c < 4; c++) dBatch = Math.max(dBatch, maxAbs(wb.subarray(c * dim, (c + 1) * dim), ref[p + c]));
    }
    // solo batched prefill (prefillTokens) on a full engine, then the last token one at a time
    const solo = await mk(0, L, true, true);
    await solo.prefillTokens(ids.slice(0, -1));
    const last = await solo.embedRun(ids[ids.length - 1], ids.length - 1);
    const soloRef = [];
    solo.reset();
    for (let p = 0; p < ids.length; p++) soloRef.push(await solo.embedRun(ids[p], p));
    const dSolo = maxAbs(last, soloRef[ids.length - 1]);
    out.push({ wg, rows, splitBatchVsSingle: +dBatch.toExponential(2), soloPrefillVsSingle: +dSolo.toExponential(2) });
  }
  return { out, errs };
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "swarm-dense-engine-"));
const d = writeDense(tmp, { seed: 5 });
// --dense-js FILE: serve FILE as /engine/dense.js (try a patch without touching the tree)
const srv = serveRepo(PORT, { "/__model.gguf": d.files.gguf, "/__config.json": d.files.cfg, ...(arg("dense-js") ? { "/engine/dense.js": path.resolve(arg("dense-js")) } : {}) });
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), args: GPU_ARGS });
let code = 0;
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${PORT}/__blank.html`);
  const r = await page.evaluate(pageMain, { configs: CONFIGS });
  for (const x of r.out) {
    const ok = x.splitBatchVsSingle < 1e-2 && x.soloPrefillVsSingle < 1e-2;
    if (!ok) code = 1;
    console.log(`${ok ? "PASS" : "FAIL"} WG=${x.wg} ROWS=${x.rows}: max |batched - one-token| hidden: split host+worker ${x.splitBatchVsSingle}, solo prefillTokens ${x.soloPrefillVsSingle}`);
  }
  if (r.errs.length) { console.log("GPU errors:", r.errs.slice(0, 3)); code = 1; }
} finally {
  await browser.close(); srv.close(); fs.rmSync(tmp, { recursive: true, force: true });
}
console.log(code ? "DENSE ENGINE FAIL" : "DENSE ENGINE PASS");
process.exit(code);
