// Long draft runs (prompt lookup up to engine.maxDrafts = 15 tokens with replay rollback): verify
// runs of perfect drafts, drafts wrong at a given position, and all-wrong drafts, with 4 and 16
// batch columns (4 columns: a 16-token verify is 4 chunks). The stream must equal plain greedy
// decoding token for token, and the state after it must be the same (the logits that follow).
//   NODE_PATH=... node tests/e2e/drafts_synth.mjs [--model f.gguf] [--tokens 48]
import fs from "fs"; import os from "os"; import path from "path";
import { loadPlaywright, chromiumPath, GPU_ARGS, serveRepo } from "./engine_synth.mjs";
import { writeSynth } from "./synth.mjs";
const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const PORT = +arg("port", 18988);

async function pageMain({ N }) {
  const { Qwen35Engine } = await import("/engine/qwen35.js");
  const { parseGGUFHeader, qwen35Weights, GGML_EMBED } = await import("/engine/gguf.js");
  const { argmax } = await import("/engine/engine.js");
  const out = [], say = (s) => out.push(s);
  const buf = await (await fetch("/__m.gguf")).arrayBuffer();
  const G = parseGGUFHeader(buf);
  const bytesOf = async (i) => new Uint8Array(buf, i.byteOffset, i.byteLength).slice();
  const L = G.meta["qwen35.block_count"] - 1;
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  const errs = []; device.addEventListener("uncapturederror", (e) => errs.push(e.error?.message));
  const prompt = Array.from({ length: 30 }, (_, i) => 33 + ((i * 7919) % 90));
  let ok = !errs.length;
  for (const NC of [4, 16]) {
    const eng = await Qwen35Engine.create({ device, meta: G.meta, layerRange: [0, L], hasEmbed: true, hasHead: true, vocab: G.tensors[GGML_EMBED].shape[0],
      maxSeq: 512, batchCols: NC, coopRowsB: NC >= 16 ? 1 : 4, coopWG: 64,
      weights: await qwen35Weights(G, bytesOf, { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true }) });   // uploads consume the weights: one set per engine
    eng.mtpFill = false;
    const begin = async () => { eng.reset(); await eng.prefillTokens(prompt.slice(0, -1)); return argmax(await eng.forwardToken(prompt[prompt.length - 1])); };
    // plain reference
    let t = await begin(); const ref = [t];
    for (let i = 1; i < N + 20; i++) { t = argmax(await eng.forwardToken(t)); ref.push(t); }
    for (const [name, mk] of [
      ["perfect 15-token runs", (i, k) => ref.slice(i + 1, i + 1 + k)],
      ["wrong at draft 9", (i, k) => ref.slice(i + 1, i + 1 + k).map((x, j) => (j === 8 ? (x + 1) % 64 : x))],
      ["wrong at draft 1", (i, k) => ref.slice(i + 1, i + 1 + k).map((x, j) => (j === 0 ? (x + 3) % 64 : x))],
    ]) {
      let next = await begin(); const got = [next]; let steps = 0;
      while (got.length < N) {
        const i = got.length - 1;
        const o = await eng.specStepDrafts(next, argmax, mk(i, eng.maxDrafts));
        got.push(...o); next = o[o.length - 1]; steps++;
      }
      const same = got.slice(0, N).every((x, j) => x === ref[j]);
      ok = ok && same;
      say(`${same ? "PASS" : "FAIL"} NC=${NC} maxDrafts=${eng.maxDrafts} ${name}: ${N} tokens in ${steps} verifies == plain`);
    }
  }
  if (errs.length) say("GPU errors: " + errs.slice(0, 2).join(" | "));
  return { out, ok: ok && !errs.length };
}
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "swarm-dr-"));
const model = arg("model") || writeSynth(path.join(tmp, "m.gguf"), {}).file;
const srv = serveRepo(PORT, { "/__m.gguf": path.resolve(model) });
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), args: GPU_ARGS });
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${PORT}/favicon.svg`);
const res = await page.evaluate(pageMain, { N: +arg("tokens", 48) });
for (const l of res.out) console.log(l);
console.log(res.ok ? "DRAFTS PASS" : "DRAFTS FAIL");
await browser.close(); srv.close(); fs.rmSync(tmp, { recursive: true, force: true });
process.exit(res.ok ? 0 : 1);
