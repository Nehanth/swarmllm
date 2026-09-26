// attn_softmax_wg vs attn_softmax: every logit of a long prefill (batched path) and a decode
// (single-token path) must be bit-identical; also times decode at that context length.
//   NODE_PATH=... node tests/e2e/softmax_synth.mjs [--model f.gguf] [--prompt-len 400] [--tokens 24]
import fs from "fs"; import os from "os"; import path from "path";
import { loadPlaywright, chromiumPath, GPU_ARGS, serveRepo } from "./engine_synth.mjs";
import { writeSynth } from "./synth.mjs";
const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const PORT = +arg("port", 18997);

async function pageMain({ plen, ntok }) {
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
  const eng = await Qwen35Engine.create({ device, meta: G.meta, layerRange: [0, L], hasEmbed: true, hasHead: true, vocab: G.tensors[GGML_EMBED].shape[0],
    maxSeq: 2048, batchCols: 16, coopRowsB: 1, coopWG: 64, weights: await qwen35Weights(G, bytesOf, { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true }) });
  const prompt = Array.from({ length: plen }, (_, i) => 33 + ((i * 7919) % 90));
  const run = async (wg) => {
    eng.softmaxWG = wg; eng.reset(); eng.mtpFill = false;
    await eng.prefillTokens(prompt.slice(0, -1));
    const logs = [];
    let lg = await eng.forwardToken(prompt[plen - 1]); logs.push(lg);
    await device.queue.onSubmittedWorkDone();
    const t0 = performance.now();
    for (let i = 1; i < ntok; i++) { lg = await eng.forwardToken(argmax(lg)); logs.push(lg); }
    return { logs, msPerTok: (performance.now() - t0) / (ntok - 1) };
  };
  const a = await run(false), b = await run(true);
  let diff = 0;
  for (let i = 0; i < ntok; i++) for (let j = 0; j < a.logs[i].length; j++) if (!Object.is(a.logs[i][j], b.logs[i][j])) diff++;
  const ok = diff === 0 && !errs.length;
  say(`${ok ? "PASS" : "FAIL"} ${ntok} decode steps after a ${plen}-token prompt: ${diff} logits differ between attn_softmax and attn_softmax_wg`);
  say(`decode at pos ~${plen} (SwiftShader, not a GPU number): serial softmax ${a.msPerTok.toFixed(0)} ms/token, workgroup softmax ${b.msPerTok.toFixed(0)} ms/token`);
  if (errs.length) say("GPU errors: " + errs.slice(0, 2).join(" | "));
  return { out, ok };
}
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "swarm-sm-"));
const model = arg("model") || writeSynth(path.join(tmp, "m.gguf"), {}).file;
const srv = serveRepo(PORT, { "/__m.gguf": path.resolve(model) });
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), args: GPU_ARGS });
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${PORT}/favicon.svg`);
const res = await page.evaluate(pageMain, { plen: +arg("prompt-len", 400), ntok: +arg("tokens", 24) });
for (const l of res.out) console.log(l);
console.log(res.ok ? "SOFTMAX PASS" : "SOFTMAX FAIL");
await browser.close(); srv.close(); fs.rmSync(tmp, { recursive: true, force: true });
process.exit(res.ok ? 0 : 1);
