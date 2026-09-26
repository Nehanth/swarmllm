// Flash attention + f16 KV cache vs the f32 scores / softmax / out path: the same prompt and greedy
// decode on two engines, compared by logits (f16 K/V is not bit-identical, so a tolerance) and by
// the greedy tokens. --prompt-len above 2048 checks long context (the old path then uses the serial
// softmax). Also times decode on both (SwiftShader: not a GPU number).
//   NODE_PATH=... node tests/e2e/flash_synth.mjs [--model f.gguf] [--prompt-len 300] [--tokens 16] [--max-seq 2048] [--q8]
// --q8: the new engine keeps K/V in int8 (kvQ8) instead of f16.
import fs from "fs"; import os from "os"; import path from "path";
import { loadPlaywright, chromiumPath, GPU_ARGS, serveRepo } from "./engine_synth.mjs";
import { writeSynth } from "./synth.mjs";
const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const PORT = +arg("port", 18992);

async function pageMain({ plen, ntok, maxSeq, q8 }) {
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
  const mk = async (attnFlash, kvQ8 = false) => Qwen35Engine.create({ device, meta: G.meta, layerRange: [0, L], hasEmbed: true, hasHead: true, vocab: G.tensors[GGML_EMBED].shape[0],
    maxSeq, batchCols: 16, coopRowsB: 1, coopWG: 64, attnFlash, kvQ8,
    weights: await qwen35Weights(G, bytesOf, { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true }) });
  const prompt = Array.from({ length: plen }, (_, i) => 33 + ((i * 7919) % 90));
  const run = async (eng) => {
    eng.reset(); eng.mtpFill = false;
    const t0 = performance.now();
    await eng.prefillTokens(prompt.slice(0, -1));
    const tPre = performance.now() - t0;
    const logs = [], toks = [];
    let lg = await eng.forwardToken(prompt[plen - 1]); logs.push(lg); toks.push(argmax(lg));
    await device.queue.onSubmittedWorkDone();
    const t1 = performance.now();
    for (let i = 1; i < ntok; i++) { lg = await eng.forwardToken(toks[i - 1]); logs.push(lg); toks.push(argmax(lg)); }
    return { logs, toks, tPre, msPerTok: (performance.now() - t1) / Math.max(1, ntok - 1) };
  };
  const A = await mk(false); const a = await run(A);
  const B = await mk(true, q8); const b = await run(B);
  say(`engine.flash: old ${A.flash}, new ${B.flash}${B.kvQ8 ? " with int8 KV" : ""} (split ${B.faSplit} positions, ${B.faSplits} splits)`);
  // compare while both follow the same tokens: stop at the first greedy divergence
  let agree = 0; while (agree < ntok && a.toks[agree] === b.toks[agree]) agree++;
  let maxd = 0, maxr = 0;
  for (let i = 0; i < Math.min(ntok, agree + 1); i++) {
    let r = 0; for (const x of a.logs[i]) r = Math.max(r, Math.abs(x));
    for (let j = 0; j < a.logs[i].length; j++) maxd = Math.max(maxd, Math.abs(a.logs[i][j] - b.logs[i][j]));
    maxr = Math.max(maxr, r);
  }
  const rel = maxd / maxr;
  const ok = rel < 2e-2 && agree >= Math.min(ntok, 4) && !errs.length && b.toks.every(Number.isInteger);
  say(`${ok ? "PASS" : "FAIL"} ${plen}-token prompt, ${ntok} greedy tokens: ${agree}/${ntok} identical before any divergence; max |logit diff| ${maxd.toExponential(2)} (${rel.toExponential(1)} of max |logit|)`);
  say(`prefill ${a.tPre.toFixed(0)} vs ${b.tPre.toFixed(0)} ms; decode ${a.msPerTok.toFixed(0)} vs ${b.msPerTok.toFixed(0)} ms/token (old vs flash, SwiftShader)`);
  if (errs.length) say("GPU errors: " + errs.slice(0, 2).join(" | "));
  return { out, ok };
}
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "swarm-fa-"));
const model = arg("model") || writeSynth(path.join(tmp, "m.gguf"), {}).file;
const srv = serveRepo(PORT, { "/__m.gguf": path.resolve(model) });
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), args: GPU_ARGS });
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${PORT}/favicon.svg`);
const res = await page.evaluate(pageMain, { plen: +arg("prompt-len", 300), ntok: +arg("tokens", 16), maxSeq: +arg("max-seq", 2048), q8: argv.includes("--q8") });
for (const l of res.out) console.log(l);
console.log(res.ok ? "FLASH PASS" : "FLASH FAIL");
await browser.close(); srv.close(); fs.rmSync(tmp, { recursive: true, force: true });
process.exit(res.ok ? 0 : 1);
