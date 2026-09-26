// One-submit draft chain (engine draftChain) vs one submit per draft, each with and without the
// one-submit verify (engine specFuse): same drafts (so the same acceptance counts) and the same
// speculative output as the fully separate path, K = 3 and 7, with and without draftVocab.
//   NODE_PATH=... node tests/e2e/draftchain_synth.mjs [--model f.gguf]
import fs from "fs"; import os from "os"; import path from "path";
import { loadPlaywright, chromiumPath, GPU_ARGS, serveRepo } from "./engine_synth.mjs";
import { writeSynth } from "./synth.mjs";
const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const PORT = +arg("port", 18999);

async function pageMain({ ntok }) {
  const { Qwen35Engine } = await import("/engine/qwen35.js");
  const { parseGGUFHeader, qwen35Weights, GGML_EMBED, tokenizerFromGGUF } = await import("/engine/gguf.js");
  const { argmax, makeTokenizer } = await import("/engine/engine.js");
  const out = [], say = (s) => out.push(s);
  const buf = await (await fetch("/__m.gguf")).arrayBuffer();
  const G = parseGGUFHeader(buf);
  const tok = makeTokenizer(tokenizerFromGGUF(G.meta)), V = tok.vocab;
  const bytesOf = async (i) => new Uint8Array(buf, i.byteOffset, i.byteLength).slice();
  const L = G.meta["qwen35.block_count"] - 1;
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  const errs = []; device.addEventListener("uncapturederror", (e) => errs.push(e.error?.message));
  const prompt = [V["<|im_start|>"], ...tok.encode("user\nWrite three sentences about the ocean."), V["<|im_end|>"], ...tok.encode("\n"), V["<|im_start|>"], ...tok.encode("assistant\n")];
  let allOk = true;
  for (const draftVocab of [0, 200]) {
    const eng = await Qwen35Engine.create({ device, meta: G.meta, layerRange: [0, L], hasEmbed: true, hasHead: true, vocab: G.tensors[GGML_EMBED].shape[0],
      maxSeq: 512, batchCols: 16, coopRowsB: 1, coopWG: 64, draftChain: true, draftVocab,
      weights: await qwen35Weights(G, bytesOf, { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true }) });
    if (!eng.draftChain) { say("FAIL draftChain did not switch on"); allOk = false; continue; }
    for (const K of [3, 7]) {
      const run = async (chain, fuse) => {
        eng.chainOn = chain; eng.specFuse = fuse; eng.reset(); eng.mtp.stats = { drafts: 0, accepted: 0 };
        await eng.prefillTokens(prompt.slice(0, -1));
        let next = argmax(await eng.forwardToken(prompt[prompt.length - 1]));
        const toks = [next]; const t0 = performance.now(); let steps = 0;
        while (toks.length < ntok) { const o = await eng.specStep(next, argmax, K); toks.push(...o); next = o[o.length - 1]; steps++; }
        return { toks: toks.slice(0, ntok).join(), st: `${eng.mtp.stats.accepted}/${eng.mtp.stats.drafts}`, ms: (performance.now() - t0) / steps };
      };
      const a = await run(false, false);
      for (const [chain, fuse] of [[true, false], [false, true], [true, true]]) {
        const b = await run(chain, fuse);
        const ok = a.toks === b.toks && a.st === b.st;
        if (!ok) allOk = false;
        say(`${ok ? "PASS" : "FAIL"} draftVocab ${draftVocab || "off"}, K=${K}, chain ${chain ? "on" : "off"}, fuse ${fuse ? "on" : "off"}: output ${a.toks === b.toks ? "identical" : "DIFFERS"}, drafts accepted separate ${a.st} vs ${b.st}; ms per step (SwiftShader) ${a.ms.toFixed(0)} vs ${b.ms.toFixed(0)}`);
      }
    }
  }
  if (errs.length) { allOk = false; say("GPU errors: " + errs.slice(0, 2).join(" | ")); }
  return { out, allOk };
}
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "swarm-dc-"));
const model = arg("model") || writeSynth(path.join(tmp, "m.gguf"), {}).file;
const srv = serveRepo(PORT, { "/__m.gguf": path.resolve(model) });
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), args: GPU_ARGS });
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${PORT}/favicon.svg`);
const res = await page.evaluate(pageMain, { ntok: +arg("tokens", 40) });
for (const l of res.out) console.log(l);
console.log(res.allOk ? "DRAFTCHAIN PASS" : "DRAFTCHAIN FAIL");
await browser.close(); srv.close(); fs.rmSync(tmp, { recursive: true, force: true });
process.exit(res.allOk ? 0 : 1);
