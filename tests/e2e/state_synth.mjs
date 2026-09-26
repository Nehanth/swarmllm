// Session state: exportState/importState, GPU slots and the OPFS state cache must resume a
// conversation exactly (bit-identical logits) after the engine was reset and dirtied with another
// prompt; speculative decoding after a restore must equal plain decoding.
//   NODE_PATH=... node tests/e2e/state_synth.mjs [--model f.gguf] [--prompt-len 60] [--tokens 12] [--flash 0|1] [--q8]
import fs from "fs"; import os from "os"; import path from "path";
import { loadPlaywright, chromiumPath, GPU_ARGS, serveRepo } from "./engine_synth.mjs";
import { writeSynth } from "./synth.mjs";
const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const PORT = +arg("port", 18990);

async function pageMain({ plen, ntok, flash, q8 }) {
  const { Qwen35Engine } = await import("/engine/qwen35.js");
  const { parseGGUFHeader, qwen35Weights, GGML_EMBED } = await import("/engine/gguf.js");
  const { argmax } = await import("/engine/engine.js");
  const { StateCache, tokenKey } = await import("/harness/statecache.js");
  const out = [], say = (s) => out.push(s);
  const buf = await (await fetch("/__m.gguf")).arrayBuffer();
  const G = parseGGUFHeader(buf);
  const bytesOf = async (i) => new Uint8Array(buf, i.byteOffset, i.byteLength).slice();
  const L = G.meta["qwen35.block_count"] - 1;
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  const errs = []; device.addEventListener("uncapturederror", (e) => errs.push(e.error?.message));
  const eng = await Qwen35Engine.create({ device, meta: G.meta, layerRange: [0, L], hasEmbed: true, hasHead: true, vocab: G.tensors[GGML_EMBED].shape[0],
    maxSeq: 1024, batchCols: 16, coopRowsB: 1, coopWG: 64, attnFlash: flash, kvQ8: q8,
    weights: await qwen35Weights(G, bytesOf, { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true }) });
  const results = [];
  const check = (name, ok, detail = "") => { results.push(ok); say(`${ok ? "PASS" : "FAIL"} ${name}${detail ? "  " + detail : ""}`); };
  const prompt = Array.from({ length: plen }, (_, i) => 33 + ((i * 7919) % 90));
  const junk = Array.from({ length: 37 }, (_, i) => 40 + ((i * 131) % 70));
  const decode = async (first, n) => {   // greedy from token `first`; returns tokens and logits
    const toks = [], logs = [];
    let t = first;
    for (let i = 0; i < n; i++) { const lg = await eng.forwardToken(t); logs.push(lg); t = argmax(lg); toks.push(t); }
    return { toks, logs };
  };
  const same = (a, b) => a.logs.length === b.logs.length && a.logs.every((l, i) => l.length === b.logs[i].length && l.every((x, j) => Object.is(x, b.logs[i][j])));
  const dirty = async () => { eng.reset(); await eng.prefillTokens(junk); await eng.forwardToken(7); };

  eng.reset(); eng.mtpFill = false;
  await eng.prefillTokens(prompt.slice(0, -1));
  const t0 = argmax(await eng.forwardToken(prompt[plen - 1]));
  const st = await eng.exportState();
  eng.saveSlot("turn1");
  const bytes = st.parts.reduce((s, p) => s + p.byteLength, 0);
  const ref = await decode(t0, ntok);

  await dirty(); eng.importState(st);
  const a = await decode(t0, ntok);
  check("exportState -> reset + other prompt -> importState resumes exactly", same(ref, a), `pos ${st.pos}, ${(bytes / 1e6).toFixed(2)} MB`);

  await dirty(); eng.loadSlot("turn1");
  const b = await decode(t0, ntok);
  check("GPU slot save/load resumes exactly", same(ref, b));

  await dirty(); eng.loadSlot("turn1");
  const spec = []; let tn = t0;
  while (spec.length < ntok) { const o = await eng.specStep(tn, argmax, 3); spec.push(...o); tn = o[o.length - 1]; }
  check("spec decoding after a restore == plain", JSON.stringify(spec.slice(0, ntok)) === JSON.stringify(ref.toks), `${eng.mtp.stats.accepted}/${eng.mtp.stats.drafts} drafts accepted`);

  let bad = false; try { eng.importState({ ...st, sig: { ...st.sig, lo: 99 } }); } catch { bad = true; }
  check("a state for other layers is refused", bad);

  const cache = new StateCache({ dirName: "test-states", budgetBytes: 1e9 });
  await cache.clear();
  const key = await tokenKey(eng.stateSignature(), prompt, "synth");
  const tW = performance.now(); await cache.put(key, st, { n: prompt.length }); const tw = performance.now() - tW;
  await dirty();
  const tR = performance.now(); const got = await cache.get(key); const tr = performance.now() - tR;
  eng.importState(got);
  const c = await decode(t0, ntok);
  check("OPFS state cache round trip resumes exactly", !!got && same(ref, c), `write ${tw.toFixed(0)} ms, read ${tr.toFixed(0)} ms for ${(bytes / 1e6).toFixed(2)} MB`);
  check("a different prompt has a different key", (await tokenKey(eng.stateSignature(), prompt.slice(1), "synth")) !== key && !(await cache.has("0".repeat(32))));
  await cache.clear();
  check("no GPU validation errors", !errs.length, errs.slice(0, 2).join(" | "));
  return { out, ok: results.every(Boolean) };
}
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "swarm-st-"));
const model = arg("model") || writeSynth(path.join(tmp, "m.gguf"), {}).file;
const srv = serveRepo(PORT, { "/__m.gguf": path.resolve(model) });
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), args: GPU_ARGS });
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${PORT}/favicon.svg`);
const res = await page.evaluate(pageMain, { plen: +arg("prompt-len", 60), ntok: +arg("tokens", 12), flash: arg("flash", "1") !== "0", q8: argv.includes("--q8") });
for (const l of res.out) console.log(l);
console.log(res.ok ? "STATE PASS" : "STATE FAIL");
await browser.close(); srv.close(); fs.rmSync(tmp, { recursive: true, force: true });
process.exit(res.ok ? 0 : 1);
