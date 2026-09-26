// Draft-cache (MTP) fill during prefill: batched (_mtpFillBatch, one pass per chunk) vs per-column
// (mtpRun per column). Same rows expected (bit-identical below the GEMM width, within prefill
// tolerance at full width where the GEMM runs), same speculative output, and the prefill time.
//   NODE_PATH=... node tests/e2e/mtp_fill_synth.mjs [--model file.gguf] [--cols 4,16] [--prompt-len 120]
import fs from "fs"; import os from "os"; import path from "path";
import { loadPlaywright, chromiumPath, GPU_ARGS, serveRepo } from "./engine_synth.mjs";
import { writeSynth } from "./synth.mjs";
const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const PORT = +arg("port", 18990);

async function pageMain({ cols, plen, ntok }) {
  const { Qwen35Engine } = await import("/engine/qwen35.js");
  const { parseGGUFHeader, qwen35Weights, GGML_EMBED } = await import("/engine/gguf.js");
  const { argmax } = await import("/engine/engine.js");
  const out = [], say = (s) => { out.push(s); console.log(s); };
  const buf = await (await fetch("/__synth.gguf")).arrayBuffer();
  const G = parseGGUFHeader(buf);
  const bytesOf = async (i) => new Uint8Array(buf, i.byteOffset, i.byteLength).slice();
  const L = G.meta["qwen35.block_count"] - 1;
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  const errs = []; device.addEventListener("uncapturederror", (e) => errs.push(e.error?.message));
  const vocab = G.tensors[GGML_EMBED].shape[0];
  // a deterministic prompt of plen tokens drawn from the printable byte tokens
  const prompt = Array.from({ length: plen }, (_, i) => 33 + ((i * 7919) % 90));
  let allOk = true;
  for (const NC of cols) {
    const eng = await Qwen35Engine.create({ device, meta: G.meta, layerRange: [0, L], hasEmbed: true, hasHead: true, vocab, maxSeq: 512,
      batchCols: NC, coopRowsB: NC >= 16 ? 1 : 4, coopWG: 64, weights: await qwen35Weights(G, bytesOf, { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true }) });
    const kv = eng.dims.kvDim;
    const rows = async (n) => {
      const st = device.createBuffer({ size: n * kv * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(eng.mtpLayer.kCache, 0, st, 0, n * kv * 4); device.queue.submit([enc.finish()]);
      await st.mapAsync(GPUMapMode.READ); const a = new Float32Array(st.getMappedRange()).slice(); st.unmap(); st.destroy(); return a;
    };
    const run = async (batch) => {
      const enc = device.createCommandEncoder(); enc.clearBuffer(eng.mtpLayer.kCache); device.queue.submit([enc.finish()]);
      eng.reset(); eng.mtpFill = true; eng.mtpBatchFill = batch; eng.mtp.stats = { drafts: 0, accepted: 0 };
      await device.queue.onSubmittedWorkDone();
      const t0 = performance.now();
      await eng.prefillTokens(prompt.slice(0, -1));
      await device.queue.onSubmittedWorkDone();
      const ms = performance.now() - t0;
      const r = await rows(plen - 1);
      // speculative decode from here: output must not depend on the fill
      let next = argmax(await eng.forwardToken(prompt[plen - 1]));
      const toks = [next];
      while (toks.length < ntok) { const o = await eng.specStep(next, argmax, 3); toks.push(...o); next = o[o.length - 1]; }
      return { ms, r, toks: toks.slice(0, ntok), acc: `${eng.mtp.stats.accepted}/${eng.mtp.stats.drafts}` };
    };
    const a = await run(false), b = await run(true);
    let md = 0, sc = 1e-9, exact = true;
    for (let i = kv; i < a.r.length; i++) { const d = Math.abs(a.r[i] - b.r[i]); if (d) exact = false; md = Math.max(md, d); sc = Math.max(sc, Math.abs(a.r[i])); }
    const sameToks = a.toks.join() === b.toks.join();
    const rowsOk = exact || md / sc < 2e-3;
    if (!sameToks || !rowsOk) allOk = false;
    say(`${sameToks && rowsOk ? "PASS" : "FAIL"} [NC=${NC}, gemm ${eng.gemmOn}] draft rows 1..${plen - 2}: ${exact ? "bit-identical" : `max rel diff ${(md / sc).toExponential(2)}`}; spec output ${sameToks ? "identical" : "DIFFERS"}; acceptance per-column ${a.acc} vs batched ${b.acc}; prefill ${plen - 1} tok: per-column ${a.ms.toFixed(0)} ms, batched ${b.ms.toFixed(0)} ms`);
  }
  if (errs.length) { allOk = false; say("FAIL GPU errors: " + errs.slice(0, 2).join(" | ")); }
  return { out, allOk };
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "swarm-mtpfill-"));
const model = arg("model") || writeSynth(path.join(tmp, "m.gguf"), {}).file;
const srv = serveRepo(PORT, { "/__synth.gguf": path.resolve(model) });
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), args: GPU_ARGS });
const page = await browser.newPage();
page.on("console", (m) => { if (m.type() === "error") console.error("[page]", m.text()); });
await page.goto(`http://127.0.0.1:${PORT}/favicon.svg`);
const res = await page.evaluate(pageMain, { cols: arg("cols", "4,16").split(",").map(Number), plen: +arg("prompt-len", 120), ntok: +arg("tokens", 24) });
for (const l of res.out) console.log(l);
console.log(res.allOk ? "MTP FILL PASS" : "MTP FILL FAIL");
await browser.close(); srv.close(); fs.rmSync(tmp, { recursive: true, force: true });
process.exit(res.allOk ? 0 : 1);
