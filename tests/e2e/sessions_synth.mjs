// harness/sessions.js: three sessions interleaved on one engine with one spare GPU slot, so one
// of them is spilled to OPFS and brought back; every session's logits must equal those of the
// same session decoded alone without interruption (bit-identical).
//   NODE_PATH=... node tests/e2e/sessions_synth.mjs [--model f.gguf] [--tokens 6]
import fs from "fs"; import os from "os"; import path from "path";
import { loadPlaywright, chromiumPath, GPU_ARGS, serveRepo } from "./engine_synth.mjs";
import { writeSynth } from "./synth.mjs";
const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const PORT = +arg("port", 18989);

async function pageMain({ n }) {
  const { Qwen35Engine } = await import("/engine/qwen35.js");
  const { parseGGUFHeader, qwen35Weights, GGML_EMBED } = await import("/engine/gguf.js");
  const { argmax } = await import("/engine/engine.js");
  const { Sessions } = await import("/harness/sessions.js");
  const { StateCache } = await import("/harness/statecache.js");
  const out = [], say = (s) => out.push(s);
  const buf = await (await fetch("/__m.gguf")).arrayBuffer();
  const G = parseGGUFHeader(buf);
  const bytesOf = async (i) => new Uint8Array(buf, i.byteOffset, i.byteLength).slice();
  const L = G.meta["qwen35.block_count"] - 1;
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  const errs = []; device.addEventListener("uncapturederror", (e) => errs.push(e.error?.message));
  const eng = await Qwen35Engine.create({ device, meta: G.meta, layerRange: [0, L], hasEmbed: true, hasHead: true, vocab: G.tensors[GGML_EMBED].shape[0],
    maxSeq: 512, batchCols: 16, coopRowsB: 1, coopWG: 64,
    weights: await qwen35Weights(G, bytesOf, { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true }) });
  eng.mtpFill = false;
  const prompts = { A: [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53], B: [90, 17, 33, 71, 72, 73, 60, 61, 62], C: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] };
  const start = async (ids) => { await eng.prefillTokens(ids.slice(0, -1)); return argmax(await eng.forwardToken(ids[ids.length - 1])); };
  const decode = async (t, k) => { const logs = []; for (let i = 0; i < k; i++) { const lg = await eng.forwardToken(t); logs.push(lg); t = argmax(lg); } return { logs, next: t }; };
  const ref = {};
  for (const [id, ids] of Object.entries(prompts)) { eng.reset(); ref[id] = (await decode(await start(ids), 2 * n)).logs; }

  const cache = new StateCache({ dirName: "test-sessions", budgetBytes: 1e9 }); await cache.clear();
  const S = new Sessions(eng, { gpuSlots: 1, cache, prefix: "t" });
  const got = { A: [], B: [], C: [] }, next = {}, froms = [];
  for (const id of ["A", "B", "C"]) {
    froms.push(id + ":" + await S.switchTo(id));
    const r = await decode(await start(prompts[id]), n); got[id].push(...r.logs); next[id] = r.next;
  }
  for (const id of ["A", "C", "B"]) {
    froms.push(id + ":" + await S.switchTo(id));
    const r = await decode(next[id], n); got[id].push(...r.logs); next[id] = r.next;
  }
  let ok = !errs.length;
  for (const id of Object.keys(prompts)) {
    const same = got[id].length === ref[id].length && got[id].every((l, i) => l.every((x, j) => Object.is(x, ref[id][i][j])));
    ok = ok && same;
    say(`${same ? "PASS" : "FAIL"} session ${id}: ${2 * n} tokens across switches == uninterrupted`);
  }
  const st = S.stats;
  const both = st.diskHits >= 1 && st.gpuHits >= 1;
  ok = ok && both;
  say(`${both ? "PASS" : "FAIL"} switches ${froms.join(" ")}; gpu ${st.gpuHits}, disk ${st.diskHits}, new ${st.fresh}, spilled ${st.spills}`);
  await cache.clear();
  if (errs.length) say("GPU errors: " + errs.slice(0, 2).join(" | "));
  return { out, ok };
}
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "swarm-ss-"));
const model = arg("model") || writeSynth(path.join(tmp, "m.gguf"), {}).file;
const srv = serveRepo(PORT, { "/__m.gguf": path.resolve(model) });
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), args: GPU_ARGS });
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${PORT}/favicon.svg`);
const res = await page.evaluate(pageMain, { n: +arg("tokens", 6) });
for (const l of res.out) console.log(l);
console.log(res.ok ? "SESSIONS PASS" : "SESSIONS FAIL");
await browser.close(); srv.close(); fs.rmSync(tmp, { recursive: true, force: true });
process.exit(res.ok ? 0 : 1);
