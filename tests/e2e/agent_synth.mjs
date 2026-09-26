// harness/engine-model.js on a real engine + tokenizer (synthetic model): a follow-up turn reuses
// the prefix the engine holds and gives exactly what a fresh engine gives; speculative decoding
// gives exactly plain decoding's text; the agent loop runs end to end on it.
//   NODE_PATH=... node tests/e2e/agent_synth.mjs [--model f.gguf]
import fs from "fs"; import os from "os"; import path from "path";
import { loadPlaywright, chromiumPath, GPU_ARGS, serveRepo } from "./engine_synth.mjs";
import { writeSynth } from "./synth.mjs";
const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf("--" + k); return i >= 0 ? argv[i + 1] : d; };
const PORT = +arg("port", 18986);

async function pageMain() {
  const { Qwen35Engine } = await import("/engine/qwen35.js");
  const { parseGGUFHeader, qwen35Weights, GGML_EMBED, tokenizerFromGGUF } = await import("/engine/gguf.js");
  const { makeTokenizer } = await import("/engine/tokenizer.js");
  const { engineModel } = await import("/harness/engine-model.js");
  const { Agent } = await import("/harness/agent.js");
  const { codingTools } = await import("/harness/codetools.js");
  const { MemoryWorkspace } = await import("/harness/workspace.js");
  const out = [], say = (s) => out.push(s);
  const buf = await (await fetch("/__m.gguf")).arrayBuffer();
  const G = parseGGUFHeader(buf);
  const tok = makeTokenizer(tokenizerFromGGUF(G.meta));
  const bytesOf = async (i) => new Uint8Array(buf, i.byteOffset, i.byteLength).slice();
  const L = G.meta["qwen35.block_count"] - 1;
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  const errs = []; device.addEventListener("uncapturederror", (e) => errs.push(e.error?.message));
  const eng = await Qwen35Engine.create({ device, meta: G.meta, layerRange: [0, L], hasEmbed: true, hasHead: true, vocab: G.tensors[GGML_EMBED].shape[0],
    maxSeq: 2048, batchCols: 16, coopRowsB: 1, coopWG: 64,
    weights: await qwen35Weights(G, bytesOf, { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true }) });
  const collect = async (it) => { let s = ""; for await (const d of it) s += d; return s; };
  const results = [];
  const check = (name, ok, detail = "") => { results.push(ok); say(`${ok ? "PASS" : "FAIL"} ${name}${detail ? "  " + detail : ""}`); };
  const system = "You are Tabby, a coding agent.";
  const u1 = "Read src/add.js and tell me what it does.", u2 = "Now fix the bug in it.";

  const m = engineModel(eng, tok, { maxNew: 16 });
  const a1 = await collect(m.generate({ system, turns: [{ role: "user", text: u1 }] }));
  const turns2 = [{ role: "user", text: u1 }, { role: "assistant", text: a1 }, { role: "user", text: u2 }];
  const before = m.stats.reused;
  const a2 = await collect(m.generate({ system, turns: turns2 }));
  const reused = m.stats.reused - before;
  check("follow-up turn reuses the engine's prefix", reused > 0, `${reused} tokens reused, ${m.stats.prefilled} prefilled in total`);

  const fresh = engineModel(eng, tok, { maxNew: 16 });
  const b2 = await collect(fresh.generate({ system, turns: turns2 }));
  check("reused prefix == fresh engine (same text)", a2 === b2, JSON.stringify(a2.slice(0, 40)));

  const plain = engineModel(eng, tok, { maxNew: 16, spec: false });
  const c2 = await collect(plain.generate({ system, turns: turns2 }));
  check("speculative == plain", c2 === b2);

  const ws = new MemoryWorkspace({ "src/add.js": "export const add = (a, b) => a - b;\n" });
  const A = new Agent({ generate: engineModel(eng, tok, { maxNew: 24 }).generate, tools: codingTools(ws).filter((t) => t.name === "read_file"), system, maxSteps: 3 });   // one tool: the synthetic tokenizer makes long prompts
  const r = await A.run(u1);
  check("agent loop runs on the engine", typeof r.text === "string" && r.steps >= 1, `${r.steps} step(s), ${r.calls} tool call(s)`);
  check("no GPU validation errors", !errs.length, errs.slice(0, 2).join(" | "));
  return { out, ok: results.every(Boolean) };
}
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "swarm-ag-"));
const model = arg("model") || writeSynth(path.join(tmp, "m.gguf"), {}).file;
const srv = serveRepo(PORT, { "/__m.gguf": path.resolve(model) });
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), args: GPU_ARGS });
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("[pageerror]", String(e).slice(0, 300)));
await page.goto(`http://127.0.0.1:${PORT}/favicon.svg`);
const res = await page.evaluate(pageMain);
for (const l of res.out) console.log(l);
console.log(res.ok ? "AGENT PASS" : "AGENT FAIL");
await browser.close(); srv.close(); fs.rmSync(tmp, { recursive: true, force: true });
process.exit(res.ok ? 0 : 1);
