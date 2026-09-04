// Split-pipeline test: two engine shards (simulated peers) run half the model
// each, hidden states hop between them — the exact protocol phase 2 runs over
// WebRTC. Output must match the golden reference token-for-token.
// usage: deno run --unstable-webgpu --allow-read test_split_deno.js
import { parseSafetensors, makeTokenizer, BelloEngine, argmax } from "../engine/engine.js";

const dir = new URL(".", import.meta.url).pathname;
const cfg = JSON.parse(await Deno.readTextFile(dir + "../models/model/config.json"));
const golden = JSON.parse(await Deno.readTextFile(dir + "./golden/golden.json"));
const tok = makeTokenizer(JSON.parse(await Deno.readTextFile(dir + "../models/model/tokenizer.json")));
const tensors = parseSafetensors((await Deno.readFile(dir + "../models/model/model.safetensors")).buffer);

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
device.addEventListener?.("uncapturederror", (e) => console.error("GPU ERROR:", e.error?.message));

const mid = Math.floor(cfg.num_hidden_layers / 2);
console.log(`host: embed + layers [0,${mid}) + head · worker: layers [${mid},${cfg.num_hidden_layers})`);
const host = await BelloEngine.create({ device, cfg, tensors, layerRange: [0, mid], hasEmbed: true, hasHead: true });
const worker = await BelloEngine.create({ device, cfg, tensors, layerRange: [mid, cfg.num_hidden_layers], hasEmbed: false, hasHead: false });

async function pipelineToken(id, pos) {
  const h1 = await host.embedRun(id, pos);       // "peer 1"
  const h2 = await worker.runHidden(h1, pos);    // -> would cross WebRTC -> "peer 2"
  return await host.headFromHidden(h2);          // -> back to peer 1 for the head
}

let pos = 0, logits = null;
for (const id of golden.ids) logits = await pipelineToken(id, pos++);

const gen = [];
const t0 = performance.now();
for (let i = 0; i < golden.generated.length; i++) {
  const next = argmax(logits);
  gen.push(next);
  logits = await pipelineToken(next, pos++);
}
const secs = (performance.now() - t0) / 1000;

console.log("generated:", JSON.stringify(tok.decode(gen)));
console.log("golden   :", JSON.stringify(tok.decode(golden.generated)));
console.log(`hidden-state hops: ${pos * 2} × ${cfg.hidden_size * 4} bytes (${(cfg.hidden_size * 4 / 1024).toFixed(1)} KB per hop)`);
console.log(`speed: ${(gen.length / secs).toFixed(1)} tok/s`);
const ok = JSON.stringify(gen) === JSON.stringify(golden.generated);
console.log(ok ? "\nSPLIT PASS ✓" : "\nSPLIT FAIL ✗");
Deno.exit(ok ? 0 : 1);
