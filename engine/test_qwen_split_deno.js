// Reproduce the user's 2-Mac scenario: Qwen3-0.6B q8 GGUF split across two shards.
import { makeTokenizer, BelloEngine, argmax } from "./engine.js";
import { parseGGUFHeader, ggufWeights } from "./gguf.js";

const dir = new URL(".", import.meta.url).pathname;
const cfg = JSON.parse(await Deno.readTextFile(dir + "qwen/config.json"));
const golden = JSON.parse(await Deno.readTextFile(dir + "golden_qwen.json"));
const tok = makeTokenizer(JSON.parse(await Deno.readTextFile(dir + "qwen/tokenizer.json")));
const raw = await Deno.readFile(dir + "qwen/model.gguf");
const G = parseGGUFHeader(raw.buffer);
const bytesOf = (info) => new Uint8Array(raw.buffer, info.byteOffset, info.byteLength);

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({
  requiredLimits: {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  },
});
device.addEventListener?.("uncapturederror", (e) => console.error("GPU ERROR:", e.error?.message));

const L = cfg.num_hidden_layers, mid = 14;
const wHost = await ggufWeights(G, bytesOf, { lo: 0, hi: mid, hasEmbed: true, hasHead: true });
const wWork = await ggufWeights(G, bytesOf, { lo: mid, hi: L, hasEmbed: false, hasHead: false });
const host = await BelloEngine.create({ device, cfg, weights: wHost, layerRange: [0, mid], hasEmbed: true, hasHead: true });
const work = await BelloEngine.create({ device, cfg, weights: wWork, layerRange: [mid, L], hasEmbed: false, hasHead: false });
console.log("shards ready");

let pos = 0;
const pipe = async (id) => {
  const h1 = await host.embedRun(id, pos);
  const h2 = await work.runHidden(h1, pos);
  pos++;
  return await host.headFromHidden(h2);
};
let logits = null;
for (const id of golden.ids) logits = await pipe(id);
const nan = logits.some?.(Number.isNaN) || Number.isNaN(logits[0]);
console.log("logits[0..5]:", Array.from(logits.slice(0, 5)).map(v => +v.toFixed(3)), "NaN?", nan);
const gen = [];
for (let i = 0; i < golden.generated.length; i++) { const n = argmax(logits); gen.push(n); logits = await pipe(n); }
console.log("generated:", JSON.stringify(tok.decode(gen)));
console.log("golden   :", JSON.stringify(tok.decode(golden.generated)));
console.log(JSON.stringify(gen) === JSON.stringify(golden.generated) ? "QWEN SPLIT PASS" : "QWEN SPLIT FAIL");
