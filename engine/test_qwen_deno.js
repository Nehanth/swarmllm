// Qwen3-0.6B Q8_0 on the WebGPU engine vs the CPU golden reference.
import { makeTokenizer, BelloEngine, argmax } from "./engine.js";
import { parseGGUFHeader, ggufWeights } from "./gguf.js";

const dir = new URL(".", import.meta.url).pathname;
const cfg = JSON.parse(await Deno.readTextFile(dir + "qwen/config.json"));
const golden = JSON.parse(await Deno.readTextFile(dir + "golden_qwen.json"));
const tok = makeTokenizer(JSON.parse(await Deno.readTextFile(dir + "qwen/tokenizer.json")));
const raw = await Deno.readFile(dir + "qwen/model.gguf");
const buf = raw.buffer;
const G = parseGGUFHeader(buf);
console.log("arch:", G.meta["general.architecture"]);

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({
  requiredLimits: {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  },
});
device.addEventListener?.("uncapturederror", (e) => console.error("GPU ERROR:", e.error?.message));

const bytesOf = (info) => new Uint8Array(buf, info.byteOffset, info.byteLength);
const L = cfg.num_hidden_layers;
const weights = await ggufWeights(G, bytesOf, { lo: 0, hi: L, hasEmbed: true, hasHead: true });
const eng = await BelloEngine.create({ device, cfg, weights });
console.log("engine ready (q8)");

let logits = null, capture = null;
for (let i = 0; i < golden.ids.length; i++) {
  const cap = i === golden.ids.length - 1 ? {} : null;
  logits = await eng.forwardToken(golden.ids[i], cap);
  if (cap) capture = cap;
}
let worst = 0, wl = -1;
for (const [li, ref] of Object.entries(golden.perLayer)) {
  for (let i = 0; i < ref.length; i++) {
    const d = Math.abs(capture[li][i] - ref[i]) / (Math.abs(ref[i]) + 1e-3);
    if (d > worst) { worst = d; wl = li; }
  }
}
console.log(`per-layer worst rel-diff ${(worst * 100).toFixed(3)}% (layer ${wl})`);
const top = [...logits.keys()].sort((a, b) => logits[b] - logits[a]).slice(0, 5);
console.log("golden top5:", JSON.stringify(golden.logitsTop));
console.log("engine top5:", JSON.stringify(top.map((i) => [i, +logits[i].toFixed(4)])));

const gen = [];
const t0 = performance.now();
for (let i = 0; i < golden.generated.length; i++) {
  const n = argmax(logits);
  gen.push(n);
  logits = await eng.forwardToken(n);
}
const secs = (performance.now() - t0) / 1000;
console.log("generated:", JSON.stringify(tok.decode(gen)));
console.log("golden   :", JSON.stringify(tok.decode(golden.generated)));
console.log(`speed: ${(gen.length / secs).toFixed(1)} tok/s`);
const ok = JSON.stringify(gen) === JSON.stringify(golden.generated) && top[0] === golden.logitsTop[0][0];
console.log(ok ? "\nQWEN Q8 PASS ✓" : "\nQWEN Q8 FAIL ✗");
Deno.exit(ok ? 0 : 1);
