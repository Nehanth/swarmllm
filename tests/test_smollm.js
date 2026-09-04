// Deno test: run the WebGPU engine against ref.js golden vectors.
// usage: deno run --unstable-webgpu --allow-read test_deno.js
import { parseSafetensors, makeTokenizer, DenseEngine, argmax } from "../engine/engine.js";

const dir = new URL(".", import.meta.url).pathname;
const cfg = JSON.parse(await Deno.readTextFile(dir + "../models/model/config.json"));
const golden = JSON.parse(await Deno.readTextFile(dir + "./golden/golden.json"));
const tok = makeTokenizer(JSON.parse(await Deno.readTextFile(dir + "../models/model/tokenizer.json")));
const st = (await Deno.readFile(dir + "../models/model/model.safetensors")).buffer;
const tensors = parseSafetensors(st);

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
device.addEventListener?.("uncapturederror", (e) => console.error("GPU ERROR:", e.error?.message));

console.log("building engine…");
let t0 = performance.now();
const eng = await DenseEngine.create({ device, cfg, tensors });
console.log(`engine ready in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

// --- forward the golden prompt, capture per-layer hiddens on last token ---
const ids = golden.ids;
let logits = null;
let capture = null;
for (let i = 0; i < ids.length; i++) {
  const cap = i === ids.length - 1 ? {} : null;
  logits = await eng.forwardToken(ids[i], cap);
  if (cap) capture = cap;
}

// --- compare per-layer hiddens ---
let worst = 0, worstLayer = -1;
for (const [li, ref] of Object.entries(golden.perLayer)) {
  const got = capture[li];
  for (let i = 0; i < ref.length; i++) {
    const d = Math.abs(got[i] - ref[i]) / (Math.abs(ref[i]) + 1e-3);
    if (d > worst) { worst = d; worstLayer = li; }
  }
}
console.log(`per-layer hidden match: worst rel-diff ${(worst * 100).toFixed(3)}% (layer ${worstLayer})`);

// --- compare top logits ---
const top = [...logits.keys()].sort((a, b) => logits[b] - logits[a]).slice(0, 5);
console.log("golden top-5:", JSON.stringify(golden.logitsTop));
console.log("engine top-5:", JSON.stringify(top.map(i => [i, +logits[i].toFixed(4)])));
const topMatch = top[0] === golden.logitsTop[0][0];

// --- greedy generation must match golden token-for-token ---
const gen = [];
t0 = performance.now();
for (let i = 0; i < golden.generated.length; i++) {
  const next = argmax(logits);
  gen.push(next);
  logits = await eng.forwardToken(next);
}
const secs = (performance.now() - t0) / 1000;
const tokMatch = JSON.stringify(gen) === JSON.stringify(golden.generated);
console.log("generated:", JSON.stringify(tok.decode(gen)));
console.log("golden   :", JSON.stringify(tok.decode(golden.generated)));
console.log(`speed: ${(gen.length / secs).toFixed(1)} tok/s`);
console.log(topMatch && tokMatch && worst < 0.02 ? "\nPASS ✓" : "\nFAIL ✗");
Deno.exit(topMatch && tokMatch ? 0 : 1);
