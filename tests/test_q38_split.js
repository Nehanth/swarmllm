// Qwen3.8 split across two shards (the 2-device scenario).
import { parseGGUFHeader, qwen35Weights, tokenizerFromGGUF } from "../engine/gguf.js";
import { makeTokenizer, argmax } from "../engine/engine.js";
import { Qwen35Engine } from "../engine/qwen35.js";

const dir = new URL(".", import.meta.url).pathname;
const file = await Deno.open(dir + "../models/q38/model.gguf", { read: true });
const headBuf = new Uint8Array(16 * 1024 * 1024);
let got = 0;
while (got < headBuf.length) { const n = await file.read(headBuf.subarray(got)); if (n === null) break; got += n; }
const G = parseGGUFHeader(headBuf.buffer);
const bytesOf = async (info) => {
  const out = new Uint8Array(info.byteLength);
  await file.seek(info.byteOffset, Deno.SeekMode.Start);
  let o = 0;
  while (o < out.length) { const n = await file.read(out.subarray(o)); if (n === null) break; o += n; }
  return out;
};
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({
  requiredLimits: {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  },
});
device.addEventListener?.("uncapturederror", (e) => console.error("GPU ERROR:", e.error?.message));

const L = 64, mid = 33;
console.log(`host [0,${mid}) + worker [${mid},${L})`);
const wHost = await qwen35Weights(G, bytesOf, { lo: 0, hi: mid, hasEmbed: true, hasHead: true });
const host = await Qwen35Engine.create({ device, meta: G.meta, weights: wHost, layerRange: [0, mid], hasEmbed: true, hasHead: true });
const wWork = await qwen35Weights(G, bytesOf, { lo: mid, hi: L, hasEmbed: false, hasHead: false });
const work = await Qwen35Engine.create({ device, meta: G.meta, weights: wWork, layerRange: [mid, L], hasEmbed: false, hasHead: false });
console.log("shards ready");

const tok = makeTokenizer(tokenizerFromGGUF(G.meta));
const ids = tok.encode("The capital of France is");
let pos = 0, logits = null;
const pipe = async (id) => {
  const h1 = await host.embedRun(id, pos);
  const h2 = await work.runHidden(h1, pos);
  pos++;
  return await host.headFromHidden(h2);
};
for (const id of ids) logits = await pipe(id);
const gen = [];
const t0 = performance.now();
for (let i = 0; i < 12; i++) { const n = argmax(logits); gen.push(n); logits = await pipe(n); }
const text = tok.decode(gen);
console.log("split :", JSON.stringify(text));
console.log("speed:", (12 / ((performance.now() - t0) / 1000)).toFixed(2), "tok/s");
console.log(text === " Paris.\nThe capital of Germany is Berlin.\nThe" ? "QWEN3.8 SPLIT PASS ✓" : "SPLIT MISMATCH");
