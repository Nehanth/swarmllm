// FULL Qwen3.8-27B on the bello engine: 64 layers + head, greedy generation.
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

const L = 64;
console.log("loading ALL", L, "layers + embed + head (15GB)…");
let t0 = performance.now();
let lastLog = 0;
const weights = await qwen35Weights(G, bytesOf, { lo: 0, hi: L, hasEmbed: true, hasHead: true }, (done) => {
  if (done - lastLog > 2e9) { lastLog = done; console.log(" …", (done / 2 ** 30).toFixed(1), "GB"); }
});
console.log("weights loaded in", ((performance.now() - t0) / 1000).toFixed(0), "s");
t0 = performance.now();
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true });
console.log("engine built in", ((performance.now() - t0) / 1000).toFixed(1), "s");

const tok = makeTokenizer(tokenizerFromGGUF(G.meta));
const ids = tok.encode("The capital of France is");
console.log("prompt ids:", JSON.stringify(ids));

let logits = null;
t0 = performance.now();
for (const id of ids) logits = await eng.forwardToken(id);
console.log("prefill:", ((performance.now() - t0) / 1000).toFixed(1), "s");

const GOLDEN = " Paris.\nThe capital of Germany is Berlin.\nThe";
const gen = [];
t0 = performance.now();
for (let i = 0; i < 12; i++) {
  const next = argmax(logits);
  gen.push(next);
  logits = await eng.forwardToken(next);
}
const secs = (performance.now() - t0) / 1000;
const text = tok.decode(gen);
console.log("engine :", JSON.stringify(text));
console.log("llama.cpp:", JSON.stringify(GOLDEN));
console.log("speed:", (12 / secs).toFixed(2), "tok/s");
console.log(text === GOLDEN ? "\nQWEN3.8-27B FULL PASS ✓✓✓" : "\nQWEN3.8 MISMATCH");
