// Qwen3.8 GPU engine vs validated CPU reference (layers 0..4, 2 tokens).
import { parseGGUFHeader, qwen35Weights, tokenizerFromGGUF } from "../engine/gguf.js";
import { makeTokenizer } from "../engine/engine.js";
import { Qwen35Engine } from "../engine/qwen35.js";

const dir = new URL(".", import.meta.url).pathname;
const file = await Deno.open(dir + "../models/q38/model.gguf", { read: true });
const headBuf = new Uint8Array(16 * 1024 * 1024);
let got = 0;
while (got < headBuf.length) {
  const n = await file.read(headBuf.subarray(got));
  if (n === null) break;
  got += n;
}
const G = parseGGUFHeader(headBuf.buffer);
const bytesOf = async (info) => {
  const out = new Uint8Array(info.byteLength);
  await file.seek(info.byteOffset, Deno.SeekMode.Start);
  let o = 0;
  while (o < out.length) {
    const n = await file.read(out.subarray(o));
    if (n === null) break;
    o += n;
  }
  return out;
};

const adapter = await navigator.gpu.requestAdapter();
console.log("maxStorageBinding:", (adapter.limits.maxStorageBufferBindingSize / 2 ** 30).toFixed(2), "GB");
const device = await adapter.requestDevice({
  requiredLimits: {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  },
});
device.addEventListener?.("uncapturederror", (e) => console.error("GPU ERROR:", e.error?.message));

const LO = 0, HI = 5;
console.log("loading shard layers", LO, "..", HI - 1);
let t0 = performance.now();
const weights = await qwen35Weights(G, bytesOf, { lo: LO, hi: HI, hasEmbed: true, hasHead: false });
console.log("weights in", ((performance.now() - t0) / 1000).toFixed(1), "s");
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [LO, HI], hasEmbed: true, hasHead: false });
console.log("engine ready");

const tok = makeTokenizer(tokenizerFromGGUF(G.meta));
const ids = tok.encode("The capital");
console.log("ids:", JSON.stringify(ids));

// reference layer_out-4 values from ref_q38.mjs (validated vs llama.cpp):
const REF = [
  [-0.04135, -0.00732],  // token 0: first two dims
  [0.12867, -0.13125],   // token 1
];
for (let ti = 0; ti < ids.length; ti++) {
  const h = await eng.embedRun(ids[ti], ti);
  const ok = Math.abs(h[0] - REF[ti][0]) < 0.02 && Math.abs(h[1] - REF[ti][1]) < 0.02;
  console.log(`token ${ti}: engine [${h[0].toFixed(5)}, ${h[1].toFixed(5)}] ref [${REF[ti]}] ${ok ? "MATCH" : "MISMATCH"}`);
}
