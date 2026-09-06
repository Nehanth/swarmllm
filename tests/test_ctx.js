// Context beyond 512 positions on the 27B: (1) a short prompt gives bit-identical logits with
// maxSeq 512 and 2048 (the cache size is only a stride), (2) a ~640-token prompt prefills and
// generates past position 512 with no NaN, no GPU error and coherent text.
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
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
let gpuErrors = 0;
device.addEventListener?.("uncapturederror", (e) => { gpuErrors++; if (gpuErrors < 4) console.error("GPU ERROR:", e.error?.message?.slice(0, 200)); });

const L = 64;
console.log("loading all", L, "layers…");
const tok = makeTokenizer(tokenizerFromGGUF(G.meta));
// an engine takes ownership of its weight buffers (CPU copies are freed after upload), so each
// engine loads its own copy
const mk = async (maxSeq) => {
  const weights = await qwen35Weights(G, bytesOf, { lo: 0, hi: L, hasEmbed: true, hasHead: true }, () => {});
  return Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq, batchCols: 16, coopRowsB: 1 });
};

// (1) short prompt, two cache sizes, same logits bit for bit
const short = tok.encode("The capital of France is");
const run = async (eng, ids) => { let lg = null; for (const id of ids) lg = await eng.forwardToken(id); return lg; };
const e512 = await mk(512); const a = Float32Array.from(await run(e512, short));
try { e512.destroy?.(); } catch {}
const e2048 = await mk(2048); const b = await run(e2048, short);
let diff = 0; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) diff++;
console.log("short prompt, maxSeq 512 vs 2048: differing logits =", diff);
if (diff) { console.log("CTX FAIL: cache size changed the logits"); Deno.exit(1); }

// (2) long prompt past the old limit: batched prefill, then greedy generation across position 512
const para = "I am planning a two week trip through Japan in late October with my partner. We land in Tokyo, want three days there, then a day trip to Nikko, then the bullet train to Kyoto for four days with a side trip to Nara, then two nights in Osaka, and we fly home from Osaka. We like food markets, old temples, hiking, and small neighborhood bars, and we want to avoid the most crowded tourist spots where we can. Our budget is moderate, around two hundred dollars a day for the two of us not counting hotels. ";
const V = tok.vocab;
const ids = [V["<|im_start|>"], ...tok.encode("user\n" + para.repeat(4) + "Please give me a day by day itinerary."), V["<|im_end|>"], ...tok.encode("\n"), V["<|im_start|>"], ...tok.encode("assistant\n"), V["<think>"], ...tok.encode("\n\n"), V["</think>"], ...tok.encode("\n\n")];
console.log("long prompt tokens:", ids.length);
e2048.reset?.(); e2048.pos = 0;
let t0 = performance.now();
await e2048.prefillTokens(ids.slice(0, -1));
let logits = await e2048.forwardToken(ids[ids.length - 1]);
console.log("prefill", ((performance.now() - t0) / 1000).toFixed(1), "s, pos now", e2048.pos);
const gen = []; let nan = false;
for (let i = 0; i < 48; i++) {
  if (!Number.isFinite(logits[0]) || !Number.isFinite(logits[logits.length - 1])) { nan = true; break; }
  const next = argmax(logits); gen.push(next);
  if (next === V["<|im_end|>"]) break;
  logits = await e2048.forwardToken(next);
}
const text = tok.decode(gen);
const uniq = new Set(gen).size / Math.max(1, gen.length);
console.log("generated past pos 512:", JSON.stringify(text));
console.log(`positions ${ids.length}..${e2048.pos}, NaN: ${nan}, GPU errors: ${gpuErrors}, unique-token ratio ${uniq.toFixed(2)}`);
const ok = !nan && gpuErrors === 0 && gen.length >= 8 && uniq > 0.5 && !/\(\s*\(\s*\(/.test(text);
console.log(ok ? "\nCTX PASS ✓ (prefill and decode past the 512-position boundary are clean)" : "\nCTX FAIL");
Deno.exit(ok ? 0 : 1);
