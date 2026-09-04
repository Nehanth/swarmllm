// Qwen3.8 batched-prefill equivalence on the first 6 layers (covers both
// DeltaNet and full-attention layer types + a batch tail).
import { Qwen35Engine } from "../engine/qwen35.js";
import { makeTokenizer, argmax } from "../engine/engine.js";
import { parseGGUFHeader, qwen35Weights, tokenizerFromGGUF } from "../engine/gguf.js";
const openFile = async (path) => {
  const fh = await Deno.open(path);
  return async (off, len) => { await fh.seek(off, Deno.SeekMode.Start); const out = new Uint8Array(len); let got = 0;
    while (got < len) { const n = await fh.read(out.subarray(got)); if (n === null) break; got += n; } return out; };
};
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
const readAt = await openFile("../models/q38/model.gguf");
const G = parseGGUFHeader((await readAt(0, 64 << 20)).buffer);
const tok = makeTokenizer(tokenizerFromGGUF(G.meta));
const L = 6;
const mk = async () => {
  const weights = await qwen35Weights(G, (i) => readAt(i.byteOffset, i.byteLength), { lo: 0, hi: L, hasEmbed: true, hasHead: true });
  return Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 64,
  batchCols: +(Deno.env.get("BCOLS") || 4), coopRowsB: +(Deno.env.get("ROWSB") || 4) });
};
const ids = tok.encode("The capital of France is Paris, and the capital of Germany is Berlin. The quick brown");
console.log("prompt tokens:", ids.length);
const e1 = await mk();
let ref = null;
for (const id of ids) ref = await e1.forwardToken(id);
const e2 = await mk();
const t0 = performance.now();
await e2.prefillTokens(ids.slice(0, -1));
const t1 = performance.now();
e2.pos = ids.length - 1;
const got = await e2.forwardToken(ids[ids.length - 1]);
let md = 0, sc = 1e-6;
for (let i = 0; i < ref.length; i++) { md = Math.max(md, Math.abs(got[i] - ref[i])); sc = Math.max(sc, Math.abs(ref[i])); }
const rel = md / sc;
console.log(`argmax seq=${argmax(ref)} batch=${argmax(got)}  relDiff=${rel.toExponential(2)}  batchedPrefill=${((ids.length - 1) / ((t1 - t0) / 1000)).toFixed(1)} tok/s`);
console.log(argmax(ref) === argmax(got) && rel < 2e-3 ? "Q38 BATCH PREFILL PASS ✓" : "Q38 BATCH PREFILL FAIL");
if (!(argmax(ref) === argmax(got) && rel < 2e-3)) Deno.exit(1);
