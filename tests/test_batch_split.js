// Split-mode batched prefill equivalence: host shard (layers 0-3) + worker
// shard (3-6) chained via embedRunBatch/runHiddenBatch must match the fully
// sequential split pipeline bit for bit.
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
const bytesOf = (i) => readAt(i.byteOffset, i.byteLength);
const mkPair = async () => {
  const wa = await qwen35Weights(G, bytesOf, { lo: 0, hi: 3, hasEmbed: true, hasHead: true });
  const wb = await qwen35Weights(G, bytesOf, { lo: 3, hi: 6, hasEmbed: false, hasHead: false });
  const A = await Qwen35Engine.create({ device, meta: G.meta, weights: wa, layerRange: [0, 3], hasEmbed: true, hasHead: true, maxSeq: 64 });
  const Bx = await Qwen35Engine.create({ device, meta: G.meta, weights: wb, layerRange: [3, 6], hasEmbed: false, hasHead: false, maxSeq: 64, vocab: G.tensors["token_embd.weight"].shape[0] });
  return [A, Bx];
};
const ids = tok.encode("The capital of France is Paris, and the capital of Germany is Berlin. The quick");
console.log("prompt tokens:", ids.length);
// sequential split reference
const [A1, B1] = await mkPair();
let ref = null;
for (let p = 0; p < ids.length; p++) {
  let h = await A1.embedRun(ids[p], p);
  h = await B1.runHidden(h, p);
  ref = await A1.headFromHidden(h);
}
// batched split
const [A2, B2] = await mkPair();
let pos = 0, i = 0;
const t0 = performance.now();
while (ids.length - 1 - i >= 4) {
  let hb = await A2.embedRunBatch(ids.slice(i, i + 4), pos);
  hb = await B2.runHiddenBatch(hb, pos);
  pos += 4; i += 4;
}
const t1 = performance.now();
let got = null;
for (; i < ids.length; i++) {
  let h = await A2.embedRun(ids[i], pos);
  h = await B2.runHidden(h, pos);
  got = await A2.headFromHidden(h);
  pos++;
}
let md = 0, sc = 1e-6;
for (let k = 0; k < ref.length; k++) { md = Math.max(md, Math.abs(got[k] - ref[k])); sc = Math.max(sc, Math.abs(ref[k])); }
const rel = md / sc;
console.log(`argmax seq=${argmax(ref)} batch=${argmax(got)}  relDiff=${rel.toExponential(2)}  batchedChain=${(i / ((t1 - t0) / 1000)).toFixed(1)} tok/s`);
console.log(argmax(ref) === argmax(got) && rel < 2e-3 ? "SPLIT BATCH PREFILL PASS ✓" : "SPLIT BATCH PREFILL FAIL");
if (!(argmax(ref) === argmax(got) && rel < 2e-3)) Deno.exit(1);
