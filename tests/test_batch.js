// Batched-prefill equivalence: prefillTokens(chunked 4) must leave the engine
// in a state where the next forwardToken's logits match the all-sequential run.
import { DenseEngine, makeTokenizer, argmax } from "../engine/engine.js";
import { parseGGUFHeader, ggufWeights } from "../engine/gguf.js";
const openFile = async (path) => {
  const fh = await Deno.open(path);
  return async (off, len) => { await fh.seek(off, Deno.SeekMode.Start); const out = new Uint8Array(len); let got = 0;
    while (got < len) { const n = await fh.read(out.subarray(got)); if (n === null) break; got += n; } return out; };
};
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
const readAt = await openFile("../models/qwen/model.gguf");
const G = parseGGUFHeader((await readAt(0, 64 << 20)).buffer, { skipTokenizer: true });
const tok = makeTokenizer(JSON.parse(await Deno.readTextFile("../models/qwen/tokenizer.json")));
const cfg = JSON.parse(await Deno.readTextFile("../models/qwen/config.json"));
const mk = async () => {
  const weights = await ggufWeights(G, (i) => readAt(i.byteOffset, i.byteLength), { lo: 0, hi: cfg.num_hidden_layers, hasEmbed: true, hasHead: true });
  return DenseEngine.create({ device, cfg, weights, layerRange: [0, cfg.num_hidden_layers], hasEmbed: true, hasHead: true, maxSeq: 128 });
};
const ids = tok.encode("The capital of France is Paris, and the capital of Germany is Berlin. The quick brown fox");
console.log("prompt tokens:", ids.length);
// sequential reference
const e1 = await mk();
let ref = null;
for (const id of ids) ref = await e1.forwardToken(id);
// batched
const e2 = await mk();
const t0 = performance.now();
await e2.prefillTokens(ids.slice(0, -1));
const t1 = performance.now();
const got = await e2.forwardToken(ids[ids.length - 1]);
let md = 0, sc = 1e-6;
for (let i = 0; i < ref.length; i++) { md = Math.max(md, Math.abs(got[i] - ref[i])); sc = Math.max(sc, Math.abs(ref[i])); }
const rel = md / sc;
console.log(`argmax seq=${argmax(ref)} batch=${argmax(got)}  relDiff=${rel.toExponential(2)}  batchedPrefill=${((ids.length - 1) / ((t1 - t0) / 1000)).toFixed(1)} tok/s`);
console.log(argmax(ref) === argmax(got) && rel < 2e-3 ? "BATCH PREFILL PASS ✓" : "BATCH PREFILL FAIL");
if (!(argmax(ref) === argmax(got) && rel < 2e-3)) Deno.exit(1);
