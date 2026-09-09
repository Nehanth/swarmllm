// Every shape the autotuner may pick must agree with the default (to float rounding): the tuner
// picks by timing, so a shape with a kernel bug shows up as a rare, run-dependent wrong answer.
import { DenseEngine } from "../engine/engine.js";
import { parseGGUFHeader, ggufWeights } from "../engine/gguf.js";
import { candidatesFor } from "../engine/autotune.js";
const openFile = async (path) => {
  const fh = await Deno.open(path);
  return async (off, len) => { await fh.seek(off, Deno.SeekMode.Start); const out = new Uint8Array(len); let got = 0;
    while (got < len) { const n = await fh.read(out.subarray(got)); if (n === null) break; got += n; } return out; };
};
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
let gpuErrors = 0;
device.addEventListener?.("uncapturederror", (e) => { gpuErrors++; console.error("GPU ERROR:", e.error?.message?.slice(0, 200)); });
const dir = new URL(".", import.meta.url).pathname;
const readAt = await openFile(dir + "../models/qwen/model.gguf");
const cfg = JSON.parse(await Deno.readTextFile(dir + "../models/qwen/config.json"));
const G = parseGGUFHeader((await readAt(0, 64 << 20)).buffer, { skipTokenizer: true });
const bytesOf = (info) => readAt(info.byteOffset, info.byteLength);
const ids = [151644, 872, 198, 40451, 752, 911, 279, 17951, 13, 151645, 198, 151644, 77091, 198, 9707, 11];
const l1 = (x) => { let s = 0; for (const v of x) s += Math.abs(v); return s; };
const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
const run = async (wg, rows) => {
  const w = await ggufWeights(G, bytesOf, { lo: 0, hi: 10, hasEmbed: true, hasHead: true });
  const e = await DenseEngine.create({ device, cfg, weights: w, layerRange: [0, 10], hasEmbed: true, hasHead: true, maxSeq: 128, coopWG: wg, coopRows: rows });
  e.reset();
  const b = Float32Array.from(await e.embedRunBatch(ids.slice(0, 4), 0));
  const s = Float32Array.from(await e.embedRun(ids[4], 4));
  const lg = Float32Array.from(await e.headFromHidden(s));
  return { b, s, lg };
};
const ref = await run(256, 4);
console.log("ref |b|=", l1(ref.b).toFixed(1), "|s|=", l1(ref.s).toFixed(1));
// shapes differ in accumulation order, so they agree to float rounding, not to the byte; the
// broken 8-row variant is off by orders of magnitude (kept here as a known failure)
const rel = (a, b) => { let d = 0, n = 0; for (let i = 0; i < a.length; i++) { d += Math.abs(a[i] - b[i]); n += Math.abs(b[i]); } return d / n; };
let fails = 0;
const shapes = [[128, 4], [64, 4], [256, 8], [128, 8]];
for (const [wg, rows] of shapes) {
  const r = await run(wg, rows);
  const e = Math.max(rel(r.b, ref.b), rel(r.s, ref.s), rel(r.lg, ref.lg));
  const inTuner = candidatesFor(false).some(([w, ro]) => w === wg && ro === rows);   // dense engines: no 8-row shapes
  const ok = e < 1e-4;
  console.log((ok ? "PASS" : (inTuner ? "FAIL" : "KNOWN-BAD (not offered to dense engines)")), `WG=${wg} ROWS=${rows} rel err ${e.toExponential(2)}`);
  if (ok && !inTuner) { console.log("FAIL: this shape is good now, offer it to dense engines again"); fails++; }
  if (!ok && inTuner) fails++;
}
// the 27B kernels (q4, Qwen35Engine, batchCols 16) with the same shapes, when the model is here
try {
  await Deno.stat(dir + "../models/q38/model.gguf");
  const { Qwen35Engine } = await import("../engine/qwen35.js");
  const { qwen35Weights } = await import("../engine/gguf.js");
  const readQ = await openFile(dir + "../models/q38/model.gguf");
  const GQ = parseGGUFHeader((await readQ(0, 16 << 20)).buffer);
  const runQ = async (wg, rows) => {
    const w = await qwen35Weights(GQ, (i) => readQ(i.byteOffset, i.byteLength), { lo: 0, hi: 4, hasEmbed: true, hasHead: true, mtp: false }, () => {});
    const e = await Qwen35Engine.create({ device, meta: GQ.meta, weights: w, layerRange: [0, 4], hasEmbed: true, hasHead: true, maxSeq: 128, batchCols: 16, coopRowsB: 1, coopWG: wg, coopRows: rows });
    e.reset();
    const b = Float32Array.from(await e.embedRunBatch(ids.slice(0, 4), 0));
    const s = Float32Array.from(await e.embedRun(ids[4], 4));
    return { b, s };
  };
  const refQ = await runQ(256, 4);
  console.log("27B ref |b|=", l1(refQ.b).toFixed(1), "|s|=", l1(refQ.s).toFixed(1));
  for (const [wg, rows] of shapes) {
    const r = await runQ(wg, rows);
    const e = Math.max(rel(r.b, refQ.b), rel(r.s, refQ.s));
    const inTuner = candidatesFor(true).some(([w, ro]) => w === wg && ro === rows);   // Qwen 3.8: all shapes offered
    const ok = e < 1e-4;
    console.log((ok ? "PASS" : (inTuner ? "FAIL" : "KNOWN-BAD")), `27B WG=${wg} ROWS=${rows} rel err ${e.toExponential(2)}`);
    if (!ok && inTuner) fails++;
  }
} catch (e) { console.log("27B section skipped:", e.message.slice(0, 80)); }
console.log(gpuErrors ? "GPU errors: " + gpuErrors : "no GPU errors");
console.log(fails || gpuErrors ? "\nTUNE FAIL" : "\nTUNE PASS ✓ (every shape the tuner may pick agrees with the default to float rounding)");
Deno.exit(fails || gpuErrors ? 1 : 0);
