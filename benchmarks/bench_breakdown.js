// Where does a batched (4-column) pass spend its time? Times full passes with
// kernel families skipped (results are garbage; timing is what matters).
import { Qwen35Engine } from "../engine/qwen35.js";
import { parseGGUFHeader, qwen35Weights } from "../engine/gguf.js";
const openFile = async (path) => { const fh = await Deno.open(path); return async (off, len) => { await fh.seek(off, Deno.SeekMode.Start); const out = new Uint8Array(len); let got = 0; while (got < len) { const n = await fh.read(out.subarray(got)); if (n === null) break; got += n; } return out; }; };
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
const readAt = await openFile("../models/q38/model.gguf");
const G = parseGGUFHeader((await readAt(0, 64 << 20)).buffer, { skipTokenizer: true });
const L = +(Deno.env.get("LAYERS") || 64);
const weights = await qwen35Weights(G, (i) => readAt(i.byteOffset, i.byteLength), { lo: 0, hi: L, hasEmbed: true, hasHead: true });
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, vocab: 248320, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 512 });
eng._initBatch();
const ids = [10, 11, 12, 13];
async function timeBatch(n = 6) {
  eng.reset(); eng.pos = 0;
  await eng.embedRunBatch(ids, 0); await device.queue.onSubmittedWorkDone();
  const t0 = performance.now();
  for (let i = 0; i < n; i++) await eng.embedRunBatch(ids, 4 * (i + 1));
  await device.queue.onSubmittedWorkDone();
  return (performance.now() - t0) / n;
}
async function timeSingle(n = 6) {
  eng.reset(); eng.pos = 0;
  await eng.forwardToken(10);
  const t0 = performance.now();
  for (let i = 0; i < n; i++) await eng.forwardToken(10);
  return (performance.now() - t0) / n;
}
const single = await timeSingle();
console.log(`single pass: ${single.toFixed(1)} ms`);
const famsS = {
  "matvec coop (all decode matvecs+head)": ["matvec_q4_coop", "matvec_q8_coop", "matvec_coop", "matvec_q4_gu", "matvec_q8_gu", "matvec_gu"],
  "dn_delta": ["dn_delta"],
  "dn_conv": ["dn_conv"],
  "dn_gates/l2/gatenorm": ["dn_gates", "dn_l2", "dn_gatenorm"],
  "rmsnorm + add_res": ["rmsnorm", "add_res"],
  "attention (scores/softmax/out)": ["attn_scores", "attn_softmax", "attn_out"],
  "qsplit/head_norm/rope/sigmoid": ["qsplit", "head_norm", "rope_part", "sigmoid_mul"],
};
for (const [name, list] of Object.entries(famsS)) {
  eng.skip = new Set(list);
  const t = await timeSingle();
  console.log(`  single without ${name.padEnd(38)} ${t.toFixed(1)} ms  -> family costs ~${(single - t).toFixed(1)} ms`);
}
eng.skip = null;
const full = await timeBatch();
console.log(`batched pass (all): ${full.toFixed(1)} ms`);
const fams = {
  "matvec _b (all batched matvecs)": ["matvec_q4_coop_b", "matvec_q8_coop_b", "matvec_coop_b", "matvec_q4_gu_b", "matvec_q8_gu_b", "matvec_gu_b"],
  "dn_delta_mc": ["dn_delta_mc"],
  "dn_conv_mc": ["dn_conv_mc"],
  "dn_gates/l2/gatenorm": ["dn_gates_mc", "dn_l2_mc", "dn_gatenorm_mc"],
  "rmsnorm_mc + add_res_mc": ["rmsnorm_mc", "add_res_mc"],
  "attention (scores/softmax/out)": ["attn_scores", "attn_softmax", "attn_out"],
  "qsplit/head_norm/rope/sigmoid": ["qsplit_mc", "head_norm_mc", "rope_part_mc", "sigmoid_mul_mc"],
};
for (const [name, list] of Object.entries(fams)) {
  eng.skip = new Set(list);
  const t = await timeBatch();
  console.log(`  without ${name.padEnd(34)} ${t.toFixed(1)} ms  -> family costs ~${(full - t).toFixed(1)} ms`);
}
eng.skip = null;
