// DenseEngine.narrow(lo, hi): serving a sub-range of the loaded layers without a reload must
// give the same bytes as a fresh load of that sub-range. Worker narrowed from the top and from
// the bottom (single and batched laps, after the batch path was already used), and the host
// engine (embed + head) narrowed from the top.
import { DenseEngine } from "../engine/engine.js";
import { parseGGUFHeader, ggufWeights } from "../engine/gguf.js";
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
const mk = async (lo, hi, hasEmbed, hasHead) => {
  const w = await ggufWeights(G, bytesOf, { lo, hi, hasEmbed, hasHead });
  return DenseEngine.create({ device, cfg, weights: w, layerRange: [lo, hi], hasEmbed, hasHead, maxSeq: 128 });
};
const dim = cfg.hidden_size;
let seed = 7; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 2 ** 32; };
const vec = (n) => { const x = new Float32Array(n); for (let i = 0; i < n; i++) x[i] = (rnd() - 0.5) * 2; return x; };
const same = (a, b) => { if (a.length !== b.length) return false; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; };
let fails = 0;
const check = (name, ok) => { console.log((ok ? "PASS " : "FAIL ") + name); if (!ok) fails++; };

// a worker's lap sequence: a batched lap of 4 at 0, then singles at 4 and 5
const laps = async (eng) => {
  seed = 7;
  eng.reset();
  const b = Float32Array.from(await eng.runHiddenBatch(vec(4 * dim), 0));
  const s4 = Float32Array.from(await eng.runHidden(vec(dim), 4));
  const s5 = Float32Array.from(await eng.runHidden(vec(dim), 5));
  return { b, s4, s5 };
};

// (1) top: [16,24) used, narrowed to [16,23), vs fresh [16,23)
{
  const a = await mk(16, 24, false, false);
  await laps(a);                           // the batch path has been built on the wide engine
  a.narrow(16, 23);
  const got = await laps(a);
  const fresh = await mk(16, 23, false, false);
  const want = await laps(fresh);
  check("worker narrowed from the top [16,24)->[16,23): batched lap", same(got.b, want.b));
  check("worker narrowed from the top: single laps", same(got.s4, want.s4) && same(got.s5, want.s5));
}
// (2) bottom: [16,24) narrowed to [18,24), vs fresh [18,24)
{
  const a = await mk(16, 24, false, false);
  await laps(a);
  a.narrow(18, 24);
  const got = await laps(a);
  const fresh = await mk(18, 24, false, false);
  const want = await laps(fresh);
  check("worker narrowed from the bottom [16,24)->[18,24): batched lap", same(got.b, want.b));
  check("worker narrowed from the bottom: single laps", same(got.s4, want.s4) && same(got.s5, want.s5));
}
// (3) host: all 28 layers with embed+head, narrowed to [0,22), vs fresh [0,22)
{
  const L = cfg.num_hidden_layers;
  const hostLaps = async (eng) => {
    eng.reset();
    const b = Float32Array.from(await eng.embedRunBatch([9707, 11, 1246, 279], 0));
    const s = Float32Array.from(await eng.embedRun(315, 4));
    const lg = Float32Array.from(await eng.headFromHidden(s));
    return { b, s, lg };
  };
  const a = await mk(0, L, true, true);
  await a.forwardToken(9707); await a.forwardToken(11);   // used solo first
  await hostLaps(a);
  a.narrow(0, 22);
  const got = await hostLaps(a);
  const fresh = await mk(0, 22, true, true);
  const want = await hostLaps(fresh);
  check("host narrowed from the top [0,28)->[0,22): batched embed lap", same(got.b, want.b));
  check("host narrowed from the top: single embed lap and head", same(got.s, want.s) && same(got.lg, want.lg));
}
// (4) the failing room layout, engines only: boss [0,10) embed+head, phone [10,16), late [16,22)
//     fresh, creator [16,28) narrowed to [22,28). Batched prefill of 4 tokens then two single
//     tokens through the chain must equal a solo engine's logits bit for bit.
{
  const L = cfg.num_hidden_layers;
  const ids = [9707, 11, 1246, 279], t4 = 315, t5 = 374;
  const solo = await mk(0, L, true, true);
  solo.reset(); await solo.prefillTokens(ids);
  const s4 = Float32Array.from(await solo.forwardToken(t4));
  const s5 = Float32Array.from(await solo.forwardToken(t5));
  const boss = await mk(0, 10, true, true), phone = await mk(10, 16, false, false), late = await mk(16, 22, false, false);
  const creator = await mk(16, 28, false, false);
  await laps(creator);             // used in the old plan first
  creator.narrow(22, 28);
  const chain = [phone, late, creator];
  for (const e of [boss, ...chain]) e.reset();
  let hb = await boss.embedRunBatch(ids, 0);
  for (const e of chain) hb = await e.runHiddenBatch(hb, 0);
  boss.pos = 4;
  let h = await boss.embedRun(t4, 4);
  for (const e of chain) h = await e.runHidden(h, 4);
  const c4 = Float32Array.from(await boss.headFromHidden(h));
  h = await boss.embedRun(t5, 5);
  for (const e of chain) h = await e.runHidden(h, 5);
  const c5 = Float32Array.from(await boss.headFromHidden(h));
  check("chain boss/phone/late/narrowed creator == solo (token 4 logits)", same(c4, s4));
  check("chain boss/phone/late/narrowed creator == solo (token 5 logits)", same(c5, s5));
}
check("no GPU errors", gpuErrors === 0);
console.log(fails ? "\nNARROW FAIL" : "\nNARROW PASS ✓ (a narrowed engine is byte-identical to a fresh load of the sub-range)");
Deno.exit(fails ? 1 : 0);
