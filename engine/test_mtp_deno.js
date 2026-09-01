// MTP speculative decoding: greedy output must be IDENTICAL to plain greedy
// decoding (verification uses the trunk; the draft can only affect speed).
// Prints acceptance rate and tok/s for both paths. Full 64-layer 27B.
import { Qwen35Engine } from "./qwen35.js";
import { makeTokenizer, argmax } from "./engine.js";
import { parseGGUFHeader, qwen35Weights, tokenizerFromGGUF } from "./gguf.js";
const N = +(Deno.env.get("TOKENS") || 40);
const K = +(Deno.env.get("K") || 3);
const openFile = async (path) => {
  const fh = await Deno.open(path);
  return async (off, len) => { await fh.seek(off, Deno.SeekMode.Start); const out = new Uint8Array(len); let got = 0;
    while (got < len) { const n = await fh.read(out.subarray(got)); if (n === null) break; got += n; } return out; };
};
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
const readAt = await openFile("q38/model.gguf");
const G = parseGGUFHeader((await readAt(0, 64 << 20)).buffer);
const tok = makeTokenizer(tokenizerFromGGUF(G.meta));
const L = +(Deno.env.get("LAYERS") || (G.meta["qwen35.block_count"] - 1));
const t0 = performance.now();
const weights = await qwen35Weights(G, (i) => readAt(i.byteOffset, i.byteLength), { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: true });
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 512 });
console.log(`loaded in ${((performance.now() - t0) / 1000).toFixed(0)}s; mtp=${!!eng.mtp}`);
const V = tok.vocab;
const prompt = [V["<|im_start|>"], ...tok.encode("user\nWrite the Python code for two sum. Code only."), V["<|im_end|>"], ...tok.encode("\n"), V["<|im_start|>"], ...tok.encode("assistant\n"), V["<think>"], ...tok.encode("\n\n"), V["</think>"], ...tok.encode("\n\n")];
// ---- plain greedy ----
eng.reset(); eng.mtpFill = false;
await eng.prefillTokens(prompt.slice(0, -1));
let logits = await eng.forwardToken(prompt[prompt.length - 1]);
let next = argmax(logits); const plain = [next];
const tp0 = performance.now();
for (let i = 1; i < N; i++) { logits = await eng.forwardToken(next); next = argmax(logits); plain.push(next); }
const plainTs = (N - 1) / ((performance.now() - tp0) / 1000);
// ---- speculative greedy ----
eng.reset(); eng.mtpFill = true; eng.mtp.stats = { drafts: 0, accepted: 0 };
await eng.prefillTokens(prompt.slice(0, -1));
logits = await eng.forwardToken(prompt[prompt.length - 1]);
next = argmax(logits); const spec = [next];
const ts0 = performance.now();
while (spec.length < N) {
  const got = await eng.specStep(next, argmax, K);
  for (const t of got) spec.push(t);
  next = spec[spec.length - 1];
}
const specTs = (spec.length - 1) / ((performance.now() - ts0) / 1000);
const same = plain.every((t, i) => spec[i] === t);
const st = eng.mtp.stats;
console.log(`K=${K} plain: ${plainTs.toFixed(2)} tok/s   spec: ${specTs.toFixed(2)} tok/s   acceptance ${st.accepted}/${st.drafts} = ${(100 * st.accepted / Math.max(1, st.drafts)).toFixed(0)}%`);
console.log("plain:", JSON.stringify(tok.decode(plain).slice(0, 120)));
console.log("spec: ", JSON.stringify(tok.decode(spec.slice(0, N)).slice(0, 120)));
console.log(same ? "MTP SPEC PASS ✓ (identical output)" : "MTP SPEC FAIL (output differs)");
if (!same) Deno.exit(1);
