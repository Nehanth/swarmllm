// Qwen3.6-35B-A3B (MoE, 256 experts, top-8) on the engine: full model, greedy output vs llama.cpp, tok/s.
// Goldens: llama.cpp b10840 CUDA, same Q4_0 file (bartowski), --temp 0.
import { Qwen35Engine } from "../engine/qwen35.js";
import { makeTokenizer, argmax } from "../engine/engine.js";
import { parseGGUFHeader, qwen35Weights, tokenizerFromGGUF } from "../engine/gguf.js";
const N = +(Deno.env.get("TOKENS") || 40), K = +(Deno.env.get("K") || 3);
const PATH = Deno.env.get("MOE") || "../models/q36moe/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf";
const openFile = async (path) => { const fh = await Deno.open(path);
  return async (off, len) => { await fh.seek(off, Deno.SeekMode.Start); const out = new Uint8Array(len); let got = 0;
    while (got < len) { const n = await fh.read(out.subarray(got)); if (n === null) break; got += n; } return out; }; };
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
device.addEventListener?.("uncapturederror", (e) => console.error("GPU ERROR:", e.error?.message));
const readAt = await openFile(PATH);
const G = parseGGUFHeader((await readAt(0, 64 << 20)).buffer);
const arch = G.meta["general.architecture"], nBlk = G.meta[arch + ".block_count"], nextn = G.meta[arch + ".nextn_predict_layers"] || 0;
const hasMtp = G.tensors ? Object.keys(G.tensors).some((k) => k.startsWith(`blk.${nBlk - 1}.`)) : false;
const L = nBlk - nextn;
const tok = makeTokenizer(tokenizerFromGGUF(G.meta));
let t0 = performance.now();
const weights = await qwen35Weights(G, (i) => readAt(i.byteOffset, i.byteLength), { lo: 0, hi: L, hasEmbed: true, hasHead: true, mtp: hasMtp });
const eng = await Qwen35Engine.create({ device, meta: G.meta, weights, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 512 });
console.log(`${arch}: ${L} layers, mtp tensors ${hasMtp}, engine mtp ${!!eng.mtp}; loaded in ${((performance.now() - t0) / 1000).toFixed(0)}s`);
const V = tok.vocab;
const chat = (q) => [V["<|im_start|>"], ...tok.encode("user\n" + q), V["<|im_end|>"], ...tok.encode("\n"), V["<|im_start|>"], ...tok.encode("assistant\n"), V["<think>"], ...tok.encode("\n\n"), V["</think>"], ...tok.encode("\n\n")];
// (plain "The capital of France is" is a near tie after " Paris": "." 19.029 vs "," 18.968 here, llama.cpp CUDA picks ",". Not used as a golden.)
const CASES = [
  ["two-sum", chat("Write the Python code for two sum. Code only."), "```python\ndef two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:"],
  ["hash-map", chat("Explain what a hash map is in two sentences."), "A hash map is a data structure that stores key-value pairs, allowing for efficient retrieval, insertion, and deletion operations. It uses a hash function to compute an index into an array of buckets or slots"],
  ["bash", chat("Write a bash one-liner that counts lines in all .js files."), "```bash\nfind . -name '*.js' -exec cat {} + | wc -l\n```"],
];
let fail = 0;
for (const [name, prompt, golden] of CASES) {
  eng.reset(); if (eng.mtp) eng.mtpFill = false;
  t0 = performance.now(); await eng.prefillTokens(prompt.slice(0, -1)); let logits = await eng.forwardToken(prompt[prompt.length - 1]); const pf = (performance.now() - t0) / 1000;
  let next = argmax(logits); const gen = [next]; const tp0 = performance.now();
  for (let i = 1; i < N; i++) { logits = await eng.forwardToken(next); next = argmax(logits); gen.push(next); }
  const ts = (N - 1) / ((performance.now() - tp0) / 1000), text = tok.decode(gen), n = Math.min(text.length, golden.length), ok = n > 20 && text.slice(0, n) === golden.slice(0, n);
  console.log(`${name}: prefill ${prompt.length} tok ${pf.toFixed(2)}s · decode ${ts.toFixed(2)} tok/s · ${ok ? "MATCH llama.cpp" : "MISMATCH"}\n  engine: ${JSON.stringify(text)}${ok ? "" : "\n  golden: " + JSON.stringify(golden)}`);
  if (!ok) fail++;
  if (eng.mtp) {
    eng.reset(); eng.mtpFill = true; eng.mtp.stats = { drafts: 0, accepted: 0 };
    await eng.prefillTokens(prompt.slice(0, -1)); logits = await eng.forwardToken(prompt[prompt.length - 1]);
    next = argmax(logits); const spec = [next]; const ts0 = performance.now();
    while (spec.length < N) { for (const t of await eng.specStep(next, argmax, K)) spec.push(t); next = spec[spec.length - 1]; }
    const sts = (spec.length - 1) / ((performance.now() - ts0) / 1000), same = gen.every((t, i) => spec[i] === t), st = eng.mtp.stats;
    console.log(`  spec K=${K}: ${sts.toFixed(2)} tok/s, acceptance ${st.accepted}/${st.drafts}, ${same ? "identical to plain" : "DIFFERS from plain"}`);
    if (!same) fail++;
  }
}
console.log(fail ? "MOE FAIL" : "MOE PASS ✓"); if (fail) Deno.exit(1);
