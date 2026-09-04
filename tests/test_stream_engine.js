// browser-like path: streamed q4/q8 uploads + reset() + generation, vs the plain loader
import { Qwen35Engine } from "../engine/qwen35.js";
import { parseGGUFHeader, qwen35Weights, tokenizerFromGGUF, streamEntryToGPU, gpuUploadEntry, GGML_EMBED } from "../engine/gguf.js";
import { makeTokenizer, argmax } from "../engine/engine.js";
const fh = await Deno.open("../models/q38/model.gguf");
const readAt = async (off, len) => { await fh.seek(off, Deno.SeekMode.Start); const out = new Uint8Array(len); let got = 0; while (got < len) { const n = await fh.read(out.subarray(got)); if (n === null) break; got += n; } return out; };
const G = parseGGUFHeader((await readAt(0, 64 << 20)).buffer);
const tok = makeTokenizer(tokenizerFromGGUF(G.meta));
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
device.addEventListener("uncapturederror", (ev) => { console.log("GPU ERROR:", ev.error.message.slice(0, 300)); });
const L = 5, opts = { lo: 0, hi: L, hasEmbed: true, hasHead: true };
const bytesOf = (info) => readAt(info.byteOffset, info.byteLength);
const run = async (eng, text) => { const ids = tok.encode(text); const out = []; for (const id of ids) out.push(argmax(await eng.forwardToken(id))); return out.join(","); };
// plain
const W1 = await qwen35Weights(G, bytesOf, opts);
const e1 = await Qwen35Engine.create({ device, meta: G.meta, weights: W1, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 64 });
const a = await run(e1, "Write the Python code for Two Sum.");
// streamed, like p2p.html
const G2 = { ...G, streamEntry: (info) => streamEntryToGPU(device, info, async (i) => new Response(new Blob([await readAt(i.byteOffset, i.byteLength)]).stream(), { status: 206 }), { staging: 8 << 20 }) };
const W2 = await qwen35Weights(G2, bytesOf, opts, () => {}, (e, name) => gpuUploadEntry(device, e, name === GGML_EMBED));
const e2 = await Qwen35Engine.create({ device, meta: G.meta, weights: W2, layerRange: [0, L], hasEmbed: true, hasHead: true, maxSeq: 64, vocab: G.tensors[GGML_EMBED].shape[0] });
e2.reset();
const b = await run(e2, "Write the Python code for Two Sum.");
await run(e2, "junk"); e2.reset();
const c = await run(e2, "Write the Python code for Two Sum.");
console.log(a === b && b === c ? "STREAM ENGINE PASS ✓" : "STREAM ENGINE FAIL\n" + a + "\n" + b + "\n" + c);
