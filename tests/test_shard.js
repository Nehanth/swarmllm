// Fetch ONLY layers [15,30) from HF via Range requests, run them as a worker
// shard against locally-loaded host shard. Must match golden.
import { parseSafetensors, makeTokenizer, DenseEngine, argmax, fetchModelShard, shardTensorNames } from "../engine/engine.js";
const dir = new URL(".", import.meta.url).pathname;
const cfg = JSON.parse(await Deno.readTextFile(dir + "../models/model/config.json"));
const golden = JSON.parse(await Deno.readTextFile(dir + "./golden/golden.json"));
const tok = makeTokenizer(JSON.parse(await Deno.readTextFile(dir + "../models/model/tokenizer.json")));
const URL_ST = "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/model.safetensors";

const mid = 15, L = cfg.num_hidden_layers;
const names = shardTensorNames(cfg, [mid, L], false, false);
console.log(`fetching ${names.length} tensors for layers [${mid},${L}) via Range…`);
let t0 = performance.now();
const remoteTensors = await fetchModelShard(URL_ST, names, (p) => {});
const mb = Object.values(remoteTensors).reduce((s, t) => s + t.byteLength, 0) / 2 ** 20;
console.log(`fetched ${mb.toFixed(1)} MB in ${((performance.now() - t0) / 1000).toFixed(1)}s (full file is 269 MB)`);

const localTensors = parseSafetensors((await Deno.readFile(dir + "../models/model/model.safetensors")).buffer);
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
const host = await DenseEngine.create({ device, cfg, tensors: localTensors, layerRange: [0, mid] });
const worker = await DenseEngine.create({ device, cfg, tensors: remoteTensors, layerRange: [mid, L], hasEmbed: false, hasHead: false });

let pos = 0, logits = null;
const pipe = async (id) => {
  const h = await worker.runHidden(await host.embedRun(id, pos), pos);
  pos++;
  return await host.headFromHidden(h);
};
for (const id of golden.ids) logits = await pipe(id);
const gen = [];
for (let i = 0; i < golden.generated.length; i++) { const n = argmax(logits); gen.push(n); logits = await pipe(n); }
console.log("generated:", JSON.stringify(tok.decode(gen)));
const ok = JSON.stringify(gen) === JSON.stringify(golden.generated);
console.log(ok ? "SHARD-FETCH PASS ✓" : "SHARD-FETCH FAIL ✗");
Deno.exit(ok ? 0 : 1);
