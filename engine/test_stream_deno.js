import { parseGGUFHeader, q4Repack, q8Repack, streamEntryToGPU, GGML_Q4_0, GGML_Q8_0 } from "./gguf.js";
const path = Deno.env.get("GGUF") || "qwen/model.gguf";
const fh = await Deno.open(path);
const readAt = async (off, len) => { const out = new Uint8Array(len); let got = 0; while (got < len) { const n = await fh.read(out.subarray(got)); if (n === null) break; got += n; } return out; };
const hdrBuf = await (async () => { await fh.seek(0, Deno.SeekMode.Start); return readAt(0, 64 << 20); })();
const G = parseGGUFHeader(hdrBuf.buffer, { skipTokenizer: true });
const slice = async (info) => { await fh.seek(info.byteOffset, Deno.SeekMode.Start); return readAt(info.byteOffset, info.byteLength); };
const adapter = await navigator.gpu.requestAdapter(); const device = await adapter.requestDevice();
const openRange = async (info) => new Response(new Blob([await slice(info)]).stream(), { status: 206 });
const readBack = async (buf, n) => {
  const rb = device.createBuffer({ size: buf.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(buf, 0, rb, 0, buf.size); device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ); const out = new Uint8Array(rb.getMappedRange().slice(0, n)); rb.unmap(); return out;
};
let tested = 0;
for (const [name, info] of Object.entries(G.tensors)) {
  if (info.shape.length !== 2 || (info.ggmlType !== GGML_Q4_0 && info.ggmlType !== GGML_Q8_0)) continue;
  if (info.byteLength > 120 * 2 ** 20) continue;   // readback buffer must fit default limits
  const ref = (info.ggmlType === GGML_Q4_0 ? q4Repack : q8Repack)(info, await slice(info));
  const e = await streamEntryToGPU(device, info, openRange, { staging: 1 << 20 });
  const qs = await readBack(e.gpu.qs, ref.qs.byteLength);
  const sc = new Float32Array((await readBack(e.gpu.sc, ref.scales.byteLength)).buffer);
  let bad = 0; for (let i = 0; i < ref.qs.length; i++) if (qs[i] !== ref.qs[i]) { bad++; break; }
  for (let i = 0; i < ref.scales.length; i++) if (sc[i] !== ref.scales[i]) { bad++; break; }
  if (bad) { console.log("MISMATCH", name); Deno.exit(1); }
  e.gpu.qs.destroy(); e.gpu.sc.destroy();
  if (++tested >= 12) break;
}
console.log("STREAM PASS ✓", tested, "tensors bit-identical");
