const adapter = await navigator.gpu.requestAdapter();
const dev = await adapter.requestDevice();
let lost = false;
dev.lost.then(() => { lost = true; });
const chunk = 512 * 2 ** 20;
const bufs = [];
let total = 0;
const t0 = performance.now();
while (total < 16 * 2 ** 30 && !lost) {
  dev.pushErrorScope("out-of-memory");
  const b = dev.createBuffer({ size: chunk, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  try {
    const enc = dev.createCommandEncoder();
    enc.clearBuffer(b);
    dev.queue.submit([enc.finish()]);
    await dev.queue.onSubmittedWorkDone();
  } catch { lost = true; }
  const err = await dev.popErrorScope().catch(() => true);
  if (err || lost) { try { b.destroy(); } catch {} break; }
  bufs.push(b);
  total += chunk;
}
for (const b of bufs) { try { b.destroy(); } catch {} }
console.log(`measured: ${(total / 2 ** 30).toFixed(1)} GB in ${((performance.now() - t0) / 1000).toFixed(1)}s (lost=${lost})`);
