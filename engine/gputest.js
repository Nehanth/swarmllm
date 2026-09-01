const adapter = await navigator.gpu?.requestAdapter();
if (!adapter) { console.log("NO ADAPTER"); Deno.exit(1); }
const info = adapter.info || {};
console.log("adapter:", info.vendor, info.architecture, info.description || "");
console.log("maxBufferSize:", (adapter.limits.maxBufferSize / 2**30).toFixed(1), "GB");
const dev = await adapter.requestDevice();
console.log("device OK");
