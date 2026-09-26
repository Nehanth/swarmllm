// Run tests/bench/bench.html in Chrome with the real GPU. node tests/bench/chrome_bench.mjs <model path under repo> [tokens]
import { chromium } from "playwright"; import { spawn } from "node:child_process";
const root = new URL("../..", import.meta.url).pathname, model = process.argv[2], N = process.argv[3] || 40;
const GOLD = { "q36moe": ["```python\ndef two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:", "A hash map is a data structure that stores key-value pairs, allowing for efficient retrieval, insertion, and deletion operations. It uses a hash function to compute an index into an array of buckets or slots"] };
const srv = spawn("node", [root + "tests/bench/serve.mjs", root, "8791"], { stdio: "inherit" }); await new Promise((r) => setTimeout(r, 600));
const b = await chromium.launch({ headless: false, args: ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--js-flags=--max-old-space-size=65536"] });
const p = await b.newPage(); p.on("console", (m) => console.log("  tab:", m.text())); p.on("crash", () => console.log("TAB CRASHED"));
const gold = GOLD[Object.keys(GOLD).find((k) => model.includes(k))] || [];
await p.goto(`http://127.0.0.1:8791/tests/bench/bench.html?model=/${model}&tokens=${N}&gold=${encodeURIComponent(JSON.stringify(gold))}`);
await p.waitForFunction(() => window.RESULT, null, { timeout: 30 * 60e3, polling: 2000 }).catch((e) => console.log("timeout", e.message));
await b.close(); srv.kill();
