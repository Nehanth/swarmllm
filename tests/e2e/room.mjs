// Room emulator: real host + worker (+ optional phone-shaped tab) in headless Chromium on one
// machine, real PeerJS signaling and WebRTC, the machine's own GPU, a small model split across
// the tabs. Manual trigger only; nothing runs this automatically.
//
//   npm run e2e -- --phone                 host + worker + phone, wire on (default stripe4)
//   npm run e2e -- --wire off              old PeerJS message path
//   npm run e2e -- --model qwen3-1.7b --prompt "..." --rounds 3
//
// The phone tab gets a mobile user agent (the room then treats it as a phone: 0.5 GB pledge,
// 256 MB buffer cap, no weight cache) and a 4x CPU throttle. Needs `npm install` (playwright)
// and a Chromium that exposes WebGPU on this machine; on Linux/NVIDIA the flags below do.
import { chromium } from "playwright";
import http from "http";
import fs from "fs";
import path from "path";

const arg = (k, d) => { const i = process.argv.indexOf("--" + k); return i > 0 ? process.argv[i + 1] : d; };
const flag = (k) => process.argv.includes("--" + k);
const WIRE = arg("wire", "stripe4"), MODEL = arg("model", "qwen3-0.6b"), ROUNDS = +arg("rounds", 2);
const PROMPT = arg("prompt", "Write three sentences about the ocean.");
const PHONE = flag("phone"), PORT = +arg("port", 8123);
const ROOT = path.resolve(new URL(".", import.meta.url).pathname, "../..");
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png" };

// static server for the repo root (the room is /p2p.html)
const srv = http.createServer((q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { r.statusCode = 404; r.end(); return; }
  r.setHeader("content-type", MIME[path.extname(p)] || "application/octet-stream"); fs.createReadStream(p).pipe(r);
}).listen(PORT, "127.0.0.1");
const BASE = `http://127.0.0.1:${PORT}/p2p.html?wire=${WIRE}`;

const args = ["--no-sandbox", "--headless=new", "--enable-unsafe-webgpu", "--use-gl=angle", "--use-angle=gl-egl", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--allow-loopback-in-peer-connection"];
const browser = await chromium.launch({ headless: false, args });
// Hugging Face answers the HeadlessChrome user agent with an HTML page: use a normal one
const UA_DESKTOP = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";
const UA_PHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1";
const ctx = await browser.newContext({ userAgent: UA_DESKTOP });
const tabs = { host: await ctx.newPage(), worker: await ctx.newPage() };
if (PHONE) {
  const pctx = await browser.newContext({ userAgent: UA_PHONE, viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  tabs.phone = await pctx.newPage();
  const cdp = await pctx.newCDPSession(tabs.phone);
  await cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 });
}
const errs = Object.fromEntries(Object.keys(tabs).map((k) => [k, []]));
for (const [name, p] of Object.entries(tabs)) {
  p.on("console", (m) => { if (m.type() === "error") errs[name].push(m.text().slice(0, 200)); });
  p.on("pageerror", (e) => errs[name].push("pageerror: " + String(e).slice(0, 200)));
}
const t0 = Date.now(); const T = () => ((Date.now() - t0) / 1000).toFixed(1) + "s";
const log = (...a) => console.error(T(), ...a);
const status = (p) => p.evaluate(() => [document.getElementById("ai-status").textContent, document.getElementById("ldg-sub").textContent, document.getElementById("ldg-fill").style.width].join(" | "));

try {
  for (const p of Object.values(tabs)) await p.goto(BASE);
  for (const p of Object.values(tabs)) await p.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
  const pledges = { host: "2", worker: "1", phone: "0.5" };
  for (const [name, p] of Object.entries(tabs)) { await p.fill("#name-input", name + "-e2e"); await p.fill("#join-gb", pledges[name]); }
  if (PHONE) log("phone tab reports:", await tabs.phone.evaluate(() => navigator.userAgent.includes("iPhone") ? "iPhone UA, phone rules apply" : "not a phone"));
  await tabs.host.click("#create-btn");
  await tabs.host.waitForFunction(() => /[A-Z0-9]{4}/.test(document.getElementById("side-code").textContent), null, { timeout: 30000 });
  const code = (await tabs.host.textContent("#side-code")).trim().match(/[A-Z0-9]{4}/)[0];
  log("room", code);
  for (const name of Object.keys(tabs).filter((k) => k !== "host")) { await tabs[name].fill("#code-input", code); await tabs[name].click("#join-btn"); }
  const N = Object.keys(tabs).length;
  for (const p of Object.values(tabs)) await p.waitForFunction((n) => document.querySelectorAll(".peer-card").length >= n, N, { timeout: 60000 });
  log(N, "devices in room");
  await tabs.host.waitForTimeout(3000);   // stripe connections
  await tabs.host.selectOption("#ai-model", MODEL);
  await tabs.host.click("#ai-start");
  log("model start pressed");
  const poll = setInterval(async () => { try { for (const [n, p] of Object.entries(tabs)) log(n + ":", (await status(p)).slice(0, 120)); } catch {} }, 15000);
  for (const [name, p] of Object.entries(tabs)) await p.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online"), null, { timeout: name === "host" ? 900000 : 120000 });
  clearInterval(poll);
  log("online:", await tabs.host.textContent("#ai-status"));
  const split = await tabs.host.evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent).filter((t) => /layer split/.test(t)).slice(-1)[0] || "");
  log(split);
  const results = [];
  for (let r = 0; r < ROUNDS; r++) {
    await tabs.host.fill("#ai-prompt", PROMPT); await tabs.host.click("#ai-send");
    await tabs.host.waitForFunction(() => /^ready — prefill|^generation failed/.test(document.getElementById("ai-status").textContent), null, { timeout: 300000 });
    const st = await tabs.host.textContent("#ai-status"); log("round", r, st); results.push(st);
    await tabs.host.waitForTimeout(1000);
  }
  const wire = {}; for (const [n, p] of Object.entries(tabs)) wire[n] = await p.evaluate(() => window.swarmDebug?.());
  const ok = results.every((s) => s.startsWith("ready")) && Object.values(errs).every((e) => e.length === 0);
  console.log(JSON.stringify({ ok, wire: WIRE, model: MODEL, phone: PHONE, code, split, results, links: wire, errors: errs }, null, 1));
  process.exitCode = ok ? 0 : 1;
} catch (e) {
  console.error("FAILED:", String(e).slice(0, 400));
  for (const [n, p] of Object.entries(tabs)) { try { console.error(n + ":", await status(p)); } catch {} }
  console.error("errors:", JSON.stringify(errs));
  process.exitCode = 2;
} finally {
  await browser.close(); srv.close();
}
