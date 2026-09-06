// Room emulator: real host + worker (+ optional phone-shaped tab) in headless Chromium on one
// machine, real PeerJS signaling and WebRTC, the machine's own GPU, a small model split across
// the tabs. Manual trigger only; nothing runs this automatically.
//
//   npm run e2e -- --phone                 host + worker + phone, wire on (default stripe4)
//   npm run e2e -- --wire off              old PeerJS message path
//   npm run e2e -- --model qwen3-1.7b --prompt "..." --rounds 3
//   npm run e2e -- --phone --model qwen3.8-27b   27B from models/q38/model.gguf, pledges 14+2+0.5 GB
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
// pledges in GB: host,worker,phone. The 27B needs 16.5 GB in the room.
const PLEDGES = (arg("pledges", MODEL === "qwen3.8-27b" ? "14,2,0.5" : "2,1,0.5")).split(",");
// GGUF files already on this machine stand in for Hugging Face (Range requests served from disk)
const LOCAL = { "Qwen3.8-27B-Q4_0.gguf": "models/q38/model.gguf", "Qwen3-0.6B-Q8_0.gguf": "models/qwen/model.gguf", "Qwen3-1.7B-Q8_0.gguf": "models/qwen17/model.gguf" };
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
// Local weight server: https (Playwright can only rewrite a request to the same protocol as the
// original, and Hugging Face is https), self-signed cert, Range requests streamed from disk.
import https from "https";
import { execSync } from "child_process";
import os from "os";
const TLS_PORT = PORT + 1;
const tlsDir = fs.mkdtempSync(path.join(os.tmpdir(), "swarm-e2e-"));
execSync(`openssl req -x509 -newkey rsa:2048 -nodes -keyout ${tlsDir}/k.pem -out ${tlsDir}/c.pem -days 2 -subj /CN=127.0.0.1 2>/dev/null`);
const wsrv = https.createServer({ key: fs.readFileSync(`${tlsDir}/k.pem`), cert: fs.readFileSync(`${tlsDir}/c.pem`) }, (q, r) => {
  const p = path.join(ROOT, decodeURIComponent(q.url.split("?")[0]));
  if (!p.startsWith(ROOT) || !fs.existsSync(p)) { r.statusCode = 404; r.end(); return; }
  const size = fs.statSync(p).size, m = /bytes=(\d+)-(\d*)/.exec(q.headers.range || "");
  const lo = m ? +m[1] : 0, hi = m && m[2] ? Math.min(+m[2], size - 1) : size - 1;
  r.writeHead(m ? 206 : 200, { "content-type": "application/octet-stream", "content-range": `bytes ${lo}-${hi}/${size}`, "accept-ranges": "bytes", "content-length": String(hi - lo + 1), "access-control-allow-origin": "*", "access-control-expose-headers": "content-range, content-length, accept-ranges" });
  fs.createReadStream(p, { start: lo, end: hi }).pipe(r);
}).listen(TLS_PORT, "127.0.0.1");
async function localWeights(context) {
  await context.route("**/*.gguf", (route) => {
    const name = route.request().url().split("/").pop().split("?")[0];
    const file = LOCAL[name];
    if (!file || !fs.existsSync(path.join(ROOT, file))) return route.continue();
    return route.continue({ url: `https://127.0.0.1:${TLS_PORT}/${file}` });
  });
}
const ctx = await browser.newContext({ userAgent: UA_DESKTOP, ignoreHTTPSErrors: true });
if (!flag("no-local-weights")) await localWeights(ctx);
const tabs = { host: await ctx.newPage(), worker: await ctx.newPage() };
if (PHONE) {
  const pctx = await browser.newContext({ userAgent: UA_PHONE, viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, ignoreHTTPSErrors: true });
  if (!flag("no-local-weights")) await localWeights(pctx);
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
  const pledges = { host: PLEDGES[0], worker: PLEDGES[1], phone: PLEDGES[2] };
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
  // the room's own log carries GPU validation errors that never reach the console
  const roomErrs = {}; for (const [n, p] of Object.entries(tabs)) roomErrs[n] = await p.evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent).filter((t) => t.includes("\u26a0")).map((t) => t.slice(0, 160)));
  const nRoomErrs = Object.values(roomErrs).reduce((a, e) => a + e.length, 0);
  const ok = results.every((s) => s.startsWith("ready")) && Object.values(errs).every((e) => e.length === 0) && nRoomErrs === 0;
  console.log(JSON.stringify({ ok, wire: WIRE, model: MODEL, phone: PHONE, code, split, results, links: wire, errors: errs, roomErrors: { count: nRoomErrs, first: Object.fromEntries(Object.entries(roomErrs).map(([k, v]) => [k, v.slice(0, 2)])) } }, null, 1));
  process.exitCode = ok ? 0 : 1;
} catch (e) {
  console.error("FAILED:", String(e).slice(0, 400));
  for (const [n, p] of Object.entries(tabs)) { try { console.error(n + ":", await status(p)); } catch {} }
  console.error("errors:", JSON.stringify(errs));
  process.exitCode = 2;
} finally {
  await browser.close(); srv.close(); wsrv.close(); fs.rmSync(tlsDir, { recursive: true, force: true });
}
