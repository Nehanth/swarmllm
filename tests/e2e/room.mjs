// Room emulator: real host + worker (+ optional phone-shaped tab) in headless Chromium on one
// machine, real PeerJS signaling and WebRTC, the machine's own GPU, a small model split across
// the tabs. Manual trigger only; nothing runs this automatically.
//
//   npm run e2e -- --phone                 host + worker + phone, wire on (default stripe4)
//   npm run e2e -- --wire off              old PeerJS message path
//   npm run e2e -- --model qwen3-1.7b --prompt "..." --rounds 3
//   npm run e2e -- --phone --model qwen3.8-27b   27B from models/q38/model.gguf, pledges 14+2+0.5 GB
//   npm run e2e -- --devices 16 --phones 8 --model qwen3.8-27b   16 tabs, half phone-shaped, 1 GB / 0.5 GB pledges
//   Signaling runs on a local PeerServer (node_modules/.bin/peerjs) unless --signal cloud.
//
// Re-deal scenarios (roadmap 12): the room comes online, then one event, then the remaining rounds
// must still be answered by a fresh plan with 0 room errors (report: plans before/after, split lines).
//   npm run e2e -- --phone --leave worker --leave-at 1.5    close the worker tab 1.5 s into round 0
//   npm run e2e -- --phone --leave phone --leave-at 1.5     same for the phone-shaped tab
//   npm run e2e -- --phone --join-after 2                   a desktop tab ('late-e2e', 1 GB) joins 2 s after online
//   npm run e2e -- --phone --redeal                         press the host's re-deal button
//   npm run e2e -- --model qwen3.8-27b --leave phone --leave-at 5      the 27B (spec decode) survives a leave (~7 min)
//   npm run e2e -- --model qwen3.8-27b --host-gb 14 --leave worker --expect-waiting   room too small after the leave
//   npm run e2e -- --phone --leave host --leave-at 1.5 --expect-over   the host tab dies mid-answer: every other
//                                                            tab must reach body[data-state]="over" with Send enabled
//   npm run e2e -- --phone --redeal --verbose     also print every tab's room log
// --leave takes a tab key (worker, worker2, phone, …; host only with --expect-over, the room is over then).
// With --leave the host pledge grows by the departed pledge so the survivors still hold the model.
// Desktop tabs share one browser context and therefore one Cache API, so a late desktop joiner is
// artificially warm.
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
const PORT = +arg("port", 8123);
// --devices N: total tabs including the host; --phones K: how many of the joiners are phone-shaped
const DEVICES = Math.max(2, +arg("devices", flag("phone") ? 3 : 2));
const PHONES = Math.min(DEVICES - 1, +arg("phones", flag("phone") ? 1 : 0));
const PHONE = PHONES > 0;
// signaling: our own PeerServer on this machine (default), or the public PeerJS cloud with --signal cloud
const SIGNAL_PORT = +arg("signal-port", 9000);
const CLOUD = arg("signal", "local") === "cloud";
// pledges in GB: host,worker,phone. The 27B needs 16.5 GB in the room.
// host pledge: whatever the joiners (1 GB desktop, 0.5 GB phone) leave of the model's need, at least 2 GB
const NEED = { "qwen3.8-27b": 16.5, "qwen3-4b": 4.6, "qwen3-1.7b": 2.0, "qwen3-0.6b": 0.8 }[MODEL] || 2;
// re-deal scenarios: --leave <tab> --leave-at <s> (close that tab s seconds into round 0),
// --join-after <s> (one more desktop tab joins s seconds after online), --redeal (host button),
// --expect-waiting (the run passes when the room ends up waiting for a device)
const LEAVE = arg("leave"), LEAVE_AT = +arg("leave-at", 1.5), JOIN_AFTER = arg("join-after"), REDEAL = flag("redeal"), EXPECT_WAITING = flag("expect-waiting"), EXPECT_OVER = flag("expect-over");
if (LEAVE === "host" && !EXPECT_OVER) { console.error("--leave host: when the host leaves the room is over; pass --expect-over to test that, or pick a worker or phone tab"); process.exit(2); }
const LEAVE_GB = LEAVE ? (LEAVE === "host" ? 0 : LEAVE.startsWith("phone") ? 0.5 : 1) : 0;
const HOST_GB = arg("host-gb", String(Math.max(2, Math.ceil(NEED + 0.5 - (DEVICES - 1 - PHONES) * 1 - PHONES * 0.5 + LEAVE_GB))));
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
const BASE = `http://127.0.0.1:${PORT}/p2p.html?wire=${WIRE}` + (CLOUD ? "" : `&signal=127.0.0.1:${SIGNAL_PORT}`);
import { spawn } from "child_process";
let peerServer = null;
if (!CLOUD) {
  peerServer = spawn(path.join(ROOT, "node_modules/.bin/peerjs"), ["--port", String(SIGNAL_PORT), "--path", "/"], { stdio: "ignore" });
  await new Promise((r) => setTimeout(r, 1500));
}

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
const tabs = { host: await ctx.newPage() };
const nDesk = DEVICES - 1 - PHONES;
for (let i = 0; i < nDesk; i++) tabs[nDesk === 1 ? "worker" : "worker" + (i + 1)] = await ctx.newPage();
let pctx = null;
if (PHONE) {
  pctx = await browser.newContext({ userAgent: UA_PHONE, viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, ignoreHTTPSErrors: true });
  if (!flag("no-local-weights")) await localWeights(pctx);
  for (let i = 0; i < PHONES; i++) {
    const pg = await pctx.newPage(); tabs[PHONES === 1 ? "phone" : "phone" + (i + 1)] = pg;
    const cdp = await pctx.newCDPSession(pg);
    await cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 });
  }
}
const errs = {};
// console errors and page errors per tab; a closed tab keeps its list (it counts too)
function attach(name, p) {
  errs[name] = errs[name] || [];
  p.on("console", (m) => { if (m.type() === "error") errs[name].push(m.text().slice(0, 200)); });
  p.on("pageerror", (e) => errs[name].push("pageerror: " + String(e).slice(0, 200)));
}
for (const [name, p] of Object.entries(tabs)) attach(name, p);
// open one more desktop tab and join the room (the late joiner of --join-after)
async function joinTab(name, gb, code) {
  const pg = await ctx.newPage(); attach(name, pg);
  await pg.goto(BASE);
  await pg.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
  await pg.fill("#name-input", name + "-e2e"); await pg.fill("#join-gb", gb); await pg.fill("#code-input", code);
  await pg.click("#join-btn");
  return pg;
}
const t0 = Date.now(); const T = () => ((Date.now() - t0) / 1000).toFixed(1) + "s";
const log = (...a) => console.error(T(), ...a);
const status = (p) => p.evaluate(() => [document.getElementById("ai-status").textContent, document.getElementById("ldg-sub").textContent, document.getElementById("ldg-fill").style.width].join(" | "));
const lastLog = (p, re) => p.evaluate((src) => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent).filter((t) => new RegExp(src).test(t)).slice(-1)[0] || "", re.source);
// a round ends with "ready — prefill …" (answered), "stopped: X left …" (a device left mid-answer) or "generation failed: …"
const ROUND_RE = /^ready — prefill|^generation failed|^stopped:/;
// a "stopped:" status is overwritten by the re-deal line ~300 ms later, so the page records it
const hookStatus = () => tabs.host.evaluate(() => { const el = document.getElementById("ai-status"); new MutationObserver(() => { if (/^stopped:/.test(el.textContent)) window.__stop = el.textContent; }).observe(el, { childList: true, characterData: true, subtree: true }); });
const ask = async () => { await tabs.host.evaluate(() => { window.__stop = null; }); await tabs.host.fill("#ai-prompt", PROMPT); await tabs.host.click("#ai-send"); };
const waitRound = () => tabs.host.waitForFunction((src) => window.__stop || new RegExp(src).test(document.getElementById("ai-status").textContent), ROUND_RE.source, { timeout: 300000 });
const roundResult = async (phase) => {
  const st = await tabs.host.evaluate(() => window.__stop || document.getElementById("ai-status").textContent);
  const reply = await tabs.host.evaluate(() => { const b = document.querySelectorAll(".m.bot .bubble"); return (b[b.length - 1]?.textContent || "").slice(0, 240); });
  return { phase, status: st, reply };
};

try { main: {
  if (LEAVE && !tabs[LEAVE]) throw new Error(`--leave ${LEAVE}: no such tab (have ${Object.keys(tabs).join(", ")})`);
  for (const p of Object.values(tabs)) await p.goto(BASE);
  for (const p of Object.values(tabs)) await p.waitForFunction(() => document.getElementById("join-gb").value !== "", null, { timeout: 60000 });
  for (const [name, p] of Object.entries(tabs)) { await p.fill("#name-input", name + "-e2e"); await p.fill("#join-gb", name === "host" ? HOST_GB : name.startsWith("phone") ? "0.5" : "1"); }
  if (PHONE) log("phone tabs:", PHONES, "iPhone UA:", await Object.values(tabs).filter((_, i) => i === Object.keys(tabs).findIndex((k) => k.startsWith("phone")))[0].evaluate(() => navigator.userAgent.includes("iPhone")));
  await tabs.host.click("#create-btn");
  await tabs.host.waitForFunction(() => /[A-Z0-9]{4}/.test(document.getElementById("side-code").textContent), null, { timeout: 30000 });
  const code = (await tabs.host.textContent("#side-code")).trim().match(/[A-Z0-9]{4}/)[0];
  log("room", code);
  for (const name of Object.keys(tabs).filter((k) => k !== "host")) { await tabs[name].fill("#code-input", code); await tabs[name].click("#join-btn"); await tabs[name].waitForTimeout(150); }
  const N = Object.keys(tabs).length;
  for (const p of Object.values(tabs)) await p.waitForFunction((n) => document.querySelectorAll(".peer-card").length >= n, N, { timeout: 60000 + 2000 * N });
  log(N, "devices in room");
  await tabs.host.waitForTimeout(3000);   // stripe connections
  await tabs.host.selectOption("#ai-model", MODEL);
  await tabs.host.click("#ai-start");
  log("model start pressed");
  const poll = setInterval(async () => { try { for (const [n, p] of Object.entries(tabs).slice(0, 4)) log(n + ":", (await status(p)).slice(0, 120)); } catch {} }, 15000);
  // fail fast: a tab that reports a load failure means the room can never come online
  const failed = setInterval(async () => { try { for (const [n, p] of Object.entries(tabs)) { const st = await p.evaluate(() => document.getElementById("ai-status").textContent); if (/^failed:/.test(st)) { clearInterval(failed); throw new Error(`${n} ${st}`); } } } catch (e) { if (String(e).includes("failed:")) { console.error("FAILED:", String(e).slice(0, 300)); process.exit(2); } } }, 5000);
  for (const [name, p] of Object.entries(tabs)) await p.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online"), null, { timeout: name === "host" ? 900000 : 300000 });
  clearInterval(failed);
  clearInterval(poll);
  log("online:", await tabs.host.textContent("#ai-status"));
  await hookStatus();
  const splitBefore = await lastLog(tabs.host, /layer split/);
  const planBefore = await tabs.host.evaluate(() => +document.body.dataset.plan);
  log(splitBefore, "(plan", planBefore + ")");

  const results = [];
  let event = null, tEvent = 0, r = 0, splitAfter = "", note = "", planAfter = planBefore, unlocked = null, lateOnline = null, waiting = false;
  if (LEAVE === "host") {
    // the host tab dies mid-answer: every other tab must notice on its own (ICE failed or 7.5 s
    // of host silence; PeerJS' own close comes 15-30 s later), end the answer, unlock Send and
    // show "this room is over" (body[data-state]="over")
    await ask();
    await tabs.host.waitForTimeout(LEAVE_AT * 1000);
    const st0 = await tabs.host.textContent("#ai-status");
    const inFlight = /^prefill|^generating/.test(st0);
    const tLeave = Date.now();
    await tabs.host.close(); delete tabs.host;
    log("closed host", inFlight ? "mid-answer" : `between answers (${st0.slice(0, 60)})`);
    const over = {};
    for (const [n, p] of Object.entries(tabs)) {
      try { await p.waitForFunction(() => document.body.dataset.state === "over", null, { timeout: 30000 }); } catch {}
      over[n] = await p.evaluate(() => ({ state: document.body.dataset.state, sendEnabled: !document.getElementById("ai-send").disabled, status: document.getElementById("ai-status").textContent.slice(0, 120), bubble: (([...document.querySelectorAll(".m.bot .stats")].pop() || {}).textContent || "").slice(0, 80) }));
      over[n].afterMs = Date.now() - tLeave;
      log(n + ":", JSON.stringify(over[n]));
    }
    if (flag("verbose")) for (const [n, p] of Object.entries(tabs)) console.error(n + " log:\n  " + (await p.evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent.slice(0, 200)))).join("\n  "));
    const roomErrs = {}; for (const [n, p] of Object.entries(tabs)) roomErrs[n] = await p.evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent).filter((t) => t.includes("\u26a0")).map((t) => t.slice(0, 160)));
    const nRoomErrs = Object.values(roomErrs).reduce((a, e) => a + e.length, 0);
    const ok = Object.values(over).every((o) => o.state === "over" && o.sendEnabled) && Object.values(errs).every((e) => e.length === 0) && nRoomErrs === 0;
    console.log(JSON.stringify({ ok, wire: WIRE, model: MODEL, devices: DEVICES, phones: PHONES, code, event: { kind: "leave", tab: "host", at: LEAVE_AT, inFlight }, over, errors: Object.fromEntries(Object.entries(errs).filter(([, v]) => v.length)), roomErrors: { count: nRoomErrs, first: roomErrs } }, null, 1));
    process.exitCode = ok ? 0 : 1;
    break main;
  }
  if (LEAVE) {
    // round 0 starts, then the tab closes LEAVE_AT seconds later (mid-answer, unless the
    // answer was already over); the host must report "stopped: X left" within ~2 s
    await ask();
    await tabs.host.waitForTimeout(LEAVE_AT * 1000);
    const st0 = await tabs.host.textContent("#ai-status");
    const inFlight = /^prefill|^generating/.test(st0);
    const tLeave = Date.now();
    await tabs[LEAVE].close(); delete tabs[LEAVE];
    log("closed", LEAVE, inFlight ? "mid-answer" : `between answers (${st0.slice(0, 60)})`);
    await waitRound();
    const res = await roundResult(inFlight ? "interrupted" : "before");
    log("round 0", res.status);
    results.push(res); r = 1;
    event = { kind: "leave", tab: LEAVE, at: LEAVE_AT, inFlight, stopAfterMs: inFlight ? Date.now() - tLeave : null };
    tEvent = tLeave;
  }
  if (JOIN_AFTER !== undefined) {
    await tabs.host.waitForTimeout(+JOIN_AFTER * 1000);
    tEvent = Date.now();
    tabs.late = await joinTab("late", "1", code);
    await tabs.host.waitForFunction(() => [...document.querySelectorAll("#chat-log div")].some((d) => /late-e2e joined/.test(d.textContent)), null, { timeout: 60000 });
    log("late-e2e joined");
    event = { kind: "join", tab: "late", at: +JOIN_AFTER };
  }
  if (REDEAL) {
    tEvent = Date.now();
    await tabs.host.click("#ai-redeal");
    log("re-deal pressed");
    event = { kind: "redeal", tab: "host", at: 0 };
  }
  if (event) {
    // the host bumps body[data-plan] when a new plan is dealt and body[data-state] goes back to
    // "online" once every device reported ready for it ("waiting" = the room can no longer hold the model)
    const h = await tabs.host.waitForFunction((p) => { const s = document.body.dataset.state; return +document.body.dataset.plan > p && (s === "online" || s === "waiting") ? s : false; }, planBefore, { timeout: 600000 });
    const state = await h.jsonValue();
    event.redealMs = Date.now() - tEvent;
    planAfter = await tabs.host.evaluate(() => +document.body.dataset.plan);
    splitAfter = await lastLog(tabs.host, /layer split/);
    note = await lastLog(tabs.host, /re-dealing:/);
    log("plan", planBefore, "->", planAfter, state, `(${event.redealMs} ms)`);
    log(splitAfter); if (note) log(note);
    if (state === "waiting") {
      waiting = true;
      const st = await tabs.host.textContent("#ai-status");
      if (!EXPECT_WAITING) throw new Error(`the room went to "waiting for a device" after the ${event.kind} (${st}); pass --expect-waiting if that is the point of this run`);
      log("room is waiting for a device, as expected:", st);
    }
    if (JOIN_AFTER !== undefined && !waiting) {
      await tabs.late.waitForFunction(() => document.getElementById("ai-panel").classList.contains("online"), null, { timeout: 300000 });
      const st = await tabs.late.textContent("#ai-status");
      lateOnline = /serving layers \d+–\d+/.test(st);
      log("late:", st);
    }
    unlocked = {}; for (const [n, p] of Object.entries(tabs)) unlocked[n] = await p.evaluate(() => !document.getElementById("ai-send").disabled);
  }
  if (!waiting) for (; r < ROUNDS; r++) {
    await ask();
    await waitRound();
    const res = await roundResult(event ? "after" : "plain");
    log("round", r, res.status);
    results.push(res);
    await tabs.host.waitForTimeout(1000);
  }
  const wire = {}; for (const [n, p] of Object.entries(tabs)) wire[n] = await p.evaluate(() => window.swarmDebug?.());
  // every tab's final status line (workers: "serving layers a–b"), and with --verbose its room log
  const serving = {}; for (const [n, p] of Object.entries(tabs)) serving[n] = await p.evaluate(() => { const s = window.swarmPlan?.() || {}; return { status: document.getElementById("ai-status").textContent.slice(0, 120), v: s.v, range: s.range, engineLayers: s.engineLayers, next: s.next }; });
  if (flag("verbose")) for (const [n, p] of Object.entries(tabs)) console.error(n + " log:\n  " + (await p.evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent.slice(0, 200)))).join("\n  "));
  // the room's own log carries GPU validation errors that never reach the console
  const roomErrs = {}; for (const [n, p] of Object.entries(tabs)) roomErrs[n] = await p.evaluate(() => [...document.querySelectorAll("#chat-log div")].map((d) => d.textContent).filter((t) => t.includes("⚠")).map((t) => t.slice(0, 160)));
  const nRoomErrs = Object.values(roomErrs).reduce((a, e) => a + e.length, 0);
  // a reply that is mostly symbols means a device served the wrong weights or a frame was garbled:
  // the round "completed" but the pipeline is broken (seen once after a re-deal: "﹤﹤﹤`;`;`;")
  const garbled = (t) => { const w = (t.match(/[A-Za-z]{3,}/g) || []).length; return t.length > 20 && w < t.length / 40; };
  for (const s of results) if (s.phase !== "interrupted" && garbled(s.reply)) s.garbled = true;
  const ok = results.every((s) => s.phase === "interrupted" ? s.status.startsWith("stopped:") : s.status.startsWith("ready") && !s.garbled)
    && Object.values(errs).every((e) => e.length === 0) && nRoomErrs === 0
    && (JOIN_AFTER === undefined || waiting || lateOnline === true)
    && (!EXPECT_WAITING || waiting)
    && (!unlocked || Object.values(unlocked).every(Boolean))
    && (!event || planAfter > planBefore);
  const linkSummary = Object.fromEntries(Object.entries(wire).map(([n, l]) => [n, (l || []).map((x) => `${x.name}:${x.chans}ch ${x.sent}/${x.recv}`).join(", ")]));
  console.log(JSON.stringify({ ok, wire: WIRE, model: MODEL, devices: DEVICES, phones: PHONES, code, event, plans: { before: planBefore, after: planAfter }, splitBefore, splitAfter, note, unlocked, waiting, serving, results, links: DEVICES > 6 ? Object.fromEntries(Object.entries(linkSummary).slice(0, 4)) : linkSummary, errors: Object.fromEntries(Object.entries(errs).filter(([, v]) => v.length)), roomErrors: { count: nRoomErrs, first: Object.fromEntries(Object.entries(roomErrs).filter(([, v]) => v.length).map(([k, v]) => [k, v.slice(0, 2)]).slice(0, 3)) } }, null, 1));
  process.exitCode = ok ? 0 : 1;
} } catch (e) {
  console.error("FAILED:", String(e).slice(0, 400));
  for (const [n, p] of Object.entries(tabs)) { try { console.error(n + ":", await status(p)); } catch {} }
  console.error("errors:", JSON.stringify(errs));
  process.exitCode = 2;
} finally {
  await browser.close(); srv.close(); wsrv.close(); fs.rmSync(tlsDir, { recursive: true, force: true }); if (peerServer) peerServer.kill();
}
