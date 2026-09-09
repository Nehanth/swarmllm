// SwarmLLM room: signaling, WebRTC mesh, layer assignment, weight streaming and the
// generation loop (prefill, decode, speculative verify). Served with p2p.html at /room.
import { autotuneCoop, makeTokenizer, DenseEngine, argmax, fetchModelShard, shardTensorNames, gpuSelfTest, kernelMicroTests }
  from "./engine/engine.js";
import { f32ToF16, f16ToF32, parseGGUFHeader, ggufWeights, ggufShardBytes, GGML_EMBED, GGML_OUTPUT, GGML_FINAL_NORM,
  ggmlLayerNames, qwen35Weights, qwen35ShardBytes, qwen35MtpBytes, qwen35LayerNames, tokenizerFromGGUF, gpuUploadEntry, streamEntryToGPU }
  from "./engine/gguf.js";
import { Qwen35Engine } from "./engine/qwen35.js";
import { WIRE_F16, badF32, f32ToB64, packF16, unpackF16, asU16, packWire, unpackWire, asF32, b64ToF32 } from "./room/wire.js";
import { esc, md } from "./room/markdown.js";
import { aiSample } from "./room/sampling.js";
import { MODELS, NEED_GB, MAX_SEQ, MAX_NEW, MIN_ROOM } from "./room/models.js";
import { makeLink, attachWire, wireReady, sendFrame } from "./room/transport.js";
import { planSplit, describeSplit, planNote } from "./room/plan.js";

// Hidden-state transport (room/transport.js). ?wire=off falls back to PeerJS messages;
// ?wire=slice uses one sliced channel; ?wire=stripeN spreads slices over N peer connections.
const WIRE = (new URLSearchParams(location.search).get("wire") || "stripe4").toLowerCase();
const WIRE_STRIPES = WIRE === "off" ? 0 : WIRE.startsWith("stripe") ? Math.max(1, Math.min(8, parseInt(WIRE.slice(6), 10) || 1)) : 1;
// Signaling: ?signal=host:port points PeerJS at our own PeerServer (the emulator and big
// rooms use one); default is the public PeerJS cloud.
const SIGNAL = new URLSearchParams(location.search).get("signal");
const SIGNAL_OPTS = SIGNAL ? (() => { const [host, port] = SIGNAL.split(":"); return { host, port: +port || 443, path: "/", secure: location.protocol === "https:" }; })() : {};

// Topology: every device keeps ONE link to the host (control, roster, tokens). Data links
// between chain neighbours open when the layers are dealt (ensureLink), so a room of N
// devices has N-1 host links plus N-1 chain links, not N*(N-1)/2. Workers learn about the
// other devices from the host's roster message and draw cards from it.
const members = new Map();   // id -> { name, meta } for everyone in the room except me
const cards = new Map();     // id -> card element

const $ = (id) => document.getElementById(id);
function toast(text) {
  const t = document.createElement("div");
  t.className = "toast";
  t.textContent = text;
  $("toasts").appendChild(t);
  setTimeout(() => t.remove(), 4200);
}
function mascot() {}
const PREFIX = "swarmllm-room-";
const rand = (n) => Array.from(crypto.getRandomValues(new Uint8Array(n)))
  .map(b => "ABCDEFGHJKMNPQRSTVWXYZ23456789"[b % 30]).join("");

// per-tab id, sent in hello: a reload keeps it (sessionStorage), a second device never shares it,
// so the host can tell a reloaded tab from a namesake that is still alive
const TAB = (() => { try { return sessionStorage.swarmTab ||= crypto.randomUUID(); } catch { return crypto.randomUUID(); } })();
let peer = null;          // my PeerJS peer
let isHost = false;
let roomCode = null;
let myName = null;
let myMeta = {};
// conns: peerId -> { conn, name, meta, rtt, mbps, card }
const conns = new Map();
// host only: roster of member peer ids -> {name, meta}
const roster = new Map();

// --- GPU capability probe (runs at page load so the join screen can offer
// contribution presets) ---
async function probeGPU() {
  const meta = { ua: navigator.userAgent.includes("iPhone") ? "iPhone" :
                     navigator.userAgent.includes("Mac") ? "Mac" :
                     navigator.userAgent.includes("Android") ? "Android" : "Device",
                 webgpu: false, gpu: "no WebGPU", maxBufGB: 0 };
  if (navigator.gpu) {
    try {
      const a = await navigator.gpu.requestAdapter();
      if (a) {
        meta.webgpu = true;
        const info = a.info || {};
        meta.gpu = [...new Set([info.vendor, info.architecture || info.device].filter(Boolean))].join(" ") || "GPU";
        meta.maxBufGB = +(a.limits.maxBufferSize / 2 ** 30).toFixed(1);
        // browsers hide real GPU memory (fingerprinting). Default to the
        // conservative per-buffer limit; the user can opt in to a real
        // measurement (see measureBudgetGB) which replaces this estimate.
        meta.budgetGB = meta.maxBufGB;
        meta.canMeasure = meta.ua !== "iPhone" && meta.ua !== "Android";
      }
    } catch {}
  }
  return meta;
}

async function measureBudgetGB(adapter, capGB) {
  try {
    const dev = await adapter.requestDevice();
    let lost = false;
    dev.lost.then(() => { lost = true; });
    const chunk = 512 * 2 ** 20;
    const bufs = [];
    let total = 0;
    while (total < capGB * 2 ** 30 && !lost) {
      dev.pushErrorScope("out-of-memory");
      const b = dev.createBuffer({ size: chunk, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      try { // commit the pages for real, or lazy allocation lies to us
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
    try { dev.destroy(); } catch {}
    return +(total / 2 ** 30).toFixed(1);
  } catch { return 0; }
}

// probe once at load; fill the contribution selector
const metaPromise = (async () => {
  const m = await probeGPU();
  if (m.webgpu && m.budgetGB) m.contribGB = Math.max(0.2, Math.round(m.budgetGB * 0.5 * 10) / 10);
  m.phone = m.ua === "iPhone" || m.ua === "Android";
  if (m.phone) { m.contribGB = 0.5; $("join-gb").min = "0.5"; $("join-gb").step = "0.5"; }
  else if (m.contribGB) m.contribGB = Math.max(1, Math.round(m.contribGB));
  if (m.contribGB) $("join-gb").value = m.contribGB;
  return m;
})();

// --- UI helpers ---
function log(from, text) {
  const div = document.createElement("div");
  div.innerHTML = `<b></b> `;
  div.querySelector("b").textContent = from;
  div.appendChild(document.createTextNode(text));
  $("chat-log").appendChild(div);
  $("chat-log").scrollTop = $("chat-log").scrollHeight;
}

function peerCard(id, name, meta, self) {
  const card = document.createElement("div");
  card.className = "peer-card" + (self ? " self" : "");
  card.innerHTML = `
    <div class="peer-name"><span class="dot ${self ? "ok" : "warn"}"></span><span class="pname"></span></div>
    <div class="peer-gpu"></div>
    <div class="peer-stats">
      <span>rtt <b class="rtt">—</b></span>
      <span>bw <b class="bw">—</b></span>
      <span>buf <b class="buf">—</b></span>
    </div>
    ${self ? "" : '<button class="bw-btn">test bandwidth</button>'}`;
  card.querySelector(".pname").textContent = name + (self ? " (you)" : "");
  card.querySelector(".peer-gpu").textContent = meta.webgpu
    ? `${meta.ua} · ${meta.gpu}` : `${meta.ua} · ⚠ no WebGPU`;
  const budget = meta.budgetGB || meta.maxBufGB;
  card.querySelector(".buf").textContent = meta.contribGB ? "gives " + meta.contribGB + " GB" : (budget ? budget + " GB" : "—");
  $("peers").appendChild(card);
  if (!self) card.querySelector(".bw-btn").addEventListener("click", () => bwTest(id));
  return card;
}

let wasReady = false;
function updateNeed(pledged) {
  const need = NEED_GB[$("ai-model").value] || 1;
  const ok = pledged >= need;
  $("need-fill").style.width = Math.min(100, pledged / need * 100).toFixed(1) + "%";
  $("need-text").textContent = ok
    ? `needs ~${need} GB \u00b7 room gives ${pledged.toFixed(1)} GB \u00b7 ready`
    : `needs ~${need} GB \u00b7 room gives ${pledged.toFixed(1)} GB \u00b7 add ${(need - pledged).toFixed(1)} GB more`;
  $("ai-need").classList.toggle("ok", ok);
  if (!ai.busy && !ai.engine && !ai.hostId && ai.state === "idle") $("ai-start").disabled = !ok;   // a newcomer in a running room cannot start a second model
  if (ok && !wasReady) { $("ai-start").classList.remove("unlocked"); void $("ai-start").offsetWidth; $("ai-start").classList.add("unlocked"); }
  wasReady = ok;
}
$("ai-model").addEventListener("change", () => updateCluster());
function updateCluster() {
  const all = [myMeta, ...[...members.values()].map(m => m.meta)];
  const gpus = all.filter(m => m && m.webgpu).length;
  const pledged = all.reduce((s, m) => s + (m?.contribGB || 0), 0);
  updateNeed(pledged);
  const mem = all.reduce((s, m) => s + (m?.budgetGB || m?.maxBufGB || 0), 0);
  $("cluster-summary").textContent =
    `${all.length} device${all.length > 1 ? "s" : ""} \u00b7 ${gpus} WebGPU \u00b7 ${pledged.toFixed(1)} GB pledged`;
}

function enterRoom() {
  $("join-screen").style.display = "none";
  $("room-screen").style.display = "flex";
  $("room-badge").style.display = "block";
  $("room-badge").textContent = roomCode;
  $("side-code").textContent = roomCode;
  $("side-code").addEventListener("click", () => { navigator.clipboard?.writeText(roomCode); toast("room code copied"); });
  peerCard("self", myName, myMeta, true);
  updateCluster();
  log("swarm", `room ${roomCode} — share this code with your other devices`);
  $("ai-panel").style.display = "flex";
  aiStatus("");
  $("ai-empty").textContent = "pick a model and press start, from any device";
  const selfCard = document.querySelector(".peer-card.self");
  if (selfCard && myMeta.webgpu) {
    const row = document.createElement("div");
    row.className = "pledge";
    row.innerHTML = `give <input type="number" min="1" max="64" step="1" value="${myMeta.contribGB}"> GB of GPU`;
    selfCard.appendChild(row);
    row.querySelector("input").addEventListener("change", (e) => {
      const v = parseFloat(e.target.value);
      if (v >= (myMeta.phone ? 0.5 : 1)) { myMeta.contribGB = v; selfCard.querySelector(".buf").textContent = "gives " + v + " GB"; updateCluster(); broadcastAll({ t: "pledge", gb: v }); }
    });
  }
}

// --- connection wiring ---
function wire(conn, name, meta, initiator = false) {
  const entry = { conn, name: name || conn.peer, meta: meta || {}, rtt: null, card: null, link: makeLink(), stripes: [] };
  conns.set(conn.peer, entry);
  if (WIRE_STRIPES > 0) {
    attachWire(entry.link, conn, (m) => onData(conn.peer, m));
    // extra associations for striping: the side that dialed opens them, the other side accepts
    // them in peer.on("connection") by label and attaches its end of the wire channel
    if (initiator) for (let i = 1; i < WIRE_STRIPES; i++) {
      const sc = peer.connect(conn.peer, { reliable: true, label: "stripe" });
      sc.on("open", () => { attachWire(entry.link, sc, (m) => onData(conn.peer, m)); });
      sc.on("error", () => {});
      entry.stripes.push(sc);
    }
  }

  conn.on("data", (d) => onData(conn.peer, d));
  conn.on("close", () => {
    const e = conns.get(conn.peer);
    if (isHost) peerGone(conn.peer, e?.name);   // before the roster goes out: the chain may need re-dealing
    conns.delete(conn.peer);
    gone.delete(conn.peer);
    if (isHost) {   // on the host a closed link means the device left; workers wait for the roster
      dropCard(conn.peer); members.delete(conn.peer); roster.delete(conn.peer); broadcastRoster();
      log("swarm", `${e?.name || conn.peer} left`);
    } else if (conn.peer === ai.hostId || entry.name === "host") { log("swarm", "lost the link to the host"); hostGone(); }
    updateCluster();
  });
  conn.on("error", () => {});
  // a dead tab shows up as ICE failure long before PeerJS closes the connection; a
  // "disconnected" that does not recover in 1.5 s counts too (host side only: the host
  // re-plans and a false alarm is undone when the link comes back)
  const pc = conn.peerConnection;
  if (pc && isHost) pc.addEventListener("iceconnectionstatechange", () => {
    const st = pc.iceConnectionState;
    if (st === "failed") peerGone(conn.peer, entry.name);
    else if (st === "disconnected") setTimeout(() => { if (conns.get(conn.peer) === entry && !/^(connected|completed)$/.test(pc.iceConnectionState)) peerGone(conn.peer, entry.name); }, 1500);
    else if ((st === "connected" || st === "completed") && gone.has(conn.peer)) { gone.delete(conn.peer); if (ai.role === "host" && ai.state !== "idle") schedulePlan(`${entry.name} is back`); }
  });
  return entry;
}

function ensureCard(id, name, meta) {
  let card = cards.get(id);
  if (!card) {
    card = peerCard(id, name || id, meta || {}, false);
    cards.set(id, card);
    updateCluster();
    log("swarm", `${name || id} joined`);
    mascot(`${name || id} joined! ${members.size + 1} devices in the room.`);
  }
  const e = conns.get(id);
  if (e) e.card = card;
  return card;
}
function dropCard(id) { const c = cards.get(id); if (c) { c.remove(); cards.delete(id); } }
// open a data link to a chain neighbour if we do not have one yet; resolves when it is up
function ensureLink(id, timeoutMs = 60000) {
  if (!id || id === "host" || conns.has(id)) return Promise.resolve(true);
  if (!ensureLink.pending.has(id)) { ensureLink.pending.add(id); meshConnect(id); }
  return new Promise((res) => {
    const t0 = performance.now();
    const t = setInterval(() => {
      if (conns.has(id)) { clearInterval(t); ensureLink.pending.delete(id); res(true); }
      else if (performance.now() - t0 > timeoutMs) { clearInterval(t); ensureLink.pending.delete(id); res(false); }
    }, 100);
  });
}
ensureLink.pending = new Set();

function sendTo(id, obj) { const e = conns.get(id); if (e?.conn?.open) e.conn.send(obj); }   // a closing connection is skipped, not thrown at
// debug: per-peer wire state (channels open, frames sent/received) — `swarmDebug()` in the console
window.swarmDebug = () => [...conns].map(([id, e]) => ({ id, name: e.name, chans: e.link?.chans.filter((c) => c.readyState === "open").length ?? 0, sent: e.link?.sent ?? 0, recv: e.link?.recv ?? 0 }));
// debug + emulator: the current layer plan and room state — `swarmPlan()` in the console
window.swarmPlan = () => ({ v: ai.planV, state: ai.state, chain: ai.plan?.chain, layersByName: ai.layersByName, range: ai.range, engineLayers: ai.engine ? [ai.engine.lo, ai.engine.hi] : null, next: ai.next, role: ai.role });
// activations go over the sliced wire channel when it is up, else as a normal message
function sendHidden(id, msg) {
  const e = conns.get(id);
  if (e?.link && wireReady(e.link) && sendFrame(e.link, msg)) return;
  sendTo(id, msg);
}
function broadcastAll(obj) { for (const [id] of conns) sendTo(id, obj); }

// bandwidth test state
const bwRecv = new Map(); // fromId -> {bytes, t0}

function onData(from, d) {
  // binary chunk = bandwidth test payload
  if (d instanceof ArrayBuffer || ArrayBuffer.isView(d)) {
    const st = bwRecv.get(from);
    if (st) st.bytes += d.byteLength || d.length;
    return;
  }
  const e = conns.get(from);
  if (e) e.pongAt = Date.now();   // any message is proof of life for the watchdogs
  if (d.t && d.t.startsWith("ai-")) { aiOnData(from, d); return; }
  switch (d.t) {
    case "hello":
      e.name = d.name; e.meta = d.meta; e.tab = d.tab;
      members.set(from, { name: d.name, meta: d.meta });
      ensureCard(from, d.name, d.meta);
      if (isHost) {
        // a reloaded tab comes back with a new peer id but the same tab id: leave + join. An older
        // peer without a tab id is matched by name, and only when the old link looks dead already
        // (a second live device with the same name is not a reload)
        for (const [oid, oe] of conns) {
          if (oid === from) continue;
          const reload = d.tab ? oe.tab === d.tab : (oe.name === d.name && (!oe.conn?.open || Date.now() - (oe.pongAt || 0) > 5000));
          if (reload) peerGone(oid, d.name);
        }
        roster.set(from, { name: d.name, meta: d.meta }); broadcastRoster();
        ai.exclude.delete(from);
        if (d.died) log("swarm", `${d.name} came back \u2014 its tab died ${d.died.ago}s ago during "${d.died.during}"`);
        if (ai.state !== "idle") {
          sendTo(from, { t: "ai-layers", v: ai.planV, model: ai.modelKey, by: ai.layersByName, state: ai.state, note: "" });
          if (d.meta?.webgpu !== false) schedulePlan(`${d.name} joined`);
        }
      }
      break;
    case "roster": {
      // the host's view of the room: draw a card per device, no mesh connections
      const seen = new Set();
      for (const m of d.members) {
        if (m.id === peer.id) continue;
        seen.add(m.id);
        members.set(m.id, { name: m.name, meta: m.meta });
        const c = ensureCard(m.id, m.name, m.meta);
        if (m.meta?.contribGB) c.querySelector(".buf").textContent = "gives " + m.meta.contribGB + " GB";
        const ce = conns.get(m.id); if (ce) ce.meta = m.meta;
      }
      for (const id of [...members.keys()]) if (!seen.has(id)) { members.delete(id); dropCard(id); }
      updateCluster();
      break;
    }
    case "ping": sendTo(from, { t: "pong", ts: d.ts }); break;
    case "pong": {
      e.rtt = Math.round(performance.now() - d.ts);
      // a pong from a peer the watchdog gave up on: it was a stall, not a leave
      if (isHost && gone.has(from)) { gone.delete(from); if (ai.role === "host" && ai.state !== "idle") schedulePlan(`${e.name} is back`); }
      if (e.card) e.card.querySelector(".rtt").textContent = e.rtt + " ms";
      break;
    }
    case "pledge":
      if (e) { e.meta = { ...e.meta, contribGB: d.gb }; if (e.card) e.card.querySelector(".buf").textContent = "gives " + d.gb + " GB"; }
      if (members.has(from)) members.get(from).meta = { ...members.get(from).meta, contribGB: d.gb };
      if (isHost && roster.has(from)) { roster.get(from).meta = { ...roster.get(from).meta, contribGB: d.gb }; broadcastRoster(); }
      updateCluster();
      break;
    case "bw-start": bwRecv.set(from, { bytes: 0, t0: performance.now() }); break;
    case "bw-end": {
      const st = bwRecv.get(from);
      if (st) {
        const secs = (performance.now() - st.t0) / 1000;
        const mbps = (st.bytes * 8 / 1e6 / secs).toFixed(0);
        sendTo(from, { t: "bw-result", mbps });
        bwRecv.delete(from);
      }
      break;
    }
    case "bw-result":
      if (e.card) e.card.querySelector(".bw").textContent = d.mbps + " Mbps";
      log("swarm", `bandwidth to ${e.name}: ${d.mbps} Mbps`);
      break;
  }
}

function broadcastRoster() {
  const members = [{ id: peer.id, name: myName, meta: myMeta },
    ...[...roster.entries()].map(([id, m]) => ({ id, ...m }))];
  broadcastAll({ t: "roster", members });
}

function meshConnect(targetId) {
  const conn = peer.connect(targetId, { reliable: true });
  conn.on("open", () => {
    wire(conn, undefined, undefined, true);
    conn.send({ t: "hello", name: myName, meta: myMeta, tab: TAB });
  });
}

async function bwTest(id) {
  const e = conns.get(id);
  if (!e) return;
  log("swarm", `testing bandwidth to ${e.name}…`);
  sendTo(id, { t: "bw-start" });
  const chunk = new Uint8Array(64 * 1024);
  const total = 4 * 1024 * 1024;
  for (let sent = 0; sent < total; sent += chunk.length) {
    e.conn.send(chunk);
    // yield so the datachannel buffer can drain
    if (e.conn.dataChannel && e.conn.dataChannel.bufferedAmount > 1 << 20)
      await new Promise(r => setTimeout(r, 20));
  }
  sendTo(id, { t: "bw-end" });
}

// --- ping loop; on the host also the liveness watchdog for chain members (3 missed pongs) ---
// Both watchdogs skip a tick that comes late (the host's own main thread stalled: GC, a long
// GPU submit, a throttled tab): pongs queued during the stall have not been dispatched yet, so
// silence measured across it says nothing about the peers. The stalled tick refreshes pongAt.
let idleTick = Date.now();
setInterval(() => {
  broadcastAll({ t: "ping", ts: performance.now() });
  const now = Date.now(), stalled = now - idleTick > 5000; idleTick = now;
  if (ai.role !== "host") return;
  for (const id of ai.chain) { const e = conns.get(id); if (!e) continue; if (stalled) e.pongAt = now; else if (now - (e.pongAt || now) > 7500) peerGone(id, e.name); }
}, 2500);
// while an answer is in flight the host pings the chain it is served by every 500 ms and gives
// up on a member 2 s after its last message (any message counts: pongs, hidden states); a tab
// that closed without a clean PeerJS close then stops the answer in ~2 s instead of 7.5 s
let genTick = Date.now();
setInterval(() => {
  const now = Date.now(), stalled = now - genTick > 1500; genTick = now;
  if (ai.role !== "host" || !ai.gen) return;
  for (const id of ai.gen.chain) {
    const e = conns.get(id); if (!e) continue;
    sendTo(id, { t: "ping", ts: performance.now() });
    if (stalled) e.pongAt = now; else if (now - (e.pongAt || now) > 2000) peerGone(id, e.name);
  }
}, 500);

const stepGB = (d) => { const i = $("join-gb"); const lo = parseFloat(i.min) || 1; const st = parseFloat(i.step) || 1; i.value = Math.min(64, Math.max(lo, (parseFloat(i.value) || lo) + d * st)); };
$("gb-minus").addEventListener("click", () => stepGB(-1));
$("gb-plus").addEventListener("click", () => stepGB(1));
// --- join / create ---
async function start(create) {
  myName = $("name-input").value.trim() || (create ? "host" : "peer") + "-" + rand(2);
  const code = create ? rand(4) : $("code-input").value.trim().toUpperCase();
  if (!code) { $("join-status").textContent = "enter a room code"; return; }
  $("create-btn").disabled = $("join-btn").disabled = true;
  $("join-status").textContent = "connecting to signaling…";
  myMeta = await metaPromise;
  const gbIn = parseFloat($("join-gb").value);
  myMeta.contribGB = Math.max(myMeta.phone ? 0.5 : 1, gbIn > 0 ? gbIn : (myMeta.contribGB || 1));

  // STUN for hole-punching; TURN as fallback for symmetric NAT / CGNAT peers.
  // ICE prefers direct candidates, so TURN only carries traffic when a direct
  // path is impossible.
  const ICE = {
    iceServers: [
      { urls: ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"] },
      // TURN fallback for symmetric-NAT peers goes here (needs credentials —
      // see TURN_CREDS below); without it, strict-NAT peers can't join.
      ...(window.TURN_SERVERS || []),
    ],
  };
  // host claims the well-known id for the code; joiners get random ids
  peer = new Peer(create ? PREFIX + code : undefined, { debug: 1, config: ICE, ...SIGNAL_OPTS });

  peer.on("open", () => {
    isHost = create;
    roomCode = code;
    if (create) { enterRoom(); return; }
    // joiner: connect to host
    $("join-status").textContent = "joining room " + code + "…";
    const conn = peer.connect(PREFIX + code, { reliable: true });
    const timeout = setTimeout(() => {
      const ice = conn.peerConnection?.iceConnectionState;
      $("join-status").textContent =
        ice === "checking" || ice === "failed" || ice === "disconnected"
          ? "found the room, but the direct connection failed (strict NAT/firewall on one side) — trying relay, give it ~20s or try another network"
          : "no room with that code (is the host page open?)";
      $("create-btn").disabled = $("join-btn").disabled = false;
    }, 15000);
    conn.on("open", () => {
      clearTimeout(timeout);
      wire(conn, "host", undefined, true);
      let died = null;
      try { const c = JSON.parse(localStorage.getItem("swarm-crumb") || "null"); if (c && Date.now() - c.t < 10 * 60 * 1000) died = { during: c.s, ago: Math.round((Date.now() - c.t) / 1000) }; } catch {}
      conn.send({ t: "hello", name: myName, meta: myMeta, tab: TAB, died });
      enterRoom();
    });
  });

  peer.on("connection", (conn) => {
    conn.on("open", () => {
      if (conn.label === "stripe") {   // extra association for the hidden-state wire, not a new peer
        const e = conns.get(conn.peer);
        if (e) { attachWire(e.link, conn, (m) => onData(conn.peer, m)); e.stripes.push(conn); }
        return;
      }
      wire(conn);
      conn.send({ t: "hello", name: myName, meta: myMeta, tab: TAB });
    });
  });

  peer.on("error", (err) => {
    if (err.type === "unavailable-id")
      $("join-status").textContent = "that code is already hosting — pick Join instead";
    else if (err.type === "peer-unavailable")
      $("join-status").textContent = "no room with that code";
    else
      $("join-status").textContent = "error: " + err.type;
    $("create-btn").disabled = $("join-btn").disabled = false;
  });
}

let wakeLock = null, awakeVideo = null;
function awakeStatus(s) { const el = $("awake"); if (el && myMeta?.phone) el.textContent = s; }
async function keepAwake() {
  // 1. the real API (iOS 16.4+, must be called from a tap)
  try {
    if (!wakeLock && navigator.wakeLock) {
      wakeLock = await navigator.wakeLock.request("screen");
      wakeLock.addEventListener("release", () => { wakeLock = null; awakeStatus("screen lock: released"); });
      awakeStatus("screen stays awake \u2713");
    }
  } catch (e) { awakeStatus("wake lock failed: " + (e?.message || e)); }
  // 2. belt and braces: a silent looping video keeps iOS from locking the screen
  try {
    if (!awakeVideo) {
      awakeVideo = document.createElement("video");
      awakeVideo.setAttribute("playsinline", ""); awakeVideo.muted = true; awakeVideo.loop = true;
      awakeVideo.style.cssText = "position:fixed;width:1px;height:1px;opacity:0.01;pointer-events:none;bottom:0;left:0";
      awakeVideo.src = "data:video/mp4;base64,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";
      document.body.appendChild(awakeVideo);
    }
    await awakeVideo.play();
    if (!wakeLock) awakeStatus("screen stays awake (video) \u2713");
  } catch (e) { if (!wakeLock) awakeStatus("\u26a0 can\u2019t keep the screen awake: set Auto-Lock to Never"); }
}
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") keepAwake(); });
document.addEventListener("touchstart", keepAwake, { passive: true });
$("create-btn").addEventListener("click", () => { keepAwake(); start(true); });
// (auto-rejoin removed: the user prefers to see what happened)
$("join-btn").addEventListener("click", () => { keepAwake(); start(false); });
$("code-input").addEventListener("keydown", (e) => { if (e.key === "Enter") start(false); });
$("room-badge").addEventListener("click", () => {
  navigator.clipboard?.writeText(roomCode);
  log("swarm", "room code copied");
});

// ================= distributed inference =================

// ---- on-disk cache of weight ranges (Cache API): a second start skips the download ----
let weightCache = null, cacheHits = 0;
async function getWeightCache() {
  if (weightCache !== null) return weightCache;
  try { weightCache = await caches.open("swarmllm-weights-v1"); } catch { weightCache = false; }
  return weightCache;
}
function cacheKey(url, lo, hi) { return "https://weights.swarmllm.ai/" + encodeURIComponent(url) + "/" + lo + "-" + hi; }
async function rangeFetch(url, lo, hi, noCache = false) {
  const c = await getWeightCache();
  const key = cacheKey(url, lo, hi);
  if (c && !noCache) {
    try {
      const hit = await c.match(key);
      if (hit) {
        // only trust a complete entry: a tab that died mid-write leaves a short one behind
        if (hit.headers.get("x-swarm-len") === String(hi - lo + 1)) { cacheHits += hi - lo + 1; return hit; }
        c.delete(key).catch(() => {});
      }
    } catch {}
  }
  const r = await fetch(url, { headers: { Range: `bytes=${lo}-${hi}` } });
  if (r.status !== 206) throw new Error("model host refused range requests");
  if (c && !myMeta?.phone) {   // phones skip the store (no spare RAM for the copy); Cache API refuses 206s, so store as a plain 200
    try {
      // buffer the copy fully first, so a complete body is the only thing that ever gets stored
      r.clone().arrayBuffer().then((buf) => {
        if (buf.byteLength !== hi - lo + 1) return;
        return c.put(key, new Response(buf, { status: 200, headers: { "content-type": "application/octet-stream", "x-swarm-len": String(buf.byteLength) } }));
      }).then(() => { ai.cachedBytes = (ai.cachedBytes || 0) + (hi - lo + 1); }, () => {});
    } catch {}
  }
  return r;
}
async function fetchGGUFHeader(url, needTokenizer = true) {
  let size = 12 * 2 ** 20;
  for (;;) {
    const r = await rangeFetch(url, 0, size - 1);   // 206 from the network, 200 from the cache
    const buf = await r.arrayBuffer();
    try { return parseGGUFHeader(buf, { skipTokenizer: !needTokenizer }); }
    catch (e) { if (size > 256 * 2 ** 20) throw e; size *= 2; }
  }
}
let pacerHook = null;
const streamWithRetry = (url, streamOpts) => async (info) => {
  try { return await streamEntryToGPU(ai.device, info, openRangeOf(url), streamOpts); }
  catch (e) {
    if (!/short tensor/.test(String(e))) throw e;
    const c = await getWeightCache();
    if (c) c.delete(cacheKey(url, info.byteOffset, info.byteOffset + info.byteLength - 1)).catch(() => {});
    return streamEntryToGPU(ai.device, info, (i) => rangeFetch(url, i.byteOffset, i.byteOffset + i.byteLength - 1, true), streamOpts);
  }
};
const openRangeOf = (url) => async (info) => {
  if (pacerHook) await pacerHook();
  crumb("streaming " + info.name + " (" + (info.byteLength / 2 ** 20).toFixed(0) + " MB)");
  return rangeFetch(url, info.byteOffset, info.byteOffset + info.byteLength - 1);
};
const rangeBytesOf = (url) => async (info) => {
  if (pacerHook) await pacerHook();
  crumb("fetching " + info.name + " (" + (info.byteLength / 2 ** 20).toFixed(0) + " MB)");
  let r = await rangeFetch(url, info.byteOffset, info.byteOffset + info.byteLength - 1);
  let bytes = new Uint8Array(await r.arrayBuffer());
  if (bytes.length !== info.byteLength) {
    r = await rangeFetch(url, info.byteOffset, info.byteOffset + info.byteLength - 1, true);
    bytes = new Uint8Array(await r.arrayBuffer());
    if (bytes.length !== info.byteLength) throw new Error(`short download for ${info.name}: ${bytes.length}/${info.byteLength} bytes`);
  }
  return bytes;
};

let ai = {
  engine: null, tok: null, cfg: null, device: null,
  role: null,            // "host" | "worker" | "guest"
  chain: [],             // host: worker peer ids in pipeline order (current plan)
  next: null,            // worker: peer id to forward hidden to, or "host"
  readyPeers: new Set(), // host: ids that sent ai-ready for the current plan
  pos: 0,
  waiters: new Map(),    // pos | "b"+pos -> {res, rej, timer}: host laps awaiting their return (waitLap)
  busy: false,           // false | true (first load) | "plan" (re-deal loading) | "gen" (answer) | "wait" (room too small)
  stopReason: null,      // host: set when a chain member left mid-answer; waitLap rejects with it
  // room state machine (docs/protocol.md § Room states)
  state: "idle",         // idle | loading | online | generating | redealing | waiting
  planV: 0,              // plan version, monotonic per room; workers: the version they hold
  plan: null,            // host: current planSplit result (+ v)
  gen: null,             // host: {v, chain} frozen for the answer in flight
  planWanted: null,      // host: reason for a pending re-plan
  planTimer: null,
  speed: new Map(),      // host: peer id -> autotune ms (feeds the speed weight)
  exclude: new Set(),    // host: ids that failed to load the current plan
  loadQ: Promise.resolve(),   // every device: shard loads/frees run one after another
  inflight: 0,           // worker: laps being computed; aiQuiesce waits for 0 before a device is destroyed
  quiesce: [],
};
const gone = new Set();  // host: peers declared gone (ICE/pong) whose connection has not closed yet

function aiStatus(s) { $("ai-status").textContent = s; crumb(s); }
// room state: drives the Send box, the re-deal button and body[data-state]/[data-plan] (the
// emulator reads those); the host mirrors it to every screen with ai-layers
function setState(state, note) {
  ai.state = state;
  document.body.dataset.state = state; document.body.dataset.plan = String(ai.planV);
  if (note) aiStatus(note);
  if (ai.role !== "host") return;
  if (state === "online") $("ai-send").disabled = false; else if (state !== "generating") $("ai-send").disabled = true;
  const rd = $("ai-redeal");
  rd.hidden = !(state === "online" || state === "waiting" || state === "redealing");
  if (state === "online") rd.textContent = "re-deal layers";
  broadcastAll({ t: "ai-layers", v: ai.planV, model: ai.modelKey, by: ai.layersByName || {}, state, note: note || "" });
}
// breadcrumb: if iOS kills the tab, the reloaded page can say where it died
function crumb(s) { try { localStorage.setItem("swarm-crumb", JSON.stringify({ s, t: Date.now(), mem: performance.memory?.usedJSHeapSize })); } catch {} }
// (crumb is kept in localStorage for debugging, not shown on the join screen)
function aiLoading(show, title) {
  $("ai-loading").style.display = show ? "block" : "none";
  if (title) $("ldg-title").textContent = title;
  $("ai-panel").classList.toggle("loading", !!show);
  $("load-card").classList.toggle("on", !!show);
  $("ai-empty").style.display = show ? "none" : "";
  if (show) { $("lc-model").textContent = MODELS[$("ai-model").value]?.label.split("\u00b7")[0].trim() || ""; loadCardRender(); }
}
function loadCardRender() {
  const rows = $("lc-rows"); if (!rows) return;
  const names = [myName, ...[...conns.values()].map((c) => c.name)];
  const layersOf = (nm) => (ai.layersByName || {})[nm];
  rows.innerHTML = names.map((nm) => {
    const pct = Math.max(0, Math.min(100, (ai.prog || {})[nm] ?? 0));
    const l = layersOf(nm);
    return `<div class="lc-row${pct >= 100 ? " done" : ""}"><div class="n">${nm}${l ? `<small>layers ${l}</small>` : ""}</div><div class="bar"><div class="fill" style="width:${pct}%"></div></div><div class="pct">${pct >= 100 ? "ready" : pct + "%"}</div></div>`;
  }).join("");
}
function aiProgress(done, total, note) {
  const pct = total ? Math.min(100, Math.round(done / total * 100)) : 0;
  $("ldg-fill").style.width = pct + "%";
  $("ldg-sub").textContent = `${(done / 2 ** 20).toFixed(0)} MB of ${(total / 2 ** 20).toFixed(0)} MB · ${pct}%` + (note ? " · " + note : "");
}
function aiOut() { const o = $("ai-output"); o.style.display = "block"; $("ai-empty").style.display = "none"; return o; }
let botEl = null;
function chatUser(name, text) {
  const o = aiOut();
  const m = document.createElement("div");
  m.className = "m user";
  m.innerHTML = `<div class="who">${esc(name)}</div><div class="bubble">${esc(text)}</div>`;
  o.appendChild(m); o.scrollTop = o.scrollHeight;
}
function chatBotStart() {
  const o = aiOut();
  const m = document.createElement("div");
  m.className = "m bot";
  m.innerHTML = `<div class="who">swarm</div><div class="bubble"><span class="cursor"></span></div>`;
  o.appendChild(m); o.scrollTop = o.scrollHeight;
  botEl = m;
}
function chatBotUpdate(raw) {
  if (!botEl) chatBotStart();
  botEl.querySelector(".bubble").innerHTML = md(raw) + '<span class="cursor"></span>';
  $("ai-output").scrollTop = $("ai-output").scrollHeight;
}
function chatBotEnd(raw, stats) {
  if (!botEl) chatBotStart();
  botEl.querySelector(".bubble").innerHTML = md(raw);
  if (stats) { const s = document.createElement("div"); s.className = "stats"; s.textContent = stats; botEl.appendChild(s); }
  botEl = null;
}


// quiet: a background re-deal load — no loading card, no pacer (that is for the first
// download, when the whole room downloads together); progress goes to the status line only
async function aiLoadShard(modelKey, range, hasEmbed, hasHead, { quiet = false } = {}) {
  const M = MODELS[modelKey];
  if (!quiet) aiLoading(true, `downloading layers ${range[0]}\u2013${range[1] - 1} of ${M.label.split("\u00b7")[0].trim()}`);
  aiStatus("requesting GPU\u2026");
  if (!quiet) mascot("Grabbing my slice of the model… hang tight.");
  // a previous attempt in this tab still owns its weights: release them first, or the
  // second load doubles GPU memory and every buffer after the limit comes back invalid
  await aiQuiesce();
  if (ai.device) { try { ai.device.destroy(); } catch {} ai.device = null; ai.engine = null; }
  ai.firstGpuError = null;
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("no WebGPU on this device");
  ai.device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: myMeta?.phone ? Math.min(adapter.limits.maxBufferSize, 256 * 2 ** 20) : adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: myMeta?.phone ? Math.min(adapter.limits.maxStorageBufferBindingSize, 256 * 2 ** 20) : adapter.limits.maxStorageBufferBindingSize,
    },
  });
  ai.device.addEventListener?.("uncapturederror", (ev) => {
    const gmsg = ev.error?.message || "";
    if (!ai.firstGpuError) { ai.firstGpuError = gmsg; aiStatus("GPU error: " + gmsg.slice(0, 300)); log("swarm", "\u26a0 FIRST GPU error on " + myName + ": " + gmsg.slice(0, 600)); }
    crumb("GPU validation error: " + gmsg.slice(0, 400));
    if (ai.hostId && ai.role !== "host") sendTo(ai.hostId, { t: "ai-error", message: "GPU error: " + (ev.error?.message || "").slice(0, 300) });
    log("swarm", "\u26a0 GPU error on " + myName + ": " + (ev.error?.message || "").slice(0, 140));
  });
  if (location.hash === "#debug") log("swarm", `${myName}: maxBuf ${(adapter.limits.maxBufferSize / 2 ** 30).toFixed(1)} GB \u00b7 maxBind ${(adapter.limits.maxStorageBufferBindingSize / 2 ** 20).toFixed(0)} MB`);
  aiStatus("testing GPU kernels on this device\u2026");
  const tAdapter = await navigator.gpu.requestAdapter();   // an adapter gives out one device only
  const tdev = await tAdapter.requestDevice();               // throwaway: its test buffers die with it
  const st = await gpuSelfTest(tdev);
  if (!st.ok) log("swarm", `${myName} GPU self-test: ${st.detail}`);
  if (!st.ok) throw new Error("GPU self-test FAILED on this device: " + st.detail + " \u2014 please screenshot this");
  const mt = await kernelMicroTests(tdev);
  if (!mt.ok) log("swarm", `${myName} kernels: ${mt.detail}`);
  if (!mt.ok) throw new Error("GPU kernel FAILED on this device \u2192 " + mt.firstFail + " \u2014 please send me this line");
  try { tdev.destroy(); } catch {}
  ai.device.lost.then((l) => crumb("GPU device lost: " + l.reason + " " + l.message));
  aiStatus("tuning kernels for this GPU\u2026");
  ai.tune = await autotuneCoop(ai.device).catch(() => ({ wg: 256, rows: 4 }));
  crumb(`autotune: WG=${ai.tune.wg} ROWS=${ai.tune.rows}`);
  if (ai.role === "host" && ai.tune?.results?.[0]?.ms) ai.speed.set(peer.id, ai.tune.results[0].ms);
  const isPhone = myMeta?.phone;
  ai.myPct = 0;
  ai.prog = { [myName]: 0 }; ai.progAt = { [myName]: Date.now() };
  const streamOpts = { pace: isPhone ? 300 : 0, staging: isPhone ? 2 * 2 ** 20 : 8 * 2 ** 20 };
  if (M.cfg) {
    ai.cfg = await (await fetch(M.cfg)).json();
    if (hasEmbed || hasHead) ai.tok = makeTokenizer(await (await fetch(M.tok)).json());
  }

  const onProg = (done, total) => {
    aiProgress(done, total);
    aiStatus(cacheHits > done * 0.5 ? `loading weights from this device's cache\u2026` : `downloading weights\u2026`);
    ai.myPct = total ? done / total * 100 : 0;
    ai.prog = ai.prog || {}; ai.progAt = ai.progAt || {};
    ai.prog[myName] = Math.round(ai.myPct); ai.progAt[myName] = Date.now();
    if (ai.role === "worker") sendTo(ai.hostId, { t: "ai-progress", pct: Math.round(ai.myPct) });
    loadCardRender();
  };
  if (ai.role === "host" && !quiet) {
    clearInterval(ai.progTimer);
    ai.progTimer = setInterval(() => { if (ai.role === "host") broadcastAll({ t: "ai-hostprog", all: ai.prog || {}, at: Date.now() }); }, 600);
  }
  // every device (host included, even when its weights come from cache) keeps within a few
  // percent of the slowest device, so the bars climb together and the room finishes as one
  const slowest = () => {
    const now = Date.now();
    let m = Infinity;
    for (const [nm, pct] of Object.entries(ai.prog || {})) {
      if (nm === myName || pct >= 100) continue;
      if (now - ((ai.progAt || {})[nm] || 0) > 30000) continue;     // silent for 30 s: don't wait on it
      m = Math.min(m, pct);
    }
    return m;
  };
  const pacer = async () => {
    while (ai.myPct < 100 && ai.myPct > slowest() + 4) {
      aiStatus(`downloading weights\u2026 in step with the room (${Math.round(ai.myPct)}%)`);
      await new Promise((r) => setTimeout(r, 250));
    }
  };
  pacerHook = quiet ? null : pacer;

  if (M.kind === "qwen35") {
    aiStatus("reading model index\u2026");
    const needTok = hasEmbed || hasHead;
    const cachedOk = ai.G && ai.GModel === modelKey && (!needTok || ai.G.meta["tokenizer.ggml.tokens"]);
    const G = cachedOk ? ai.G : await fetchGGUFHeader(M.gguf, needTok);
    ai.G = G; ai.GModel = modelKey;
    ai.cfg = { num_hidden_layers: G.meta["qwen35.block_count"] - (G.meta["qwen35.nextn_predict_layers"] || 0) };
    if (hasEmbed || hasHead) ai.tok = makeTokenizer(tokenizerFromGGUF(G.meta));
    // the host also loads the model's multi-token-prediction block: it drafts
    // tokens that the trunk then verifies in one batched pass (same output, faster)
    const opts = { lo: range[0], hi: range[1], hasEmbed, hasHead, mtp: hasHead };
    const total = qwen35ShardBytes(G, opts);
    G.streamEntry = streamWithRetry(M.gguf, streamOpts);
    const weights = await qwen35Weights(G, rangeBytesOf(M.gguf), opts, (done) => onProg(done, total),
      (e, name) => gpuUploadEntry(ai.device, e, name === GGML_EMBED));   // straight to the GPU, RAM stays flat
    aiStatus("building GPU pipelines (compiling shaders)\u2026");
    ai.engine = await Qwen35Engine.create({
      device: ai.device, meta: G.meta, weights, vocab: G.tensors[GGML_EMBED]?.shape?.[0],
      layerRange: range, hasEmbed, hasHead, maxSeq: MAX_SEQ,
      coopWG: ai.tune?.wg, coopRows: ai.tune?.rows,
      // 16 batch columns: prefill passes go through the row-stationary GEMM
      // (docs/research/prefill-gemm-v2.md). Speculative verifies are <= 8
      // columns and drop to the 8- or 4-column GEMV twins automatically, so
      // the generated stream is unchanged.
      batchCols: 16, coopRowsB: 1,
    });
  } else if (M.kind === "gguf") {
    aiStatus("reading model index\u2026");
    const G = ai.G && ai.GModel === modelKey ? ai.G : await fetchGGUFHeader(M.gguf, false);   // vocab comes from tokenizer.json
    ai.G = G; ai.GModel = modelKey;
    const opts = { lo: range[0], hi: range[1], hasEmbed, hasHead };
    const total = ggufShardBytes(G, opts);
    G.streamEntry = streamWithRetry(M.gguf, streamOpts);
    const weights = await ggufWeights(G, rangeBytesOf(M.gguf), opts, (done) => onProg(done, total),
      (e, name) => gpuUploadEntry(ai.device, e, name === GGML_EMBED));
    aiStatus("building GPU pipelines\u2026");
    ai.engine = await DenseEngine.create({
      coopWG: ai.tune?.wg, coopRows: ai.tune?.rows,
      device: ai.device, cfg: ai.cfg, weights,
      layerRange: range, hasEmbed, hasHead, maxSeq: MAX_SEQ,
    });
  } else {
    const names = shardTensorNames(ai.cfg, range, hasEmbed, hasHead);
    const tensors = await fetchModelShard(M.st, names, (p, done, total) => onProg(done, total));
    aiStatus("building GPU pipelines\u2026");
    ai.engine = await DenseEngine.create({
      coopWG: ai.tune?.wg, coopRows: ai.tune?.rows,
      device: ai.device, cfg: ai.cfg, tensors,
      layerRange: range, hasEmbed, hasHead, maxSeq: MAX_SEQ,
    });
  }
  ai.range = range;
  ai.model = modelKey;
  if (!quiet) aiLoading(false);
}
// a device that lost all its layers in a re-deal frees the GPU; the bytes stay in the Cache API
async function aiFree() {
  await aiQuiesce();
  try { ai.device?.destroy(); } catch {}
  ai.device = ai.engine = null; ai.range = null;
  aiLoading(false);
}
// resolves once no lap is being computed on this device (frames in flight finish first)
function aiQuiesce() { return ai.inflight ? new Promise((r) => ai.quiesce.push(r)) : Promise.resolve(); }

// ---- host ----
// planner inputs from the room: pledges in bytes, WebGPU flag, and the autotune ms per device
// (ai.speed, filled from ai-ready) — see room/plan.js
function aiMembersForPlan() {
  const pledgeOf = (m) => ((m?.contribGB ?? (m?.maxBufGB ? m.maxBufGB * 0.5 : 0.5))) * 2 ** 30;
  const speed = ai.speed || new Map();
  return {
    host: { id: peer.id, name: myName, pledgeBytes: pledgeOf(myMeta), ms: speed.get(peer.id) },
    workers: [...conns].filter(([id]) => !gone.has(id)).map(([id, e]) => ({ id, name: e.name || id, pledgeBytes: pledgeOf(e.meta), webgpu: e.meta?.webgpu !== false, ms: speed.get(id) })),
  };
}
function biggestPeerId() {
  const gb = (m) => m?.contribGB ?? 0;
  let best = peer.id, bestGB = gb(myMeta);
  for (const [id, e] of conns) if (gb(e.meta) > bestGB || (gb(e.meta) === bestGB && id < best)) { best = id; bestGB = gb(e.meta); }
  return best;
}
function aiStartAnywhere() {
  const model = $("ai-model").value;
  const boss = biggestPeerId();
  if (boss === peer.id) { aiStart(model); return; }
  $("ai-start").disabled = true; $("ai-model").disabled = true;
  aiLoading(true, `starting ${MODELS[model].label.split("\u00b7")[0].trim()}`);
  $("ldg-sub").textContent = `${conns.get(boss)?.name || "the biggest device"} is dealing the layers`;
  $("ldg-fill").style.width = "0%";
  aiStatus(`asked ${conns.get(boss)?.name || "the biggest device"} to start ${MODELS[model].label.split("\u00b7")[0].trim()}\u2026`);
  broadcastAll({ t: "ai-start-req", model, boss, by: myName });
}
async function aiStart(modelArg) {
  if (ai.engine || ai.busy || ai.state !== "idle") return;
  ai.busy = true;
  if (typeof modelArg === "string") $("ai-model").value = modelArg;
  $("ai-start").disabled = true;
  $("ai-model").disabled = true;
  try {
    ai.role = "host";
    const modelKey = $("ai-model").value;
    const M = MODELS[modelKey];
    let L, layerBytes, embedBytes, cfg = null;
    if (M.kind === "qwen35") {
      aiStatus("reading model index\u2026 (11 MB)");
      ai.G = await fetchGGUFHeader(M.gguf);
      ai.GModel = modelKey;
      L = ai.G.meta["qwen35.block_count"] - (ai.G.meta["qwen35.nextn_predict_layers"] || 0);
      layerBytes = qwen35ShardBytes(ai.G, { lo: 0, hi: 4, hasEmbed: false, hasHead: false }) / 4;
      embedBytes = (ai.G.tensors[GGML_EMBED]?.byteLength || 0) + (ai.G.tensors[GGML_OUTPUT]?.byteLength || 0) + qwen35MtpBytes(ai.G);
    } else {
      cfg = await (await fetch(M.cfg)).json();
      L = cfg.num_hidden_layers;
    }

    // real per-shard byte costs (gguf: from the file's own index)
    if (M.kind === "gguf") {
      aiStatus("reading model index\u2026");
      ai.G = await fetchGGUFHeader(M.gguf, false);
      ai.GModel = modelKey;
      layerBytes = Object.values(ggmlLayerNames(0))
        .reduce((s, nm) => s + (ai.G.tensors[nm]?.byteLength || 0), 0);
      embedBytes = (ai.G.tensors[GGML_EMBED]?.byteLength || 0) + (ai.G.tensors[GGML_OUTPUT]?.byteLength || 0);
    } else if (M.kind === "safetensors") {
      const d = cfg.hidden_size;
      const kvDim = cfg.num_key_value_heads * ((cfg.head_dim || d / cfg.num_attention_heads));
      layerBytes = (2 * d * d + 2 * kvDim * d + 3 * cfg.intermediate_size * d) * 4;
      embedBytes = cfg.vocab_size * d * 4;
    }
    ai.dims = { L, layerBytes, embedBytes };
    ai.modelKey = modelKey;
    ai.plan = null; ai.exclude = new Set(); ai.speed = new Map();
    // the first plan goes through the same queue as every later one, so a device leaving
    // during the download re-plans right after the host's own load
    const p = ai.loadQ.then(() => applyPlan("start"));
    ai.loadQ = p.catch(() => {});   // the queue itself never stays rejected
    await p;
  } catch (err) {
    clearInterval(ai.progTimer);
    aiLoading(false);
    ai.engine = null;
    ai.plan = null; ai.chain = [];
    $("ai-panel").classList.remove("online");
    aiStatus("failed: " + err.message);
    ai.busy = false;
    setState("idle", "");
    $("ai-start").disabled = false;
    $("ai-model").disabled = false;
  }
}

// deal (or re-deal) the layers over the devices in the room and get every device loading its
// range; called only from schedulePlan (queued) or aiStart, never while an answer is in flight
async function applyPlan(reason) {
  if (ai.gen) { ai.planWanted = ai.planWanted || reason; return; }   // the chain is frozen for the answer: queue instead
  const v = ++ai.planV;
  const prev = ai.plan;
  const M = MODELS[ai.modelKey];
  const first = !ai.engine;
  const nameOf = (id) => id === peer.id ? "you" : (conns.get(id)?.name || id);
  const plan = planSplit({ ...ai.dims, ...aiMembersForPlan(), prev, exclude: ai.exclude });
  if (!plan.fits) {
    ai.plan = null; ai.chain = []; ai.busy = "wait"; ai.layersByName = {};
    log("swarm", `${M.label} \u2014 the room holds ${plan.held} of ${plan.L} layers${reason ? ` (${reason})` : ""} \u2014 waiting for a device`);
    setState("waiting", `waiting for a device \u2014 room holds ${plan.held} of ${plan.L} layers`);
    $("ai-empty").textContent = "waiting for a device \u2014 share the room code";
    return;
  }
  const needGB = plan.needBytes / 2 ** 30, haveGB = plan.haveBytes / 2 ** 30;
  if (first && needGB > haveGB * 1.15)
    log("swarm", `\u26a0 this model needs ~${needGB.toFixed(1)} GB but the room pledged ~${haveGB.toFixed(1)} GB \u2014 it may not fit`);
  ai.plan = { ...plan, v };
  ai.chain = plan.chain.map((c) => c.id);
  ai.readyPeers = new Set();
  const label = (r) => `${r[0]}\u2013${r[1] - 1}`;
  ai.layersByName = Object.fromEntries([[myName, label(plan.hostRange)], ...plan.chain.map((c) => [conns.get(c.id)?.name || c.id, label(c.range)])]);
  ai.busy = ai.busy === true ? true : "plan";
  plan.chain.forEach((c, i) => sendTo(c.id, { t: "ai-load", v, model: ai.modelKey, range: c.range, next: i + 1 < plan.chain.length ? plan.chain[i + 1].id : "host", host: peer.id }));
  for (const id of plan.idle) sendTo(id, { t: "ai-wait", v });
  const note = prev ? planNote(prev, plan, nameOf) : "";
  log("swarm", describeSplit(plan, nameOf, M.label) + (!first && reason ? ` (${reason})` : ""));
  if (prev) log("swarm", note || "layer split unchanged");
  setState(first ? "loading" : "redealing", first ? "" : (note || "re-dealing layers"));
  const hostSame = ai.engine && ai.model === ai.modelKey && prev && prev.hostRange[0] === plan.hostRange[0] && prev.hostRange[1] === plan.hostRange[1];
  try {
    if (!hostSame) await aiLoadShard(ai.modelKey, plan.hostRange, true, true, { quiet: !first });
    if (v !== ai.planV) return;   // superseded while loading
    const n = ai.chain.length + 1;
    if (first) aiStatus(n === 1
      ? `solo: all ${ai.dims.L} layers local \u2014 ready`
      : `layers ${label(plan.hostRange)} ready \u00b7 syncing with ${ai.chain.length} device${ai.chain.length > 1 ? "s" : ""}\u2026`);
    aiMaybeReady();
  } catch (err) {
    if (first) throw err;   // aiStart's catch resets the panel
    log("swarm", "host reload failed: " + err.message);
    setState("waiting", "host reload failed: " + err.message);
  }
}
// ask for a re-plan; coalesces bursts (300 ms) and waits for the answer in flight (aiGenerate's
// finally picks planWanted up), then runs on the load queue behind any load still going
function schedulePlan(reason) {
  ai.planWanted = reason;
  if (ai.busy === "gen" || ai.gen) return;
  clearTimeout(ai.planTimer);
  ai.planTimer = setTimeout(() => {
    if (ai.busy === "gen" || ai.gen) return;   // a question started inside the 300 ms: planWanted stays, aiGenerate's tail re-schedules
    const r = ai.planWanted; ai.planWanted = null;
    if (r === null) return;
    ai.loadQ = ai.loadQ.then(() => applyPlan(r)).catch((e) => log("swarm", "plan failed: " + e.message));
  }, 300);
}
// a device left (connection closed, ICE failed or pongs stopped): stop the answer it was
// serving and re-plan without it. Idempotent per peer id.
function peerGone(id, name) {
  if (ai.role !== "host" || ai.state === "idle" || gone.has(id)) return;
  gone.add(id);
  name = name || conns.get(id)?.name || id;
  if (ai.gen?.chain.includes(id)) {
    ai.stopReason = `stopped: ${name} left (layers ${ai.layersByName?.[name] || "?"})`;
    rejectLaps(new Error(ai.stopReason));
  }
  ai.speed.delete(id);
  ai.readyPeers.delete(id);
  if (ai.chain.includes(id) || ai.busy === "wait") schedulePlan(`${name} left`);
}
// non-host: the host is gone, so is the room
function hostGone() {
  if (ai.role === "host" || ai.hostGone) return;
  ai.hostGone = true;
  if (botEl) chatBotEnd(ai.remoteReply || "", "stopped: the host left");
  aiLoading(false);
  $("ai-send").disabled = false;
  $("ai-panel").classList.remove("online");
  document.body.dataset.state = "over";
  aiStatus("the host left \u2014 this room is over");
  $("ai-empty").textContent = "the host left. create a new room to keep going.";
}

// host: the current plan is served once every chain member reported ready for it
function aiMaybeReady() {
  if (ai.role !== "host" || !ai.engine || !ai.plan || ai.busy === "gen" || ai.gen) return;   // a stray ready must not unlock a running answer
  if (!ai.chain.every((id) => ai.readyPeers.has(id))) return;
  ai.busy = false;
  clearInterval(ai.progTimer);
  const n = ai.chain.length + 1;
  aiStatus(`cluster online — ${n} device${n > 1 ? "s" : ""}, ${ai.cfg.num_hidden_layers} layers split ${n} ways`);
  $("ai-panel").classList.add("online");
  $("ai-row").style.display = "flex";
  $("ai-empty").textContent = "cluster online. ask anything.";
  $("ai-prompt").focus();
  setState("online", "");
  broadcastAll({ t: "ai-ready-all" });
  mascot("Cluster online! Ask anything. Everyone in the room can.");
  if (ai.planWanted) schedulePlan(ai.planWanted);
}

// one outstanding lap: resolved by ai-hiddenret(-b), rejected by the timer or by rejectLaps
// (a chain member left). Resolving or rejecting clears the timer and the map entry.
function waitLap(key, ms) {
  if (ai.stopReason) return Promise.reject(new Error(ai.stopReason));
  return new Promise((resolve, reject) => {
    const w = { res: null, rej: null, timer: null };
    const done = () => { clearTimeout(w.timer); ai.waiters.delete(key); };
    w.res = (h) => { done(); resolve(h); };
    w.rej = (err) => { done(); reject(err); };
    w.timer = setTimeout(() => w.rej(new Error("pipeline timeout (peer gone?)")), ms);
    ai.waiters.set(key, w);
  });
}
function rejectLaps(err) { for (const w of [...ai.waiters.values()]) w.rej(err); }

// run one token through the whole pipeline, returns logits
async function aiPipeToken(id, needLogits = true) {
  const pos = ai.pos;
  const chain = ai.gen ? ai.gen.chain : ai.chain;   // the chain is frozen for the answer
  if (!chain.length && !needLogits) {
    // solo prefill: layers only, no head, no readback; sync every 8 tokens
    ai.engine.pos = pos;
    await ai.engine.prefillToken(id);
    if (pos % 8 === 7) await ai.device.queue.onSubmittedWorkDone();
    ai.pos++;
    return null;
  }
  let h = await ai.engine.embedRun(id, pos);
  if (badF32(h)) throw new Error(`NaN after HOST layers (pos ${pos}) — host GPU kernel issue`);
  if (chain.length) {
    const returned = waitLap(pos, 30000);
    sendHidden(chain[0], { t: "ai-hidden", pos, v: ai.gen?.v, ...packWire(h) });
    h = await returned;
    if (badF32(h)) throw new Error(`NaN in hidden returned by peers (pos ${pos}) — check peer status lines`);
    ai.lastHidden = h;
  } // solo mode: engine holds every layer, embedRun already produced the final hidden
  if (!needLogits) { ai.pos++; return null; }   // prefill: skip the head entirely
  const logits = await ai.engine.headFromHidden(h);
  if (badF32(logits)) throw new Error(`NaN in logits (pos ${ai.pos}) — head/lm_head kernel issue on host`);
  ai.pos++;
  return logits;
}


async function aiGenerate(textArg, who) {
  const text = (textArg ?? $("ai-prompt").value).trim();
  const asker = who || myName;
  if (!text || ai.busy || !ai.engine || ai.state !== "online") return;
  try { ai.engine.reset?.(); } catch {}
  ai.pos = 0;
  broadcastAll({ t: "ai-reset" });
  ai.busy = "gen";
  // the chain serving this answer is frozen: a plan arriving now waits for ai-gendone, and a
  // plan already ticking down its 300 ms is held back too (planWanted keeps the reason)
  clearTimeout(ai.planTimer);
  ai.gen = { v: ai.planV, chain: ai.chain.slice() };
  const chain = ai.gen.chain;
  for (const id of chain) { const e = conns.get(id); if (e) e.pongAt = Date.now(); }   // fresh 2 s liveness budget
  ai.stopReason = null;
  setState("generating", "");
  $("ai-prompt").value = "";
  $("ai-send").disabled = true;
  let imEnd, eot;
  try {
    const V = ai.tok.vocab;
    const imStart = V["<|im_start|>"]; imEnd = V["<|im_end|>"]; eot = V["<|endoftext|>"];
    const ids = [imStart, ...ai.tok.encode("user\n" + text), imEnd,
      ...ai.tok.encode("\n"), imStart, ...ai.tok.encode("assistant\n")];
    // qwen3 thinking models: pre-close the think block so answers come straight
    if (V["<think>"] !== undefined && V["</think>"] !== undefined)
      ids.push(V["<think>"], ...ai.tok.encode("\n\n"), V["</think>"], ...ai.tok.encode("\n\n"));
    if (ids.some((t) => !Number.isInteger(t)))
      throw new Error("tokenizer produced an invalid token id (special tokens missing) \u2014 " + JSON.stringify(ids.slice(0, 6)));

    chatUser(asker, text);
    chatBotStart();
    broadcastAll({ t: "ai-genstart", name: asker, text });
    mascot("Thinking… every word is taking a lap through the room.");
    aiStatus(`prefill: ${ids.length} tokens…`);

    // the prompt must fit the context with room for an answer; never silently truncate
    if (ids.length > MAX_SEQ - MIN_ROOM)
      throw new Error(`prompt is ${ids.length} tokens; this room's context is ${MAX_SEQ} tokens and an answer needs at least ${MIN_ROOM}. Shorten the prompt.`);
    const maxNew = Math.min(MAX_NEW, MAX_SEQ - ids.length);   // answer cap for this prompt
    let capped = false;   // set when generation stops because the context filled up
    let logits = null;
    const tPre = performance.now();
    if (!chain.length && ai.engine.prefillTokens && ids.length > 1) {
      // solo: batched prefill, 4 prompt tokens per GPU pass
      ai.engine.pos = ai.pos;
      await ai.engine.prefillTokens(ids.slice(0, -1));
      ai.pos = ai.engine.pos;
      logits = await aiPipeToken(ids[ids.length - 1]);
    } else if (ai.engine.embedRunBatch && ids.length > 5) {
      // split: up to 16 prompt tokens per network round (4 GPU passes of 4)
      let i = 0;
      const hdim = ai.engine.dims.dim;
      const NC = ai.engine.NC || 4;   // columns per GPU pass; up to 16 tokens per network round
      // step down 16 -> 8 -> 4 on the tail: without this a remainder of up to
      // NC-1 tokens costs one network lap each
      const widths = [NC, ...[8, 4].filter((w) => w < NC)];
      for (const W of widths) while (ids.length - 1 - i >= W) {
        const nChunks = Math.max(1, Math.min(Math.floor(16 / W), Math.floor((ids.length - 1 - i) / W)));
        const NCW = W;
        const basePos = ai.pos;
        const hb = new Float32Array(nChunks * NCW * hdim);
        for (let c = 0; c < nChunks; c++)
          hb.set(await ai.engine.embedRunBatch(ids.slice(i + c * NCW, i + (c + 1) * NCW), basePos + c * NCW), c * NCW * hdim);
        if (badF32(hb)) throw new Error(`NaN in batched prefill (pos ${basePos})`);
        if (chain.length) {
          const returned = waitLap("b" + basePos, 90000);
          sendHidden(chain[0], { t: "ai-hidden-b", basePos, n: nChunks * NCW, v: ai.gen.v, ...packWire(hb) });
          await returned;
        }
        ai.pos = basePos + nChunks * NCW;
        i += nChunks * NCW;
        aiStatus(`prefill: ${i}/${ids.length} tokens\u2026`);
      }
      for (; i < ids.length; i++) logits = await aiPipeToken(ids[i], i === ids.length - 1);
    } else {
      for (let i = 0; i < ids.length; i++) logits = await aiPipeToken(ids[i], i === ids.length - 1);
    }
    const t0 = performance.now();
    let count = 0, reply = "";
    const emit = (tok) => {
      const piece = ai.tok.decode([tok]);
      reply += piece;
      count++;
      chatBotUpdate(reply);
      broadcastAll({ t: "ai-token", text: piece });
      aiStatus(`generating… ${count} tok · ${(count / ((performance.now() - t0) / 1000)).toFixed(1)} tok/s`);
    };
    if (ai.engine.mtp && ai.engine.specStep) {
      // speculative decoding: the model's own draft head proposes up to 3 tokens,
      // one batched trunk pass verifies them (byte-identical to plain decoding)
      const spec = chain.length ? {
        runTrunk: async (tokens, pos) => {
          const tLap = performance.now();
          const n = tokens.length, hdim = ai.engine.dims.dim, NC = ai.engine.NC || 4;
          const hb = new Float32Array(n * hdim);
          for (let c = 0; c < n; c += NC) {
            const m = Math.min(NC, n - c);
            hb.set(await ai.engine.embedRunBatch(tokens.slice(c, c + m), pos + c, { base: c, total: n }), c * hdim);
          }
          if (badF32(hb)) throw new Error(`NaN after HOST layers (pos ${pos})`);
          const returned = waitLap("b" + pos, 90000);
          sendHidden(chain[0], { t: "ai-hidden-b", basePos: pos, n: tokens.length, spec: 1, v: ai.gen.v, ...packWire(hb) });
          const h = await returned;
          if (badF32(h)) throw new Error(`NaN in hidden returned by peers (pos ${pos})`);
          const dt = performance.now() - tLap;
          ai.lapMs = ai.lapMs ? 0.7 * ai.lapMs + 0.3 * dt : dt;
          return h;
        },
        onReject: async (k) => { for (const id of chain) sendTo(id, { t: "ai-rollback", k }); },
      } : {};
      if (chain.length && ai.lastHidden) ai.engine.setHidden(ai.lastHidden);
      ai.engine.pos = ai.pos;
      ai.lapMs = 0;
      // draft depth: pick by MEASURED tokens/sec per depth (K=3 warm-up, probe
      // 5 and 7 once, keep the best, re-probe now and then). Deep chains only
      // pay when the network round-trip dominates the lap; a lap-time
      // threshold can't tell GPU time from RTT and gets stuck deep.
      const kc = { cand: [3, 5, 7], ema: {}, n: {}, step: 0, used: {} };
      const pickK = () => {
        if (!chain.length) return 3;
        kc.step++;
        if (kc.step <= 3) return 3;
        const untried = kc.cand.find((k) => !kc.n[k]);
        if (untried) return untried;
        let best = 3;
        for (const k of kc.cand) if (kc.ema[k] > kc.ema[best]) best = k;
        if (kc.step % 16 === 0) { const alt = kc.cand.filter((k) => k !== best); return alt[(kc.step / 16) % alt.length | 0]; }
        return best;
      };
      // the first answer token is sampled here; specStep treats it as already chosen for this
      // position and returns only the tokens after it, so it has to be emitted (or end the
      // answer) before the loop, or the reply starts one word late
      let next = aiSample(logits), done = false;
      if (next === imEnd || next === eot) done = true; else emit(next);
      while (!done && count < maxNew) {
        if (ai.stopReason) throw new Error(ai.stopReason);
        // a speculative step touches positions pos .. pos+K (K drafts verified in one pass) and
        // drafts one more; shrink K near the end of the context and stop before it overflows
        let K = pickK();
        const roomLeft = MAX_SEQ - ai.engine.pos - 2;
        if (roomLeft < 1) { capped = true; break; }
        K = Math.min(K, roomLeft, maxNew - count + 1);
        const tStep = performance.now();
        const toks = await ai.engine.specStep(next, aiSample, K, spec);
        const tps = toks.length / ((performance.now() - tStep) / 1000);
        kc.ema[K] = kc.n[K] ? 0.6 * kc.ema[K] + 0.4 * tps : tps;
        kc.n[K] = (kc.n[K] || 0) + 1; kc.used[K] = (kc.used[K] || 0) + toks.length;
        for (const tk of toks) {
          if (tk === imEnd || tk === eot) { done = true; break; }
          if (count >= maxNew) { done = true; capped = maxNew < MAX_NEW; break; }
          emit(tk);
        }
        next = toks[toks.length - 1];
      }
      if (!done && count >= maxNew) capped = maxNew < MAX_NEW;
      ai.pos = ai.engine.pos;
      const st = ai.engine.mtp.stats;
      if (st.drafts) crumb(`spec: ${st.accepted}/${st.drafts} drafts accepted${ai.lapMs ? ` · lap ${Math.round(ai.lapMs)}ms` : ""}`
        + (chain.length ? ` · K tok/s ${kc.cand.map((k) => `${k}:${kc.ema[k] ? kc.ema[k].toFixed(1) : "-"}`).join(" ")} · tokens by K ${JSON.stringify(kc.used)}` : ""));
    } else {
      for (let i = 0; i < maxNew; i++) {
        if (ai.stopReason) throw new Error(ai.stopReason);
        const next = aiSample(logits);
        if (next === imEnd || next === eot) { await aiPipeToken(next, false); break; }
        emit(next);
        if (ai.pos >= MAX_SEQ - 1) { capped = true; break; }   // no position left for another token
        logits = await aiPipeToken(next);
      }
    }
    const secs = (performance.now() - t0) / 1000;
    const stats = `${count} tok · ${(count / secs).toFixed(1)} tok/s · ${chain.length + 1} devices${capped ? ` · stopped: context full (${MAX_SEQ} tokens)` : ""}`;
    chatBotEnd(reply, stats);
    broadcastAll({ t: "ai-gendone", stats });
    mascot("Done. Anyone in the room can ask the next one.");
    aiStatus(`ready — prefill ${((t0 - tPre) / 1000).toFixed(1)}s, ${stats}`);
  } catch (err) {
    // "stopped: X left" is the room changing shape, not a failure: no warning sign, the
    // re-deal is already scheduled
    const stopped = /^stopped:/.test(err.message);
    aiStatus(stopped ? err.message : "generation failed: " + err.message);
    chatBotEnd(stopped ? err.message : "\u26a0 " + err.message, "");
    broadcastAll({ t: "ai-gendone", stats: stopped ? err.message : "failed: " + err.message });   // unlock everyone's send box
  }
  ai.gen = null;
  ai.busy = false;
  $("ai-send").disabled = false;
  if (ai.planWanted) schedulePlan(ai.planWanted); else setState("online", "");
}

// ---- worker + shared message handling ----
async function aiOnData(from, d) {
  const e = conns.get(from);
  switch (d.t) {
    case "ai-wait":   // no layers in plan v: free the GPU, keep the cached bytes
      if (ai.role === "host" || !(d.v > (ai.planV || 0))) break;
      ai.planV = d.v;
      ai.role = "worker"; ai.hostId = from;
      ai.loadQ = ai.loadQ.then(aiFree).catch(() => {});
      aiStatus("standing by \u2014 no layers this round, yours stay cached");
      break;
    case "ai-load": {
      if (ai.role === "host") break;
      if (!(d.v > (ai.planV || 0))) break;   // older plan, or a host without versions
      ai.planV = d.v;
      if (MODELS[d.model]) $("ai-model").value = d.model;
      ai.role = "worker";
      ai.next = d.next;
      ai.hostId = d.host;
      ai.loadQ = ai.loadQ.then(async () => {
        if (d.v !== ai.planV) return;   // superseded while an earlier load was running
        ai.next = d.next; ai.hostId = d.host;
        // same model and range as the engine already holds: rewire only, no reload
        const same = ai.engine && ai.model === d.model && ai.range?.[0] === d.range[0] && ai.range?.[1] === d.range[1];
        ensureLink(d.next);   // open the link to my chain neighbour while the weights download
        try {
          if (!same) await aiLoadShard(d.model || "smollm-135m", d.range, false, false, { quiet: !!ai.engine });
          if (!(await ensureLink(d.next))) throw new Error("could not connect to the next device in the chain");
          if (d.v !== ai.planV) return;
          aiStatus(`layers ${d.range[0]}\u2013${d.range[1] - 1} ready \u00b7 syncing with the room\u2026`);
          if ($("ai-loading").style.display !== "none") {
            aiLoading(true, `layers ${d.range[0]}\u2013${d.range[1] - 1} ready`);
            $("ldg-sub").textContent = "syncing with the rest of the room";
            $("ldg-fill").style.width = "100%";
          }
          sendTo(ai.hostId, { t: "ai-ready", v: d.v, ms: ai.tune?.results?.[0]?.ms });
        } catch (err) {
          aiLoading(false);
          aiStatus("failed: " + err.message);
          sendTo(ai.hostId, { t: "ai-error", v: d.v, message: err.message });
        }
      }).catch((err) => aiStatus("failed: " + err.message));
      break;
    }
    case "ai-hostprog": {
      const now = Date.now();
      ai.prog = { ...(d.all || {}), [myName]: Math.round(ai.myPct || 0) };
      ai.progAt = ai.progAt || {};
      for (const nm of Object.keys(d.all || {})) if (nm !== myName) ai.progAt[nm] = now;
      loadCardRender();
      break;
    }
    case "ai-progress":
      if (e?.card) e.card.querySelector(".bw").textContent = "dl " + d.pct + "%";
      ai.prog = ai.prog || {}; ai.progAt = ai.progAt || {};
      ai.prog[e?.name || from] = d.pct; ai.progAt[e?.name || from] = Date.now(); loadCardRender();
      if (ai.state === "redealing" && /^re-dealing/.test($("ai-status").textContent)) aiStatus(`re-dealing: ${e?.name || from} loading ${d.pct}%`);
      break;
    case "ai-reset": try { ai.engine?.reset?.(); } catch {} break;
    case "ai-layers":   // room state broadcast from the host
      if (ai.role === "host" || d.v < (ai.layersV || 0)) break;
      ai.layersV = d.v;
      ai.hostId = ai.hostId || from;
      if (MODELS[d.model]) $("ai-model").value = d.model;
      $("ai-start").disabled = true; $("ai-model").disabled = true;
      ai.layersByName = d.by; loadCardRender();
      document.body.dataset.state = d.state; document.body.dataset.plan = String(d.v);
      if (d.note) aiStatus(d.note);
      $("ai-send").disabled = !(d.state === "online" || d.state === "generating");
      if (d.state === "waiting") $("ai-empty").textContent = "waiting for a device";
      break;
    case "ai-start-req":
      if (MODELS[d.model]) $("ai-model").value = d.model;   // every screen shows the model that was actually started
      $("ai-start").disabled = true; $("ai-model").disabled = true;
      if (d.boss !== peer.id) { aiLoading(true, `starting ${MODELS[d.model]?.label.split("\u00b7")[0].trim()}`); $("ldg-sub").textContent = `${d.by} pressed start`; $("ldg-fill").style.width = "0%"; }
      if (d.boss === peer.id) { toast(`${d.by} started ${MODELS[d.model]?.label.split("\u00b7")[0].trim()}`); aiStart(d.model); }
      else aiStatus(`${d.by} started the model\u2026`);
      break;
    case "ai-ready":
      if (ai.role !== "host" || d.v !== ai.planV) break;   // a ready for an older plan
      if (d.ms > 0) ai.speed.set(from, d.ms);
      ai.readyPeers.add(from);
      if (e?.card) e.card.querySelector(".bw").textContent = "ready";
      aiMaybeReady();
      break;
    case "ai-error":
      aiStatus(`peer ${e?.name || from} failed: ${d.message}`);
      if (ai.role === "host" && d.v === ai.planV && ai.chain.includes(from)) { ai.exclude.add(from); schedulePlan(`${e?.name || from} failed to load`); }
      break;
    case "ai-hidden-b": {
      // worker: n hiddens in (multiple of 4), my layers (batched), n hiddens on
      if (!ai.engine || (d.v && d.v !== ai.planV)) return;   // a frame from another plan
      ai.inflight++;
      try {
        const xs = unpackWire(d);
        const nTok = d.n || 4;
        const wdim = ai.engine.dims.dim;
        const hb = new Float32Array(nTok * wdim);
        const NC = ai.engine.NC || 4;
        for (let c = 0; c < nTok; c += NC) {
          const m = Math.min(NC, nTok - c);
          hb.set(await ai.engine.runHiddenBatch(xs.subarray(c * wdim, (c + m) * wdim), d.basePos + c, d.spec ? { base: c, total: nTok } : false), c * wdim);
        }
        if (badF32(hb)) { aiStatus(`\u26a0 NaN in batched prefill on this device`); sendTo(ai.hostId, { t: "ai-error", message: "NaN in batched prefill" }); }
        const bmsg = { basePos: d.basePos, n: nTok, v: d.v, ...packWire(hb) };
        if (ai.next === "host") sendHidden(ai.hostId, { t: "ai-hiddenret-b", ...bmsg });
        else sendHidden(ai.next, { t: "ai-hidden-b", ...bmsg });
      } catch (err) { aiStatus("lap dropped: " + err.message); }
      finally { ai.inflight--; if (!ai.inflight) ai.quiesce.splice(0).forEach((r) => r()); }
      break;
    }
    case "ai-rollback": {
      // host rejected a speculative suffix: recurrent state back to after column k
      ai.engine?.restoreDN?.(d.k);
      break;
    }
    case "ai-hiddenret-b": {
      if (d.v && d.v !== ai.gen?.v) break;   // a lap from a plan that is no longer served
      const w = ai.waiters.get("b" + d.basePos);
      if (w) w.res(unpackWire(d));
      break;
    }
    case "ai-hidden": {
      // worker: run my layers, forward along the chain
      if (!ai.engine || (d.v && d.v !== ai.planV)) return;
      ai.inflight++;
      try {
        const hin = unpackWire(d);
        if (badF32(hin)) { aiStatus(`\u26a0 NaN ARRIVED at this device (pos ${d.pos}) — upstream peer broken`); }
        const h = await ai.engine.runHidden(hin, d.pos);
        if (badF32(h)) { aiStatus(`\u26a0 NaN PRODUCED by this device (pos ${d.pos}, layers ${ai.range[0]}\u2013${ai.range[1] - 1}) — GPU kernel issue here`); sendTo(ai.hostId, { t: "ai-error", message: `NaN produced on worker layers ${ai.range[0]}\u2013${ai.range[1] - 1}` }); }
        const msg = { pos: d.pos, v: d.v, ...packWire(h) };
        if (ai.next === "host") sendHidden(ai.hostId, { t: "ai-hiddenret", ...msg });
        else sendHidden(ai.next, { t: "ai-hidden", ...msg });
        if (d.pos % 8 === 0) aiStatus(`serving layers ${ai.range[0]}–${ai.range[1] - 1} — pos ${d.pos}`);
      } catch (err) { aiStatus("lap dropped: " + err.message); }
      finally { ai.inflight--; if (!ai.inflight) ai.quiesce.splice(0).forEach((r) => r()); }
      break;
    }
    case "ai-hiddenret": {
      // host: pipeline round-trip complete
      if (d.v && d.v !== ai.gen?.v) break;
      const w = ai.waiters.get(d.pos);
      if (w) w.res(unpackWire(d));
      break;
    }
    case "ai-genstart":
      ai.remoteReply = "";
      chatUser(d.name, d.text);
      chatBotStart();
      $("ai-send").disabled = true;
      mascot(`${d.name} asked something. Thinking…`);
      break;
    case "ai-token": ai.remoteReply = (ai.remoteReply || "") + d.text; chatBotUpdate(ai.remoteReply); break;
    case "ai-gendone": chatBotEnd(ai.remoteReply || "", d.stats); $("ai-send").disabled = false; mascot("Your turn. Ask anything."); break;
    case "ai-ready-all":   // after every completed plan, so a late joiner gets here too
      if (ai.role === "host") break;
      aiLoading(false);
      $("ai-panel").classList.add("online");
      ai.role = ai.role || "guest"; ai.hostId = from;
      $("ai-row").style.display = "flex";
      $("ai-send").disabled = false;
      $("ai-empty").textContent = "cluster online. ask anything.";
      aiStatus(ai.range ? `cluster online · serving layers ${ai.range[0]}–${ai.range[1] - 1}` : "cluster online · standing by");
      mascot("Cluster online! Type a question, the whole room answers.");
      break;
    case "ai-ask":
      if (ai.role !== "host") break;
      if (ai.busy || ai.state !== "online") { sendTo(from, { t: "ai-busy" }); break; }
      aiGenerate(d.text, d.name);
      break;
    case "ai-busy": toast("the swarm is still answering, try again in a moment"); break;
  }
}

$("ai-start").addEventListener("click", aiStartAnywhere);
// manual re-deal: the same planner, now (or right after the answer in flight)
$("ai-redeal").addEventListener("click", () => {
  if (ai.role !== "host" || ai.state === "idle") return;
  if (ai.busy === "gen") $("ai-redeal").textContent = "re-deal after this answer";
  schedulePlan("re-deal requested");
});
$("cache-clear").addEventListener("click", async (ev) => {
  ev.preventDefault();
  try { await caches.delete("swarmllm-weights-v1"); weightCache = null; toast("cached weights cleared"); } catch { toast("could not clear the cache"); }
});
function aiSubmit() {
  const text = $("ai-prompt").value.trim();
  if (!text) return;
  if (ai.role === "host") { aiGenerate(); return; }
  const hostId = ai.hostId;
  if (!conns.has(hostId)) { toast("not connected to the host"); return; }
  $("ai-prompt").value = "";
  sendTo(hostId, { t: "ai-ask", text, name: myName });
}
$("ai-send").addEventListener("click", aiSubmit);
$("ai-prompt").addEventListener("keydown", (e) => { if (e.key === "Enter") aiSubmit(); });
mascot("Hi! I'm Swarmy. Create a room, or type a friend's code to join one.");
