// SwarmLLM room: signaling, WebRTC mesh, layer assignment, weight streaming and the
// generation loop (prefill, decode, speculative verify). Served with p2p.html at /room.
import { autotuneCoop, makeTokenizer, DenseEngine, argmax, fetchModelShard, shardTensorNames, gpuSelfTest, kernelMicroTests }
  from "./engine/engine.js";
import { f32ToF16, f16ToF32, parseGGUFHeader, ggufWeights, ggufShardBytes, GGML_EMBED, GGML_OUTPUT, GGML_FINAL_NORM,
  ggmlLayerNames, qwen35Weights, qwen35ShardBytes, qwen35MtpBytes, qwen35LayerNames, tokenizerFromGGUF, gpuUploadEntry, streamEntryToGPU }
  from "./engine/gguf.js";
import { Qwen35Engine } from "./engine/qwen35.js";
import { WIRE_F16, badF32, f32ToB64, packF16, unpackF16, asU16, packWire, unpackWire, asF32, b64ToF32 } from "./room/wire.js";
import { esc, md, mdChat } from "./room/markdown.js";
import { pickSampler, SAMPLING } from "./room/sampling.js";
import { chatRecipients } from "./room/visibility.js";
import { MODELS, NEED_GB, MAX_SEQ, MAX_NEW, MAX_NEW_THINKING, MIN_ROOM } from "./room/models.js";
import { makeLink, attachWire, wireReady, sendFrame, PROTOCOL } from "./room/transport.js";
import { PERSONAS, specials, fitContext, reusablePrefix } from "./room/conversation.js";
import { planSplit, ladder, bestFit, codeFromLocation } from "./room/plan.js";
import { qrSVG } from "./room/qr.js";
import { drawCard } from "./room/card.js";
import { probe as preflight, deviceKind } from "./room/preflight.js";

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
  const meta = { ua: deviceKind({ ua: navigator.userAgent, touchPoints: navigator.maxTouchPoints || 0, mobile: !!navigator.userAgentData?.mobile }),
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

// the join screen says up front whether this browser can hold layers, and what to do if not
preflight().then((v) => { if (!v.ok && !$("join-status").textContent) { $("join-status").textContent = v.line; $("join-status").classList.add("warn"); } });
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
// The model ladder: every model with what this room still needs for it, smallest first. Until
// someone picks a model by hand, the select follows the largest model the room can run.
let modelTouched = false;
const shortName = (key) => (MODELS[key]?.label || key).split("\u00b7")[0].replace(/^Qwen\s*[\d.]+\s+/, "").trim();
function renderLadder(pledged) {
  const el = $("ai-ladder"); if (!el) return;
  el.innerHTML = ladder(NEED_GB, pledged).map((x) => `<button type="button" class="rung${x.ok ? " ok" : ""}${x.key === $("ai-model").value ? " sel" : ""}" data-k="${x.key}" title="needs ~${x.need} GB">${esc(shortName(x.key))} <b>${x.ok ? "\u2713" : "+" + x.short + " GB"}</b></button>`).join("");
}
$("ai-ladder").addEventListener("click", (e) => {
  const b = e.target.closest(".rung"); if (!b || $("ai-model").disabled) return;
  $("ai-model").value = b.dataset.k; modelTouched = true; updateCluster();
});
function updateNeed(pledged) {
  if (!modelTouched && !ai.engine && !ai.busy && !$("ai-model").disabled) $("ai-model").value = bestFit(NEED_GB, pledged);
  renderLadder(pledged);
  const need = NEED_GB[$("ai-model").value] || 1;
  const ok = pledged >= need;
  $("need-fill").style.width = Math.min(100, pledged / need * 100).toFixed(1) + "%";
  $("need-text").textContent = ok
    ? `needs ~${need} GB \u00b7 room gives ${pledged.toFixed(1)} GB \u00b7 ready`
    : `needs ~${need} GB \u00b7 room gives ${pledged.toFixed(1)} GB \u00b7 add ${(need - pledged).toFixed(1)} GB more`;
  $("ai-need").classList.toggle("ok", ok);
  if (!ai.busy && !ai.engine) $("ai-start").disabled = !ok;
  if (ok && !wasReady) { $("ai-start").classList.remove("unlocked"); void $("ai-start").offsetWidth; $("ai-start").classList.add("unlocked"); }
  wasReady = ok;
}
$("ai-model").addEventListener("change", () => { modelTouched = true; updateCluster(); });
function updateCluster() {
  const all = [myMeta, ...[...members.values()].map(m => m.meta)];
  const gpus = all.filter(m => m && m.webgpu).length;
  // only devices with WebGPU hold layers; the others join as ask-only guests
  const pledged = all.reduce((s, m) => s + (m?.webgpu ? m?.contribGB || 0 : 0), 0);
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
  $("side-code").addEventListener("click", openShare);
  if (isHost) $("host-controls").hidden = false;
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
    if (e && e.conn !== conn) return;   // an older link to the same device
    conns.delete(conn.peer);
    if (isHost) {   // on the host a closed link means the device left; workers wait for the roster
      dropCard(conn.peer); members.delete(conn.peer); roster.delete(conn.peer); broadcastRoster();
      log("swarm", `${e?.name || conn.peer} left`);
      aiPeerLeft(conn.peer, e?.name);
    } else if (conn.peer === PREFIX + roomCode) { log("swarm", "lost the link to the host"); hostGone(); }
    updateCluster();
  });
  conn.on("error", () => {});
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

function sendTo(id, obj) { conns.get(id)?.conn.send(obj); }
// debug: per-peer wire state (channels open, frames sent/received) — `swarmDebug()` in the console
window.swarmDebug = () => [...conns].map(([id, e]) => ({ id, name: e.name, chans: e.link?.chans.filter((c) => c.readyState === "open").length ?? 0, sent: e.link?.sent ?? 0, recv: e.link?.recv ?? 0 }));
// activations go over the sliced wire channel when it is up, else as a normal message
// ?netlag=ms delays every activation frame this device sends, to emulate a slow link in tests
// (equal delays keep send order)
const NETLAG = Math.max(0, parseInt(new URLSearchParams(location.search).get("netlag"), 10) || 0);
function sendHidden(id, msg) {
  if (NETLAG) { setTimeout(() => sendHiddenNow(id, msg), NETLAG); return; }
  sendHiddenNow(id, msg);
}
function sendHiddenNow(id, msg) {
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
  if (!d || typeof d.t !== "string") return;
  if (d.t.startsWith("ai-")) { aiOnData(from, d); return; }
  switch (d.t) {
    case "hello":
      // one protocol per room: a tab from an older or newer deploy is told to reload
      if (d.v !== PROTOCOL) {
        sendTo(from, { t: "bye", reason: `this room runs SwarmLLM protocol ${PROTOCOL} and your tab runs ${d.v ?? 1}: reload both pages so they match` });
        log("swarm", `${d.name || from} runs a different SwarmLLM version (protocol ${d.v ?? 1}); asked it to reload`);
        break;
      }
      e.name = d.name; e.meta = d.meta;
      members.set(from, { name: d.name, meta: d.meta });
      ensureCard(from, d.name, d.meta);
      if (isHost) {
        roster.set(from, { name: d.name, meta: d.meta }); broadcastRoster();
        aiRejoin(from, d.name);
        if (d.died) log("swarm", `${d.name} came back: its tab was killed ${d.died.ago} s ago while ${d.died.during}. Phones kill background tabs; keep the screen on.`);
        if (ai.visibility !== "all") sendTo(from, { t: "ai-visibility", mode: ai.visibility });
        aiWelcome(from);
      }
      break;
    case "leaving":   // the tab is closing: treat the link as gone now instead of waiting for ICE to time out
      conns.get(from)?.conn.close();
      break;
    case "bye":
      toast(d.reason);
      log("swarm", d.reason);
      if (from === PREFIX + roomCode) { $("room-over").hidden = false; $("room-over-why").textContent = d.reason; }
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
    conn.send({ t: "hello", name: myName, meta: myMeta, v: PROTOCOL });
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

// a closing tab says so, so the others fail fast (the data channel close can take ~30 s to surface)
window.addEventListener("pagehide", () => { try { broadcastAll({ t: "leaving" }); } catch {} });

// --- ping loop ---
setInterval(() => broadcastAll({ t: "ping", ts: performance.now() }), 2500);

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
      conn.send({ t: "hello", name: myName, meta: myMeta, died, v: PROTOCOL });
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
      conn.send({ t: "hello", name: myName, meta: myMeta, v: PROTOCOL });
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
      awakeVideo.src = "data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAbBbW9vdgAAAGxtdmhkAAAAAAAAAAAAAAAAAAAD6AAAB9AAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAy50cmFrAAAAXHRraGQAAAADAAAAAAAAAAAAAAABAAAAAAAAB9AAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAEAAAABAAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAAAfQAAAAAAABAAAAAAKmbWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAAAoAAAAUABVxAAAAAAALWhkbHIAAAAAAAAAAHZpZGUAAAAAAAAAAAAAAABWaWRlb0hhbmRsZXIAAAACUW1pbmYAAAAUdm1oZAAAAAEAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAAhFzdGJsAAAAuXN0c2QAAAAAAAAAAQAAAKlhdmMxAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAEAAQABIAAAASAAAAAAAAAABFUxhdmM2MC4zMS4xMDIgbGlieDI2NAAAAAAAAAAAAAAAGP//AAAAL2F2Y0MBQsAe/+EAF2dCwB7ZBCbARAAAAwAEAAADAFA8WLkgAQAFaMuDyyAAAAAQcGFzcAAAAAEAAAABAAAAFGJ0cnQAAAAAAAANNAAADTQAAAAYc3R0cwAAAAAAAAABAAAAFAAABAAAAAAUc3RzcwAAAAAAAAABAAAAAQAAAHBzdHNjAAAAAAAAAAgAAAABAAAAAQAAAAEAAAAFAAAAAgAAAAEAAAAGAAAAAQAAAAEAAAAJAAAAAgAAAAEAAAAKAAAAAQAAAAEAAAAMAAAAAgAAAAEAAAANAAAAAQAAAAEAAAAQAAAAAgAAAAEAAABkc3RzegAAAAAAAAAAAAAAFAAAAo8AAAAKAAAACgAAAAoAAAAKAAAACgAAAAoAAAAKAAAACgAAAAoAAAAKAAAACgAAAAoAAAAKAAAACgAAAAoAAAAKAAAACgAAAAoAAAAKAAAAUHN0Y28AAAAAAAAAEAAABwYAAAmZAAAJpwAACbUAAAnDAAAJ2wAACekAAAn3AAAKBQAACh0AAAorAAAKOQAAClEAAApfAAAKbQAACnsAAAK9dHJhawAAAFx0a2hkAAAAAwAAAAAAAAAAAAAAAgAAAAAAAAfQAAAAAAAAAAAAAAABAQAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAJGVkdHMAAAAcZWxzdAAAAAAAAAABAAAH0AAABAAAAQAAAAACNW1kaWEAAAAgbWRoZAAAAAAAAAAAAAAAAAAAH0AAAEKAVcQAAAAAAC1oZGxyAAAAAAAAAABzb3VuAAAAAAAAAAAAAAAAU291bmRIYW5kbGVyAAAAAeBtaW5mAAAAEHNtaGQAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAAaRzdGJsAAAAfnN0c2QAAAAAAAAAAQAAAG5tcDRhAAAAAAAAAAEAAAAAAAAAAAABABAAAAAAH0AAAAAAADZlc2RzAAAAAAOAgIAlAAIABICAgBdAFQAAAAAAH0AAAAE/BYCAgAUViFblAAaAgIABAgAAABRidHJ0AAAAAAAAH0AAAAE/AAAAIHN0dHMAAAAAAAAAAgAAABAAAAQAAAAAAQAAAoAAAAAcc3RzYwAAAAAAAAABAAAAAQAAAAEAAAABAAAAWHN0c3oAAAAAAAAAAAAAABEAAAAVAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAAQAAAAEAAAABAAAAFRzdGNvAAAAAAAAABEAAAbxAAAJlQAACaMAAAmxAAAJvwAACdcAAAnlAAAJ8wAACgEAAAoZAAAKJwAACjUAAApNAAAKWwAACmkAAAp3AAAKjwAAABpzZ3BkAQAAAHJvbGwAAAACAAAAAf//AAAAHHNiZ3AAAAAAcm9sbAAAAAEAAAARAAAAAQAAAGJ1ZHRhAAAAWm1ldGEAAAAAAAAAIWhkbHIAAAAAAAAAAG1kaXJhcHBsAAAAAAAAAAAAAAAALWlsc3QAAAAlqXRvbwAAAB1kYXRhAAAAAQAAAABMYXZmNjAuMTYuMTAwAAAACGZyZWUAAAOqbWRhdN4CAExhdmM2MC4zMS4xMDIAAjBADgAAAnEGBf//bdxF6b3m2Ui3lizYINkj7u94MjY0IC0gY29yZSAxNjQgcjMxMDggMzFlMTlmOSAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjMgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0wIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDE6MHgxMTEgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTAgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz0yIGxvb2thaGVhZF90aHJlYWRzPTEgc2xpY2VkX3RocmVhZHM9MCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MCB3ZWlnaHRwPTAga2V5aW50PTI1MCBrZXlpbnRfbWluPTEwIHNjZW5lY3V0PTQwIGludHJhX3JlZnJlc2g9MCByY19sb29rYWhlYWQ9NDAgcmM9Y3JmIG1idHJlZT0xIGNyZj0yMy4wIHFjb21wPTAuNjAgcXBtaW49MCBxcG1heD02OSBxcHN0ZXA9NCBpcF9yYXRpbz0xLjQwIGFxPTE6MS4wMACAAAAAFmWIhA/yYoAAw+ycnJ1111111111114BGCAHAAAABkGaOB/hGAEYIAcAAAAGQZpUB/hGARggBwAAAAZBmmA/wjABGCAHAAAABkGagD/CMAAAAAZBmqA/wjABGCAHAAAABkGawD/CMAEYIAcAAAAGQZrgP8IwARggBwAAAAZBmwA/wjABGCAHAAAABkGbID/CMAAAAAZBm0A/wjABGCAHAAAABkGbYD/CMAEYIAcAAAAGQZuAP8IwARggBwAAAAZBm6A/wjAAAAAGQZvAP8IwARggBwAAAAZBm+A/wjABGCAHAAAABkGaAD/CMAEYIAcAAAAGQZogP8IwARggBwAAAAZBmkA7wjAAAAAGQZpgN8IwARggBw==";
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
// Join links: swarmllm.ai/r/ABCD opens this page and joins the room with no typing. Served
// elsewhere (a local static server, the emulator), the link keeps this page's path and query
// (signal=, wire=) and adds ?code=.
function roomLink() {
  if (location.pathname === "/room" || location.pathname.startsWith("/r/")) return `${location.origin}/r/${roomCode}`;
  const q = new URLSearchParams(location.search); q.set("code", roomCode);
  return `${location.origin}${location.pathname}?${q}`;
}
function copyRoomLink() {
  const url = roomLink();
  if (!navigator.clipboard) { toast("room code: " + roomCode); return; }
  navigator.clipboard.writeText(url).then(() => toast("join link copied")).catch(() => { navigator.clipboard.writeText(roomCode); toast("room code copied"); });
}
function openShare() {
  const url = roomLink();
  $("share-qr").innerHTML = qrSVG(url, { size: 220 });
  $("share-url").textContent = url;
  $("share-code").textContent = roomCode;
  $("share-native").hidden = !navigator.share;
  $("share").hidden = false;
}
$("room-badge").addEventListener("click", openShare);
$("share-btn").addEventListener("click", openShare);
$("share-close").addEventListener("click", () => { $("share").hidden = true; });
$("share").addEventListener("click", (e) => { if (e.target === $("share")) $("share").hidden = true; });
$("share-copy").addEventListener("click", copyRoomLink);
$("share-native").addEventListener("click", () => navigator.share?.({ title: "Join my SwarmLLM room", text: `Room ${roomCode}: lend this device's GPU to a model we run together`, url: roomLink() }).catch(() => {}));
$("room-over-new").addEventListener("click", () => { location.href = location.pathname.startsWith("/r/") ? "/room" : location.pathname.replace(/\?.*$/, ""); });
// a link with a room code fills it in and joins once the GPU probe is done
const linkCode = codeFromLocation(location.pathname, location.search, location.hash);
if (linkCode) {
  $("code-input").value = linkCode;
  $("join-status").textContent = `joining room ${linkCode}\u2026`;
  metaPromise.then(() => { if (!peer) start(false); });
}

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
  visibility: "all",   // who sees the chat: all | host | asker (room/visibility.js)
  engine: null, tok: null, cfg: null, device: null,
  role: null,            // "host" | "worker" | "guest"
  chain: [],             // host: worker peer ids in pipeline order
  next: null,            // worker: peer id to forward hidden to, or "host"
  readyPeers: new Set(),
  pos: 0,
  waiters: new Map(),    // host: lap key (pos, or "b" + basePos) -> { res, rej } for a frame on its way round the chain
  busy: false,
  abort: false,          // host: Stop was pressed; the decode loop ends after the lap in flight
  degraded: false,       // host: a device in the chain left; generation needs a re-deal first
  askerId: null,         // host: who asked the question being answered
  conv: { turns: [] },   // host: the conversation (room/conversation.js)
  fed: [],               // host: the exact tokens every device's caches hold, in order; null = unknown, reset first
  pendingCtl: {},        // host: control for the chain that rides on the next frame ({ reset } or { rb })
  settings: { persona: "default", sampling: "creative", thinking: false, length: "normal" },
  transcript: [],        // host: [{ name, text, reply, stats }] for devices that join later
  teleBy: new Map(),     // host: worker id -> compute ms per frame kind, from ai-tele
  q: Promise.resolve(),  // worker: frames run strictly one after another, in arrival order
};

function aiStatus(s) { $("ai-status").textContent = s; crumb(s); }
// breadcrumb: if iOS kills the tab, the reloaded page can say where it died
function crumb(s) { try { localStorage.setItem("swarm-crumb", JSON.stringify({ s, t: Date.now(), mem: performance.memory?.usedJSHeapSize })); } catch {} }
// (crumb is kept in localStorage for debugging, not shown on the join screen)
function aiLoading(show, title) {
  $("ai-loading").style.display = show ? "block" : "none";
  if (title) $("ldg-title").textContent = title;
  $("ai-panel").classList.toggle("loading", !!show);
  $("load-card").classList.toggle("on", !!show);
  $("ai-empty").style.display = show ? "none" : "";
  if (show) { $("lc-model").textContent = MODELS[$("ai-model").value]?.label.split("·")[0].trim() || ""; loadCardRender(); }
}
function loadCardRender() {
  const rows = $("lc-rows"); if (!rows) return;
  const names = [myName, ...[...conns.values()].map((c) => c.name)];
  const layersOf = (nm) => (ai.layersByName || {})[nm];
  rows.innerHTML = names.map((nm) => {
    const pct = Math.max(0, Math.min(100, (ai.prog || {})[nm] ?? 0));
    const l = layersOf(nm);
    return `<div class="lc-row${pct >= 100 ? " done" : ""}"><div class="n">${esc(String(nm))}${l ? `<small>layers ${l}</small>` : ""}</div><div class="bar"><div class="fill" style="width:${pct}%"></div></div><div class="pct">${pct >= 100 ? "ready" : pct + "%"}</div></div>`;
  }).join("");
}
function aiProgress(done, total, note) {
  const pct = total ? Math.min(100, Math.round(done / total * 100)) : 0;
  $("ldg-fill").style.width = pct + "%";
  $("ldg-sub").textContent = `${(done / 2 ** 20).toFixed(0)} MB of ${(total / 2 ** 20).toFixed(0)} MB · ${pct}%` + (note ? " · " + note : "");
}
function aiOut() { const o = $("ai-output"); o.style.display = "block"; $("ai-empty").style.display = "none"; return o; }

// ---- chat transcript ----
// A bot message keeps its answer as pieces ({ t: text, d: 1 when the token was an accepted
// speculative draft }) so "show drafts" can re-render it with the drafted tokens marked.
let botEl = null;
let draftView = false;
function scrollChat() { const o = $("ai-output"); o.scrollTop = o.scrollHeight; }
function chatUser(name, text) {
  const o = aiOut();
  const m = document.createElement("div");
  m.className = "m user";
  m.innerHTML = `<div class="who">${esc(name)}</div><div class="bubble">${esc(text).replace(/\n/g, "<br>")}</div>`;
  m.dataset.name = name; m.dataset.text = text;
  o.appendChild(m); scrollChat();
}
function chatBotStart(mid) {
  const o = aiOut();
  const m = document.createElement("div");
  m.className = "m bot";
  if (mid != null) m.dataset.mid = mid;
  m.innerHTML = `<div class="who">swarm</div><div class="bubble"><span class="cursor"></span></div>`;
  m.pieces = [];
  o.appendChild(m); scrollChat();
  botEl = m;
}
function renderBot(m, live) {
  const b = m.querySelector(".bubble");
  if (draftView && m.pieces.length) {
    b.classList.add("drafts");
    b.innerHTML = m.pieces.map((p) => p.d ? `<span class="dr">${esc(p.t)}</span>` : esc(p.t)).join("") + (live ? '<span class="cursor"></span>' : "");
  } else {
    b.classList.remove("drafts");
    b.innerHTML = mdChat(m.pieces.map((p) => p.t).join("")) + (live ? '<span class="cursor"></span>' : "");
  }
}
function chatBotPiece(text, d) {
  if (!botEl) chatBotStart();
  botEl.pieces.push({ t: text, d: d ? 1 : 0 });
  renderBot(botEl, true);
  scrollChat();
}
function chatBotEnd(note, stats) {
  if (!botEl) chatBotStart();
  if (note) botEl.pieces = [{ t: note, d: 0 }];
  renderBot(botEl, false);
  if (stats) { const s = document.createElement("div"); s.className = "stats"; s.textContent = stats; botEl.appendChild(s); }
  if (!note && botEl.dataset.mid) {
    const r = document.createElement("div");
    r.className = "reacts";
    r.innerHTML = REACTIONS.map((e) => `<button type="button" data-e="${e}" aria-label="react ${e}">${e}<b></b></button>`).join("");
    botEl.appendChild(r);
    if (readAloud && botEl.pieces.length) speak(botEl.pieces.map((p) => p.t).join(""));
  }
  botEl = null;
}

// ---- the room's social bits: reactions, who is typing, answers read aloud ----
const REACTIONS = ["\u{1F44D}", "\u{1F525}", "\u{1F92F}", "\u{1F602}", "\u{1F41D}"];
const myReacts = new Set();   // "mid|emoji" this device has on
function renderReacts(mid, counts) {
  const m = document.querySelector(`#ai-output .m.bot[data-mid="${CSS.escape(String(mid))}"]`); if (!m) return;
  for (const b of m.querySelectorAll(".reacts button")) {
    const n = counts?.[b.dataset.e] || 0;
    b.querySelector("b").textContent = n ? String(n) : "";
    b.classList.toggle("on", n > 0);
    b.classList.toggle("mine", myReacts.has(mid + "|" + b.dataset.e));
  }
}
function hostReact(mid, e, from) {
  if (!REACTIONS.includes(e)) return;
  ai.reacts ||= new Map();
  const per = ai.reacts.get(mid) || {}; ai.reacts.set(mid, per);
  const set = per[e] ||= new Set();
  if (set.has(from)) set.delete(from); else set.add(from);
  const counts = Object.fromEntries(Object.entries(per).map(([k, v]) => [k, v.size]));
  broadcastAll({ t: "ai-reacts", mid, counts });
  renderReacts(mid, counts);
}
$("ai-output").addEventListener("click", (ev) => {
  const b = ev.target.closest(".reacts button"); if (!b) return;
  const mid = b.closest(".m.bot")?.dataset.mid; if (!mid) return;
  const key = mid + "|" + b.dataset.e;
  if (myReacts.has(key)) myReacts.delete(key); else myReacts.add(key);
  if (ai.role === "host") hostReact(mid, b.dataset.e, peer.id);
  else if (ai.hostId) sendTo(ai.hostId, { t: "ai-react", mid, e: b.dataset.e });
});
let typingAt = 0;
function noteTyping() {
  const now = Date.now();
  if (now - typingAt < 2000 || !$("ai-prompt").value.trim()) return;
  typingAt = now;
  if (ai.role === "host") { if (ai.visibility === "all") broadcastAll({ t: "ai-typing", name: myName }); }
  else if (ai.hostId && conns.has(ai.hostId)) sendTo(ai.hostId, { t: "ai-typing" });
}
let typingTimer = null;
function showTyping(name) {
  $("typing-note").textContent = `${name} is typing\u2026`;
  clearTimeout(typingTimer);
  typingTimer = setTimeout(() => { $("typing-note").textContent = ""; }, 3500);
}
let readAloud = false;
function speak(raw) {
  try {
    const text = raw.replace(/<think>[\s\S]*?(<\/think>|$)/g, "").replace(/[*_`#>]+/g, "").trim();
    if (!text) return;
    speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance(text.slice(0, 4000)));
  } catch {}
}
function setDraftView(on) {
  draftView = on;
  $("draft-view").classList.toggle("on", on);
  $("draft-view").textContent = on ? "hide drafts" : "show drafts";
  for (const m of document.querySelectorAll("#ai-output .m.bot")) if (m.pieces) renderBot(m, m === botEl);
  if (on) toast("tinted words were guessed by the draft head and confirmed by the whole swarm in one lap");
}
// the swarm card: this room's best finished answer speed, its devices and layers, as a PNG
function openCard() {
  const nodes = lastMap?.nodes?.length ? lastMap.nodes : [{ name: myName, layers: "", host: 1 }];
  const tps = bestTps || lastMap?.st?.tps || lastSoloTps || 0;
  drawCard($("card-canvas"), { model: (MODELS[ai.model || $("ai-model").value]?.label || "").split("\u00b7")[0].trim(),
    code: roomCode, nodes, tps, acc: lastMap?.st?.acc, lap: lastMap?.st?.lap, date: new Date().toISOString().slice(0, 10) });
  $("card").hidden = false;
}
async function cardBlob() { return new Promise((res) => $("card-canvas").toBlob(res, "image/png")); }
$("card-btn").addEventListener("click", openCard);
$("card-close").addEventListener("click", () => { $("card").hidden = true; });
$("card").addEventListener("click", (e) => { if (e.target === $("card")) $("card").hidden = true; });
$("card-save").addEventListener("click", async () => {
  const a = document.createElement("a"); a.href = URL.createObjectURL(await cardBlob()); a.download = `swarm-${roomCode || "room"}.png`; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
});
$("card-share").addEventListener("click", async () => {
  const file = new File([await cardBlob()], `swarm-${roomCode || "room"}.png`, { type: "image/png" });
  if (navigator.canShare?.({ files: [file] })) navigator.share({ files: [file], title: "Our SwarmLLM room" }).catch(() => {});
  else toast("this browser can't share images: use save");
});
let lastSoloTps = 0;
function exportChat() {
  const lines = [`# SwarmLLM room ${roomCode || ""}`, "", `_${new Date().toISOString().slice(0, 16).replace("T", " ")} · ${MODELS[ai.model || $("ai-model").value]?.label || ""}_`, ""];
  for (const m of document.querySelectorAll("#ai-output .m")) {
    if (m.classList.contains("user")) lines.push(`**${m.dataset.name || "?"}:** ${m.dataset.text || ""}`, "");
    else if (m.pieces) {
      lines.push(m.pieces.map((p) => p.t).join(""), "");
      const st = m.querySelector(".stats")?.textContent;
      if (st) lines.push(`<sub>${st}</sub>`, "");
    }
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/markdown" }));
  a.download = `swarm-chat-${roomCode || "room"}.md`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

// Send turns into Stop while an answer streams: always on the host, on the asker's screen for
// guests (the host honours ai-stop from the asker only).
function setBusyUI(busy, canStop) {
  const b = $("ai-send");
  b.dataset.canStop = busy && canStop ? "1" : "";
  b.disabled = false;
  $("ai-row").classList.toggle("busy", !!busy);
  sendLabel();
}
// while busy, the button stops the answer when the box is empty and queues the text otherwise
function sendLabel() {
  const b = $("ai-send"), busy = $("ai-row").classList.contains("busy"), typed = !!$("ai-prompt").value.trim();
  const stop = busy && b.dataset.canStop === "1" && !typed;
  b.classList.toggle("stop", stop);
  b.textContent = stop ? "Stop" : busy ? "Queue" : "Send";
}
function setCtx(used, max) {
  const el = $("ctx-meter"); if (!el) return;
  if (!used) { el.textContent = ""; return; }
  el.textContent = `context ${used} / ${max}`;
  el.classList.toggle("warn", used > max * 0.8);
}

async function aiLoadShard(modelKey, range, hasEmbed, hasHead) {
  const M = MODELS[modelKey];
  aiLoading(true, `downloading layers ${range[0]}\u2013${range[1] - 1} of ${M.label.split("\u00b7")[0].trim()}`);
  aiStatus("requesting GPU\u2026");
  mascot("Grabbing my slice of the model… hang tight.");
  // a previous attempt in this tab still owns its weights: release them first, or the
  // second load doubles GPU memory and every buffer after the limit comes back invalid
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
  if (ai.role === "host") {
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
  pacerHook = pacer;

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
  aiLoading(false);
}

// ---- host ----
function biggestPeerId() {
  const gb = (m) => (m?.webgpu ? m?.contribGB ?? 0 : 0);
  let best = peer.id, bestGB = gb(myMeta);
  for (const [id, e] of conns) if (gb(e.meta) > bestGB || (gb(e.meta) === bestGB && id < best)) { best = id; bestGB = gb(e.meta); }
  return best;
}
function aiStartAnywhere() {
  const model = $("ai-model").value;
  const boss = biggestPeerId();
  if (boss === peer.id) { aiStart(model); return; }
  $("ai-start").disabled = true; $("ai-model").disabled = true;
  aiLoading(true, `starting ${MODELS[model].label.split("·")[0].trim()}`);
  $("ldg-sub").textContent = `${conns.get(boss)?.name || "the biggest device"} is dealing the layers`;
  $("ldg-fill").style.width = "0%";
  aiStatus(`asked ${conns.get(boss)?.name || "the biggest device"} to start ${MODELS[model].label.split("·")[0].trim()}…`);
  broadcastAll({ t: "ai-start-req", model, boss, by: myName });
}
async function aiStart(modelArg) {
  if (ai.engine || ai.busy) return;
  ai.busy = true;
  if (typeof modelArg === "string") $("ai-model").value = modelArg;
  $("ai-start").disabled = true;
  $("ai-model").disabled = true;
  try {
    ai.role = "host";
    ai.degraded = false;
    ai.readyPeers = new Set();
    ai.teleBy = new Map();
    const modelKey = $("ai-model").value;
    const M = MODELS[modelKey];
    // devices without WebGPU join as ask-only guests: they get the chat, not layers
    ai.chain = [...conns.keys()].filter((id) => conns.get(id)?.meta?.webgpu).sort();
    ai.plan = new Map();                      // name -> load message, so a reloaded device can be re-seated
    ai.chainNames = ai.chain.map((id) => conns.get(id)?.name || id);
    const n = ai.chain.length + 1;
    let L, layerBytes, embedBytes, cfg = null;
    if (M.kind === "qwen35") {
      aiStatus("reading model index… (11 MB)");
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
      aiStatus("reading model index…");
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
    const pledgeOf = (m) => ((m?.contribGB ?? (m?.maxBufGB ? m.maxBufGB * 0.5 : 0.5))) * 2 ** 30;
    const caps = [Math.max(pledgeOf(myMeta) - embedBytes, layerBytes / 2),
      ...ai.chain.map((id) => Math.max(pledgeOf(conns.get(id)?.meta), layerBytes / 2))];
    const { assigned, ranges } = planSplit(L, caps);

    const needGB = (L * layerBytes + embedBytes) / 2 ** 30;
    const haveGB = caps.reduce((s, c) => s + c, embedBytes) / 2 ** 30;
    if (needGB > haveGB * 1.15)
      log("swarm", `⚠ this model needs ~${needGB.toFixed(1)} GB but the room pledged ~${haveGB.toFixed(1)} GB — it may not fit`);

    ai.deferred = [];
    ai.chain.forEach((id, i) => {
      const msg = {
        t: "ai-load", model: modelKey, range: ranges[i + 1],
        next: i + 1 < ai.chain.length ? ai.chain[i + 1] : "host",
        host: peer.id,
      };
      ai.plan.set(conns.get(id)?.name || id, { msg, small: false });
      sendTo(id, msg);
    });
    ai.layersByName = Object.fromEntries([[myName, `${ranges[0][0]}–${ranges[0][1] - 1}`], ...ai.chain.map((id, i) => [conns.get(id)?.name || id, `${ranges[i + 1][0]}–${ranges[i + 1][1] - 1}`])]);
    broadcastAll({ t: "ai-layers", by: ai.layersByName });
    const splitDesc = [`you ${assigned[0]}+embed`, ...ai.chain.map((id, i) =>
      `${conns.get(id)?.name || id} ${assigned[i + 1]}`)].join(" · ");
    log("swarm", `${M.label} — layer split by pledge: ${splitDesc}`);
    await aiLoadShard(modelKey, ranges[0], true, true);
    aiStatus(n === 1
      ? `solo: all ${L} layers local — ready`
      : `layers ${ranges[0][0]}–${ranges[0][1] - 1} ready · syncing with ${ai.chain.length} device${ai.chain.length > 1 ? "s" : ""}…`);
    ai.fed = [];                              // fresh engines everywhere: nothing cached yet
    ai.pendingCtl = {};
    aiMaybeReady();
  } catch (err) {
    clearInterval(ai.progTimer);
    aiLoading(false);
    ai.engine = null;
    $("ai-panel").classList.remove("online");
    aiStatus("failed: " + err.message);
    ai.busy = false;
    $("ai-start").disabled = false;
    $("ai-model").disabled = false;
  }
}

// Deal the layers again over whoever is in the room now: after a device left (the room is
// degraded) or to bring in devices that joined after the start. Cached ranges reload in seconds;
// the conversation is kept and re-prefilled on the next question.
async function aiRedeal() {
  if (ai.role !== "host" || ai.busy === "gen") return;
  const model = ai.model || $("ai-model").value;
  failWaiters(new Error("re-dealing the layers"));
  ai.engine = null; ai.busy = false; ai.fed = null;
  $("ai-panel").classList.remove("online");
  $("ai-row").style.display = "none";
  showRedeal(false);
  broadcastAll({ t: "ai-redeal", by: myName, model });
  aiLoading(true, "re-dealing the layers");
  await aiStart(model);
}
function showRedeal(on, why) {
  const b = $("ai-redeal");
  b.hidden = !on || ai.role !== "host";
  if (why) $("redeal-why").textContent = why;
  $("redeal-why").hidden = b.hidden;
}
// devices with a GPU that are in the room but hold no layers (joined after the start)
function sparePeers() { return [...conns.keys()].filter((id) => conns.get(id)?.meta?.webgpu && !ai.chain.includes(id)); }
function offerRedealForNewcomers() {
  if (ai.role !== "host" || !ai.engine || ai.degraded) return;
  const spare = sparePeers();
  if (spare.length) showRedeal(true, `${spare.map((id) => conns.get(id)?.name || id).join(", ")} joined after the start; re-deal to give ${spare.length > 1 ? "them" : "it"} layers`);
}

// a device in the chain left: every lap in flight fails now instead of timing out, and the room
// waits for a re-deal
function aiPeerLeft(id, name) {
  if (ai.role !== "host") return;
  if (!ai.chain.includes(id)) { offerRedealForNewcomers(); if (!sparePeers().length && !ai.degraded) showRedeal(false); return; }
  const layers = ai.layersByName?.[name];
  const why = `${name || "a device"} left${layers ? ` (layers ${layers})` : ""}`;
  ai.degraded = true;
  ai.readyPeers.delete(id);
  ai.fed = null;
  failWaiters(new Error(why));
  $("ai-row").style.display = ai.engine ? "flex" : "none";
  aiStatus(`${why} — re-deal the layers to keep going`);
  showRedeal(true, `${why}. Re-deal to split the model over the devices still here; cached layers reload in seconds.`);
  broadcastAll({ t: "ai-degraded", why });
}

// a newcomer while the room is online gets the chat as a guest, and the conversation so far
function aiWelcome(id) {
  if (ai.role !== "host" || !ai.engine || ai.readyPeers.size < ai.chain.length || ai.chain.includes(id)) return;
  sendTo(id, { t: "ai-ready-all", model: ai.model });
  if (ai.visibility === "all" && ai.transcript.length) sendTo(id, { t: "ai-history", items: ai.transcript.slice(-20) });
  offerRedealForNewcomers();
}

// a device whose tab got reloaded comes back with a new peer id: put it back in its slot
function aiRejoin(newId, name) {
  if (ai.role !== "host" || !ai.plan?.has(name)) return;
  const i = ai.chainNames.indexOf(name);
  if (i < 0 || ai.chain[i] === newId || ai.chain.includes(newId)) return;
  const oldId = ai.chain[i];
  ai.chain[i] = newId;
  ai.readyPeers.delete(oldId);
  const { msg } = ai.plan.get(name);
  const fresh = { ...msg, next: i + 1 < ai.chain.length ? ai.chain[i + 1] : "host", host: peer.id };
  if (i > 0) sendTo(ai.chain[i - 1], { t: "ai-next", next: newId });
  sendTo(newId, fresh);
  ai.fed = null;                            // its fresh engine holds nothing: re-prefill next time
  log("swarm", `${name} came back — reloading its layers`);
  aiStatus(`${name} reconnected, reloading its layers…`);
  $("ai-row").style.display = ai.readyPeers.size >= ai.chain.length ? "flex" : "none";
}
function aiMaybeReady() {
  if (ai.role !== "host" || !ai.engine) return;
  if (ai.readyPeers.size < ai.chain.length) return;
  const n = ai.chain.length + 1;
  ai.degraded = false;
  ai.busy = false;
  showRedeal(false);
  aiStatus(`cluster online — ${n} device${n > 1 ? "s" : ""}, ${ai.cfg.num_hidden_layers} layers split ${n} ways`);
  clearInterval(ai.progTimer);
  $("ai-panel").classList.add("online");
  $("ai-row").style.display = "flex";
  $("chat-tools").hidden = false;
  $("new-chat").hidden = false;
  $("ai-empty").textContent = "cluster online. ask anything.";
  $("ai-prompt").focus();
  broadcastAll({ t: "ai-ready-all", model: ai.model });
  pushMap(0, null, false, true);
  offerRedealForNewcomers();
  mascot("Cluster online! Ask anything. Everyone in the room can.");
}

// ---- laps ----
// A lap is one frame's trip round the chain. Its waiter resolves with the returned hidden
// state(s), or rejects on timeout or as soon as a device in the chain leaves.
function lapWait(key, ms, what) {
  return new Promise((res, rej) => {
    const timer = setTimeout(() => { ai.waiters.delete(key); rej(new Error(`pipeline timeout (${what})`)); }, ms);
    ai.waiters.set(key, {
      res: (h) => { clearTimeout(timer); res(h); },
      rej: (e) => { clearTimeout(timer); rej(e); },
    });
  });
}
function failWaiters(err) { for (const [k, w] of ai.waiters) { ai.waiters.delete(k); w.rej(err); } }
function lapDone(key, h) { const w = ai.waiters.get(key); if (w) { ai.waiters.delete(key); w.res(h); } }
// send a frame to the first device of the chain; a pending reset or rollback rides with it,
// so it reaches every device strictly before the frame it applies to
function sendChain(msg) {
  ai.frames = (ai.frames || 0) + 1;
  const ctl = ai.pendingCtl; ai.pendingCtl = {};
  sendHidden(ai.chain[0], { ...msg, ...ctl });
}
// forget the conversation state on every device: here now, on the chain with the next frame
function resetState() {
  try { ai.engine.reset?.(); } catch {}
  ai.pos = 0;
  ai.fed = [];
  ai.pendingCtl = ai.chain.length ? { reset: 1 } : {};
}

// Speculative drafting reads the draft block's own KV cache, which prefill fills with the trunk's
// final hidden state at every prompt position. Solo prefill does it inside the engine; in a room
// the returned hidden states come back from the chain, so the host fills it as they arrive
// (roadmap 25: +18–45% tokens per lap after a prompt). Drafts only change speed, never output.
// ?fill=0 turns it off for A/B runs.
const FILL_DRAFTS = new URLSearchParams(location.search).get("fill") !== "0";
function fillDrafts(h, ids, i0, basePos, n) {
  if (!FILL_DRAFTS || !ai.engine?.mtp) return;
  const dim = ai.engine.dims.dim;
  for (let c = 0; c < n; c++) {
    const next = ids[i0 + c + 1];
    if (next === undefined) break;
    ai.engine.setHidden(h.subarray(c * dim, (c + 1) * dim));
    ai.engine.mtpRun(null, next, basePos + c + 1, false);   // no readback: queued, returns at once
  }
}

// run one token through the whole pipeline, returns logits (or null for a prompt token).
// fillNext: the prompt token after this one, to fill the draft cache with this position's hidden.
async function aiPipeToken(id, needLogits = true, fillNext) {
  const pos = ai.pos;
  if (!ai.chain.length && !needLogits) {
    // solo prefill: layers only, no head, no readback; sync every 8 tokens
    ai.engine.pos = pos;
    await ai.engine.prefillToken(id);
    if (pos % 8 === 7) await ai.device.queue.onSubmittedWorkDone();
    ai.pos++; ai.fed?.push(id);
    return null;
  }
  const tHost = performance.now();
  let h = await ai.engine.embedRun(id, pos);
  if (badF32(h)) throw new Error(`NaN after HOST layers (pos ${pos}) — host GPU kernel issue`);
  if (ai.chain.length) {
    const hostMs = performance.now() - tHost;
    const returned = lapWait(pos, 30000, "token");
    sendChain({ t: "ai-hidden", pos, ...packWire(h) });
    h = await returned;
    if (badF32(h)) throw new Error(`NaN in hidden returned by peers (pos ${pos}) — check peer status lines`);
    noteLap(performance.now() - tHost, hostMs);
    ai.lastHidden = h;
    if (!needLogits && fillNext !== undefined) fillDrafts(h, [id, fillNext], 0, pos, 1);
  } // solo mode: engine holds every layer, embedRun already produced the final hidden
  ai.pos++; ai.fed?.push(id);
  if (!needLogits) return null;   // prefill: skip the head entirely
  const logits = await ai.engine.headFromHidden(h);
  if (badF32(logits)) throw new Error(`NaN in logits (pos ${ai.pos}) — head/lm_head kernel issue on host`);
  return logits;
}

// Prefill `ids` (the part of the conversation the caches do not hold yet) from ai.pos; returns
// the logits after the last one.
const PREFILL_WINDOW = 6;
const TAIL_FRAME = new URLSearchParams(location.search).get("tail") !== "0";   // ?tail=0: old per-token tail, for A/B   // prefill rounds in flight round the chain at once
async function aiPrefill(ids) {
  if (!ai.chain.length && ai.engine.prefillTokens && ids.length > 1) {
    // solo: batched prefill, several prompt tokens per GPU pass
    ai.engine.pos = ai.pos;
    await ai.engine.prefillTokens(ids.slice(0, -1));
    ai.pos = ai.engine.pos;
    ai.fed.push(...ids.slice(0, -1));
    // prefillTokens drafts every row but the last prompt token's (it never sees that token);
    // this.x holds the hidden state just before it, which is what that row needs
    if (FILL_DRAFTS && ai.engine.mtp) ai.engine.mtpRun(null, ids[ids.length - 1], ai.pos, false);
    return aiPipeToken(ids[ids.length - 1]);
  }
  let i = 0;
  // the hybrid engine takes any column count per frame (speculative verifies already send 2..8),
  // so the prompt's tail, last token included, goes round the chain as ONE frame instead of one
  // serial lap per token; short follow-ups become a single lap
  const flex = TAIL_FRAME && !!(ai.chain.length && ai.engine.specStep && ai.engine.embedRunBatch);
  let tailLogits = null;
  if (ai.engine.embedRunBatch && (ids.length > 5 || flex)) {
    // split: up to 16 prompt tokens per round, and several rounds in flight at once. Every device
    // runs frames in send order, so round r+1 can enter the host's layers while round r is on a
    // worker: the chain works like a pipeline instead of one device at a time.
    const hdim = ai.engine.dims.dim;
    const NC = ai.engine.NC || 4;   // columns per GPU pass
    // step down 16 -> 8 -> 4 on the tail: without this a remainder of up to NC-1 tokens costs one
    // network lap each
    const widths = [NC, ...[8, 4].filter((w) => w < NC)];
    const inflight = [];
    try {
      outer: for (const W of widths) while (ids.length - 1 - i >= W) {
        if (ai.abort) break outer;
        const nChunks = Math.max(1, Math.min(Math.floor(16 / W), Math.floor((ids.length - 1 - i) / W)));
        const n = nChunks * W, basePos = ai.pos, i0 = i;
        const hb = new Float32Array(n * hdim);
        for (let c = 0; c < nChunks; c++)
          hb.set(await ai.engine.embedRunBatch(ids.slice(i + c * W, i + (c + 1) * W), basePos + c * W), c * W * hdim);
        if (badF32(hb)) throw new Error(`NaN in batched prefill (pos ${basePos})`);
        if (ai.chain.length) {
          while (inflight.length >= PREFILL_WINDOW) await inflight.shift();
          const p = lapWait("b" + basePos, 90000, "batch prefill").then((h) => fillDrafts(h, ids, i0, basePos, n));
          p.catch(() => {});
          inflight.push(p);
          sendChain({ t: "ai-hidden-b", basePos, n, ...packWire(hb) });
        }
        ai.pos = basePos + n;
        ai.fed.push(...ids.slice(i0, i0 + n));
        i += n;
        aiStatus(`prefill: ${i}/${ids.length} tokens…`);
      }
      if (flex && !ai.abort && i < ids.length) {
        const n = ids.length - i, basePos = ai.pos, i0 = i;   // n <= 4: what the widths above left
        const hb = await ai.engine.embedRunBatch(ids.slice(i), basePos);
        if (badF32(hb)) throw new Error(`NaN in batched prefill (pos ${basePos})`);
        const p = lapWait("b" + basePos, 90000, "prefill tail");
        p.catch(() => {});
        sendChain({ t: "ai-hidden-b", basePos, n, ...packWire(hb) });
        ai.pos = basePos + n;
        ai.fed.push(...ids.slice(i0));
        i = ids.length;
        for (const q of inflight) await q;
        const h = await p;
        if (badF32(h)) throw new Error(`NaN in hidden returned by peers (pos ${basePos})`);
        fillDrafts(h, ids, i0, basePos, n);
        const dim = ai.engine.dims.dim;
        ai.lastHidden = h.slice((n - 1) * dim, n * dim);
        tailLogits = await ai.engine.headFromHidden(ai.lastHidden);
        if (badF32(tailLogits)) throw new Error(`NaN in logits (pos ${ai.pos}) — head/lm_head kernel issue on host`);
      }
      for (const p of inflight) await p;
    } catch (err) { failWaiters(err); throw err; }
  }
  if (ai.abort) return null;
  if (tailLogits) return tailLogits;
  let logits = null;
  for (; i < ids.length; i++) {
    if (ai.abort) return null;
    logits = await aiPipeToken(ids[i], i === ids.length - 1, ids[i + 1]);
  }
  return logits;
}

// ---- telemetry and the swarm map ----
// Workers report their compute per frame kind (ai-tele); the host times each lap, so what is left
// is the wire. The map shows the chain, what each device holds and how long its part takes.
function noteLap(lapMs, hostMs) {
  const L = ai.lapStat ||= { lap: 0, host: 0, n: 0 };
  L.lap = L.n ? 0.7 * L.lap + 0.3 * lapMs : lapMs;
  L.host = L.n ? 0.7 * L.host + 0.3 * hostMs : hostMs;
  L.n++;
}
function mapNodes(kind = "spec") {
  const nodes = [{ name: myName, layers: ai.layersByName?.[myName] || "", ms: ai.lapStat?.host, host: 1 }];
  for (const id of ai.chain) {
    const t = ai.teleBy.get(id) || {};
    const name = conns.get(id)?.name || id;
    nodes.push({ name, layers: ai.layersByName?.[name] || "", ms: t[kind] ?? t.spec ?? t.one });
  }
  return nodes;
}
function mapStats(tps, acc) {
  const lap = ai.lapStat?.lap;
  if (!ai.chain.length || !lap) return { tps, acc };
  const gpu = mapNodes().reduce((s, x) => s + (x.ms || 0), 0);
  return { tps, acc, lap: Math.round(lap), gpu: Math.round(gpu), net: Math.max(0, Math.round(lap - gpu)) };
}
let lastMap = null, bestTps = 0;
function renderMap(nodes, st, live) {
  const el = $("swarm-map"); if (!el || !nodes?.length) return;
  lastMap = { nodes, st: { ...(lastMap?.st || {}), ...(st || {}) } };
  if (st?.tps && !live) bestTps = Math.max(bestTps, st.tps);
  el.hidden = false;
  el.classList.toggle("live", !!live);
  const lap = Math.max(120, Math.min(4000, st?.lap || 600));
  el.style.setProperty("--lap", lap + "ms");
  el.style.setProperty("--n", nodes.length);
  el.querySelector(".sm-track").innerHTML = nodes.map((x, i) => `<div class="sm-node${x.host ? " host" : ""}" style="--i:${i}">
      <div class="sm-dot"></div><div class="sm-name">${esc(String(x.name))}</div>
      <div class="sm-sub">${x.host ? "embed · " : ""}${x.layers ? "L" + esc(String(x.layers)) : ""}${x.host ? " · head" : ""}</div>
      <div class="sm-ms">${x.ms ? Math.round(x.ms) + " ms" : ""}</div></div>`).join('<div class="sm-link"><i></i></div>')
    + (nodes.length > 1 ? '<div class="sm-link back"><i></i></div>' : "");
  const bits = [];
  if (st?.tps) bits.push(`${st.tps.toFixed(1)} tok/s`);
  if (st?.lap) bits.push(`lap ${st.lap} ms = GPUs ${st.gpu} + wire ${st.net}`);
  if (st?.acc != null) bits.push(`${Math.round(st.acc * 100)}% of drafts accepted`);
  el.querySelector(".sm-meta").textContent = bits.join(" · ") || `${nodes.length} device${nodes.length > 1 ? "s" : ""} · every token takes a lap through all of them`;
}
let mapAt = 0;
function pushMap(tps, acc, live, force) {
  const now = performance.now();
  if (!force && now - mapAt < 800) return;
  mapAt = now;
  const nodes = mapNodes(), st = mapStats(tps, acc);
  renderMap(nodes, st, live);
  broadcastAll({ t: "ai-map", nodes, st, live: live ? 1 : 0 });
}
// worker: EMA of compute ms per frame kind, reported to the host at most every 700 ms
function teleNote(kind, ms) {
  const T = ai.tele ||= { at: 0, k: {} };
  const k = T.k[kind] ||= { ema: ms, n: 0 };
  k.ema = k.n ? 0.7 * k.ema + 0.3 * ms : ms; k.n++;
  const now = performance.now();
  if (now - T.at > 700 && ai.hostId) {
    T.at = now;
    sendTo(ai.hostId, { t: "ai-tele", k: Object.fromEntries(Object.entries(T.k).map(([a, b]) => [a, Math.round(b.ema * 10) / 10])) });
  }
}

// who sees the chat: the host's dropdown. The full message goes to the screens allowed to see
// the text, the hidden stand-in (same type, `hidden: true`) to the others, so every screen still
// locks and unlocks its Send box with the answer.
function sendChat(msg, askerId) {
  const { full, hidden } = chatRecipients(ai.visibility || "all", askerId, [...conns.keys()]);
  for (const id of full) sendTo(id, msg);
  if (msg.t !== "ai-token") for (const id of hidden) sendTo(id, { t: msg.t, name: msg.name, stats: msg.stats, asker: msg.asker, ctx: msg.ctx, hidden: true });
}

// answer length the host picks; ?maxnew=N overrides it (tests)
const ANSWER_LEN = { short: 150, normal: MAX_NEW, long: 1200 };
const MAXNEW_PARAM = Math.max(0, parseInt(new URLSearchParams(location.search).get("maxnew"), 10) || 0);
// mode: "ask" a new question, or "continue" the last answer (it stopped at the length cap)
async function aiGenerate(textArg, who, askerId = peer.id, mode = "ask") {
  const cont = mode === "continue";
  const lastTurn = ai.conv.turns[ai.conv.turns.length - 1];
  if (cont && lastTurn?.role !== "assistant") return;
  const text = cont ? "(continue)" : (textArg ?? $("ai-prompt").value).trim();
  const asker = who || myName;
  if (!text || ai.busy === "gen" || !ai.engine) return;
  if (ai.degraded) {
    if (askerId === peer.id) toast("a device left: re-deal the layers first");
    else sendTo(askerId, { t: "ai-busy", why: "a device left the room; the host has to re-deal the layers first" });
    return;
  }
  ai.busy = "gen";
  ai.abort = false;
  ai.askerId = askerId;
  ai.lastAsker = askerId;
  setBusyUI(true, true);
  const S = specials(ai.tok);
  const persona = PERSONAS[ai.settings.persona] || PERSONAS.default;
  const thinking = !!ai.settings.thinking && S.think !== undefined;
  const sample = pickSampler(ai.settings.sampling);
  const eos = (t) => t === S.imEnd || t === S.eot;

  setAfterAnswer(false, false);
  if (!cont) chatUser(asker, text);
  const mid = ai.msgSeq = (ai.msgSeq || 0) + 1;
  chatBotStart(mid);
  sendChat({ t: "ai-genstart", name: asker, text, asker: askerId, cont: cont ? 1 : 0, mid }, askerId);
  mascot("Thinking… every word is taking a lap through the room.");

  const answer = [];          // sampled ids of this answer, verbatim, for the next turn's history
  let reply = "", count = 0, capped = false, dropped = 0, failed = null, stats = "";
  const t0Gen = performance.now();
  let tDecode = 0, tPre = 0, prefilled = 0, reused = 0, preFrames = 0;
  try {
    // the conversation with this question, trimmed to fit, and how much the caches already hold
    const fit = fitContext(ai.tok, { system: persona.system, turns: cont ? [...ai.conv.turns.slice(0, -1), { ...lastTurn, open: true }] : [...ai.conv.turns, { role: "user", text, name: asker }], thinking }, MAX_SEQ, MIN_ROOM);
    dropped = fit.dropped;
    ai.conv.turns = fit.turns;
    reused = reusablePrefix(ai.fed, fit.ids);
    if (!reused) resetState();
    const ids = fit.ids.slice(reused);
    // a follow-up's first token needs a draft-cache row too: the trunk hidden at the position
    // before it is still in the engine when the last answer ended on a speculative step
    if (reused && FILL_DRAFTS && ai.engine.mtp && ai.xAt === ai.pos) ai.engine.mtpRun(null, ids[0], ai.pos, false);
    ai.xAt = null;
    prefilled = ids.length;
    const cap = thinking ? MAX_NEW_THINKING : (ANSWER_LEN[ai.settings.length] ?? MAX_NEW);
    const maxNew = Math.min(MAXNEW_PARAM || cap, MAX_SEQ - fit.ids.length);
    aiStatus(reused ? `prefill: ${ids.length} new tokens (${reused} already in the room's caches)…` : `prefill: ${ids.length} tokens…`);
    const t0Pre = performance.now();
    ai.frames = 0;
    let logits = await aiPrefill(ids);
    tPre = performance.now() - t0Pre;
    preFrames = ai.frames;

    const t0 = performance.now();
    const emit = (tok, drafted) => {
      const piece = ai.tok.decode([tok]);
      answer.push(tok);
      reply += piece;
      count++;
      chatBotPiece(piece, drafted);
      sendChat({ t: "ai-token", text: piece, d: drafted ? 1 : 0 }, askerId);
      const tps = count / ((performance.now() - t0) / 1000);
      aiStatus(`generating… ${count} tok · ${tps.toFixed(1)} tok/s`);
    };
    let acc = null;
    if (!logits) { /* stopped during prefill */ }
    else if (ai.engine.mtp && ai.engine.specStep) {
      // speculative decoding: the model's own draft head proposes up to K tokens,
      // one batched trunk pass verifies them (byte-identical to plain decoding)
      const spec = ai.chain.length ? {
        runTrunk: async (tokens, pos) => {
          const tLap = performance.now();
          const n = tokens.length, hdim = ai.engine.dims.dim, NC = ai.engine.NC || 4;
          const hb = new Float32Array(n * hdim);
          for (let c = 0; c < n; c += NC) {
            const m = Math.min(NC, n - c);
            hb.set(await ai.engine.embedRunBatch(tokens.slice(c, c + m), pos + c, { base: c, total: n }), c * hdim);
          }
          if (badF32(hb)) throw new Error(`NaN after HOST layers (pos ${pos})`);
          const hostMs = performance.now() - tLap;
          const returned = lapWait("b" + pos, 90000, "verify");
          sendChain({ t: "ai-hidden-b", basePos: pos, n: tokens.length, spec: 1, ...packWire(hb) });
          const h = await returned;
          if (badF32(h)) throw new Error(`NaN in hidden returned by peers (pos ${pos})`);
          noteLap(performance.now() - tLap, hostMs);
          return h;
        },
        // the rollback rides on the next frame (sendChain), strictly before it on every device
        onReject: async (k) => { ai.pendingCtl = { rb: k }; },
      } : {};
      if (ai.chain.length && ai.lastHidden) ai.engine.setHidden(ai.lastHidden);
      ai.engine.pos = ai.pos;
      // draft depth: pick by MEASURED tokens/sec per depth (K=3 warm-up, probe
      // 5 and 7 once, keep the best, re-probe now and then). Deep chains only
      // pay when the network round-trip dominates the lap; a lap-time
      // threshold can't tell GPU time from RTT and gets stuck deep.
      const kc = { cand: [3, 5, 7], ema: {}, n: {}, step: 0, used: {} };
      const pickK = () => {
        if (!ai.chain.length) return 3;
        kc.step++;
        if (kc.step <= 3) return 3;
        const untried = kc.cand.find((k) => !kc.n[k]);
        if (untried) return untried;
        let best = 3;
        for (const k of kc.cand) if (kc.ema[k] > kc.ema[best]) best = k;
        if (kc.step % 16 === 0) { const alt = kc.cand.filter((k) => k !== best); return alt[(kc.step / 16) % alt.length | 0]; }
        return best;
      };
      const st0 = { ...ai.engine.mtp.stats };
      // the first answer token is sampled here; specStep treats it as already chosen for this
      // position and returns only the tokens after it, so it has to be emitted (or end the
      // answer) before the loop, or the reply starts one word late
      let next = sample(logits), done = false;
      if (eos(next)) done = true; else emit(next, false);
      while (!done && count < maxNew && !ai.abort) {
        // a speculative step touches positions pos .. pos+K (K drafts verified in one pass) and
        // drafts one more; shrink K near the end of the context and stop before it overflows
        let K = pickK();
        const roomLeft = MAX_SEQ - ai.engine.pos - 2;
        if (roomLeft < 1) { capped = true; break; }
        // never draft past the answer cap: every token a step writes into the caches is then an
        // emitted one, so a capped answer is still a prefix of the next turn and nothing re-prefills
        K = Math.min(K, roomLeft, maxNew - count);
        const tStep = performance.now();
        const toks = await ai.engine.specStep(next, sample, K, spec);
        // specStep wrote `next` and the accepted drafts; its last token is the next `next`
        ai.fed.push(next, ...toks.slice(0, -1));
        const tps = toks.length / ((performance.now() - tStep) / 1000);
        kc.ema[K] = kc.n[K] ? 0.6 * kc.ema[K] + 0.4 * tps : tps;
        kc.n[K] = (kc.n[K] || 0) + 1; kc.used[K] = (kc.used[K] || 0) + toks.length;
        for (let j = 0; j < toks.length; j++) {
          const tk = toks[j];
          if (eos(tk)) { done = true; break; }
          if (count >= maxNew) { done = true; capped = true; break; }
          emit(tk, j < toks.length - 1);   // all but the last were drafts the trunk accepted
        }
        next = toks[toks.length - 1];
        const d = ai.engine.mtp.stats.drafts - st0.drafts;
        acc = d ? (ai.engine.mtp.stats.accepted - st0.accepted) / d : null;
        if (ai.chain.length) pushMap(count / ((performance.now() - t0) / 1000), acc, true);
      }
      if (!done && count >= maxNew) capped = true;
      ai.pos = ai.engine.pos;
      ai.xAt = ai.pos;   // specStep left the trunk hidden at ai.pos - 1 in the engine
      const st = ai.engine.mtp.stats;
      if (st.drafts) crumb(`spec: ${st.accepted}/${st.drafts} drafts accepted${ai.lapStat ? ` · lap ${Math.round(ai.lapStat.lap)}ms` : ""}`
        + (ai.chain.length ? ` · K tok/s ${kc.cand.map((k) => `${k}:${kc.ema[k] ? kc.ema[k].toFixed(1) : "-"}`).join(" ")} · tokens by K ${JSON.stringify(kc.used)}` : ""));
    } else {
      // plain decoding. An end token is not piped through the chain: the next turn's template
      // writes <|im_end|> itself, so both paths leave the caches holding exactly prompt + answer
      for (let i = 0; i < maxNew && !ai.abort; i++) {
        const next = sample(logits);
        if (eos(next)) break;
        emit(next, false);
        if (ai.pos >= MAX_SEQ - 1) { capped = true; break; }   // no position left for another token
        logits = await aiPipeToken(next);
        if (ai.chain.length) pushMap(count / ((performance.now() - t0) / 1000), null, true);
      }
      if (count >= maxNew) capped = true;
    }
    tDecode = performance.now() - t0;
    const secs = tDecode / 1000;
    stats = `${count} tok · ${(count / Math.max(secs, 1e-3)).toFixed(1)} tok/s · ${ai.chain.length + 1} device${ai.chain.length ? "s" : ""}`
      + (acc != null ? ` · ${Math.round(acc * 100)}% drafts accepted` : "")
      + (ai.abort ? " · stopped" : "")
      + (capped ? (ai.pos >= MAX_SEQ - 2 ? ` · stopped: context full (${MAX_SEQ} tokens)` : ` · stopped at ${count} tokens`) : "")
      + (dropped ? ` · ${dropped} oldest exchange${dropped > 1 ? "s" : ""} forgotten to fit` : "");
    if (ai.chain.length && count) pushMap(count / Math.max(secs, 1e-3), acc, false, true);
    else if (count > 8) lastSoloTps = Math.max(lastSoloTps, count / Math.max(secs, 1e-3));
  } catch (err) {
    failed = err;
    ai.fed = null;            // the caches are in an unknown state: the next question starts clean
    ai.pendingCtl = {};
    stats = "failed: " + err.message;
    aiStatus("generation failed: " + err.message);
  }
  // the answer (even a partial one) joins the history, so the next turn reads what was said
  const tail = ai.conv.turns[ai.conv.turns.length - 1];
  if (tail?.role === "user") ai.conv.turns.push({ role: "assistant", ids: answer });
  else if (tail?.open) { tail.ids = [...tail.ids, ...answer]; delete tail.open; }
  const ctx = { used: ai.fed ? ai.pos : 0, max: MAX_SEQ };
  if (failed) chatBotEnd(reply ? null : "⚠ " + failed.message, stats);
  else chatBotEnd(null, stats);
  const canContinue = capped && !failed && !ai.abort;
  sendChat({ t: "ai-gendone", stats, ctx, failed: failed ? 1 : 0, capped: canContinue ? 1 : 0 }, askerId);   // unlocks every send box
  setAfterAnswer(canContinue, !failed);
  if (cont && ai.transcript.length) { const t = ai.transcript[ai.transcript.length - 1]; t.reply += reply; t.stats = stats; }
  else ai.transcript.push({ name: asker, text, reply, stats, mid });
  if (ai.transcript.length > 50) ai.transcript.shift();
  setCtx(ctx.used, ctx.max);
  if (!failed) aiStatus(`ready — prefill ${prefilled} tok in ${(tPre / 1000).toFixed(1)}s${ai.chain.length ? ` / ${preFrames} frame${preFrames === 1 ? "" : "s"}` : ""}${reused ? ` (${reused} reused)` : ""}, ${stats}`);
  mascot("Done. Anyone in the room can ask the next one.");
  ai.busy = false;
  ai.abort = false;
  setBusyUI(false);
  setTimeout(nextQueued, 0);
  if (ai.degraded) showRedeal(true);
  void t0Gen;
}

// Continue / Regenerate the last answer: the host, or whoever asked it. Regenerate drops the last
// exchange from the conversation and asks it again; the caches no longer match, so it
// re-prefills (with "exact" sampling it gives the same answer, which is the point of exact).
function setAfterAnswer(canContinue, ok) {
  $("continue-btn").hidden = !canContinue;
  $("regen-btn").hidden = !ok;
}
function aiCommand(cmd, from) {
  if (ai.role !== "host" || ai.busy === "gen" || !ai.engine) return;
  const byAsker = from === ai.lastAsker || from === peer.id;
  if (!byAsker) { sendTo(from, { t: "ai-busy", why: "only the host or whoever asked can do that" }); return; }
  const turns = ai.conv.turns;
  if (cmd === "continue") { aiGenerate(null, from === peer.id ? myName : conns.get(from)?.name, from, "continue"); return; }
  if (cmd === "regen" && turns.length >= 2 && turns[turns.length - 1].role === "assistant") {
    const q = turns[turns.length - 2];
    ai.conv.turns = turns.slice(0, -2);
    ai.transcript.pop();
    broadcastAll({ t: "ai-regen" });
    markReplaced();
    aiGenerate(q.text, q.name || (from === peer.id ? myName : conns.get(from)?.name), from);
  }
}
function markReplaced() {
  const ms = [...document.querySelectorAll("#ai-output .m")];
  for (const m of ms.slice(-2)) m.classList.add("replaced");
}

// Start a new conversation: the next question prefills from scratch on every device.
function aiNewChat() {
  if (ai.role !== "host" || ai.busy === "gen") return;
  ai.conv = { turns: [] };
  ai.fed = null;
  ai.transcript = [];
  clearChat();
  setAfterAnswer(false, false);
  broadcastAll({ t: "ai-reset", by: myName });
  setCtx(0);
  toast("new chat: the swarm forgot the conversation");
}
function clearChat() {
  $("ai-output").innerHTML = "";
  botEl = null;
  setCtx(0);
}

// ---- worker ----
// Frames run strictly one after another in arrival order (the transport delivers them in send
// order), so several prefill rounds can be queued here while the GPU works. Control that rides
// on a frame (reset, rollback) applies before it, and goes on down the chain with it.
async function workerFrame(d) {
  if (!ai.engine) return;
  const ctl = {};
  if (d.reset) { ai.engine.reset?.(); ctl.reset = 1; }
  else if (d.rb != null) { ai.engine.restoreDN?.(d.rb); ctl.rb = d.rb; }
  const t0 = performance.now();
  if (d.t === "ai-hidden-b") {
    // n hiddens in, my layers (batched), n hiddens on
    const xs = unpackWire(d);
    const nTok = d.n || 4;
    const wdim = ai.engine.dims.dim;
    const hb = new Float32Array(nTok * wdim);
    const NC = ai.engine.NC || 4;
    for (let c = 0; c < nTok; c += NC) {
      const m = Math.min(NC, nTok - c);
      hb.set(await ai.engine.runHiddenBatch(xs.subarray(c * wdim, (c + m) * wdim), d.basePos + c, d.spec ? { base: c, total: nTok } : false), c * wdim);
    }
    if (badF32(hb)) { aiStatus(`⚠ NaN in batched prefill on this device`); sendTo(ai.hostId, { t: "ai-error", message: "NaN in batched prefill" }); }
    teleNote(d.spec ? "spec" : "pre", performance.now() - t0);
    // the verify flag travels with the frame: every device snapshots its recurrent state per
    // column, or a later rollback on it restores a stale snapshot
    const bmsg = { basePos: d.basePos, n: nTok, ...(d.spec ? { spec: 1 } : {}), ...packWire(hb) };
    if (ai.next === "host") sendHidden(ai.hostId, { t: "ai-hiddenret-b", ...bmsg });
    else sendHidden(ai.next, { t: "ai-hidden-b", ...bmsg, ...ctl });
  } else {
    // one token: run my layers, forward along the chain
    const hin = unpackWire(d);
    if (badF32(hin)) { aiStatus(`⚠ NaN ARRIVED at this device (pos ${d.pos}) — upstream peer broken`); }
    const h = await ai.engine.runHidden(hin, d.pos);
    if (badF32(h)) { aiStatus(`⚠ NaN PRODUCED by this device (pos ${d.pos}, layers ${ai.range[0]}–${ai.range[1] - 1}) — GPU kernel issue here`); sendTo(ai.hostId, { t: "ai-error", message: `NaN produced on worker layers ${ai.range[0]}–${ai.range[1] - 1}` }); }
    teleNote("one", performance.now() - t0);
    const msg = { pos: d.pos, ...packWire(h) };
    if (ai.next === "host") sendHidden(ai.hostId, { t: "ai-hiddenret", ...msg });
    else sendHidden(ai.next, { t: "ai-hidden", ...msg, ...ctl });
    if (d.pos % 8 === 0) aiStatus(`serving layers ${ai.range[0]}–${ai.range[1] - 1} — pos ${d.pos}`);
  }
}

// the host's tab closed: the room is over for everyone else
function hostGone() {
  if (ai.role === "host") return;
  failWaiters(new Error("the host left"));
  ai.engine = null;
  $("ai-row").style.display = "none";
  $("room-over").hidden = false;
  $("room-over-why").textContent = "The host's tab closed, and the host holds the conversation and the model's first and last layers, so this room can't answer any more.";
  aiStatus("the host left; this room is over");
  mascot("The host left. Start a new room?");
}

// ---- messages: worker, guest and host ----
async function aiOnData(from, d) {
  const e = conns.get(from);
  switch (d.t) {
    case "ai-start-req":
      if (MODELS[d.model]) $("ai-model").value = d.model;   // every screen shows the model that was actually started
      $("ai-start").disabled = true; $("ai-model").disabled = true;
      if (d.boss !== peer.id) { aiLoading(true, `starting ${MODELS[d.model]?.label.split("·")[0].trim()}`); $("ldg-sub").textContent = `${d.by} pressed start`; $("ldg-fill").style.width = "0%"; }
      if (d.boss === peer.id) { toast(`${d.by} started ${MODELS[d.model]?.label.split("·")[0].trim()}`); aiStart(d.model); }
      else aiStatus(`${d.by} started the model…`);
      break;
    case "ai-next": ai.next = d.next; ensureLink(d.next); break;
    case "ai-layers": ai.layersByName = d.by; loadCardRender(); break;
    case "ai-reset":   // the host started a new chat
      clearChat();
      toast(`${d.by || "the host"} started a new chat`);
      break;
    case "ai-redeal":
      $("ai-panel").classList.remove("online");
      $("ai-row").style.display = "none";
      $("room-over").hidden = true;
      if (MODELS[d.model]) $("ai-model").value = d.model;
      aiLoading(true, "re-dealing the layers");
      $("ldg-sub").textContent = `${d.by} is re-dealing the layers over the devices in the room`;
      aiStatus(`${d.by} is re-dealing the layers…`);
      break;
    case "ai-degraded":
      aiStatus(`${d.why} — waiting for the host to re-deal the layers`);
      toast(d.why);
      break;
    case "ai-load": {
      if (MODELS[d.model]) $("ai-model").value = d.model;
      ai.role = "worker";
      ai.next = d.next;
      ai.hostId = d.host;
      ai.q = Promise.resolve();
      ensureLink(d.next);   // open the link to my chain neighbour while the weights download
      try {
        await aiLoadShard(d.model || "smollm-135m", d.range, false, false);
        if (!(await ensureLink(d.next))) throw new Error("could not connect to the next device in the chain");
        aiStatus(`layers ${d.range[0]}–${d.range[1] - 1} ready · syncing with the room…`);
        aiLoading(true, `layers ${d.range[0]}–${d.range[1] - 1} ready`);
        $("ldg-sub").textContent = "syncing with the rest of the room";
        $("ldg-fill").style.width = "100%";
        sendTo(ai.hostId, { t: "ai-ready" });
      } catch (err) {
        aiLoading(false);
        aiStatus("failed: " + err.message);
        sendTo(ai.hostId, { t: "ai-error", message: err.message });
      }
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
      break;
    case "ai-ready":
      ai.readyPeers.add(from);
      if (e?.card) e.card.querySelector(".bw").textContent = "ready";
      aiMaybeReady();
      break;
    case "ai-error":
      aiStatus(`peer ${e?.name || from} failed: ${d.message}`);
      break;
    case "ai-tele": if (ai.role === "host") ai.teleBy.set(from, d.k || {}); break;
    case "ai-map": renderMap(d.nodes, d.st, d.live); break;
    case "ai-hidden-b":
    case "ai-hidden":
      if (ai.role !== "worker") break;
      ai.q = ai.q.then(() => workerFrame(d)).catch((err) => {
        aiStatus("⚠ " + err.message);
        sendTo(ai.hostId, { t: "ai-error", message: err.message });
      });
      break;
    case "ai-hiddenret-b": lapDone("b" + d.basePos, unpackWire(d)); break;
    case "ai-hiddenret": lapDone(d.pos, unpackWire(d)); break;
    case "ai-visibility":
      ai.visibility = d.mode;
      toast(d.mode === "all" ? "the host shows the chat to everyone" : d.mode === "host" ? "the host keeps the chat private" : "the host shows each answer to whoever asked");
      break;
    case "ai-style":
      toast(`answers now: ${PERSONAS[d.persona]?.label || d.persona}${d.thinking ? " · thinking first" : ""}`);
      break;
    case "ai-regen": markReplaced(); break;
    case "ai-react": if (ai.role === "host") hostReact(String(d.mid), d.e, from); break;
    case "ai-reacts": renderReacts(String(d.mid), d.counts); break;
    case "ai-typing":
      if (ai.role === "host") {
        if (ai.visibility !== "all") break;
        const name = conns.get(from)?.name || "someone";
        for (const id of conns.keys()) if (id !== from) sendTo(id, { t: "ai-typing", name });
        showTyping(name);
      } else showTyping(d.name || "someone");
      break;
    case "ai-cmd": aiCommand(d.cmd, from); break;
    case "ai-genstart":
      setAfterAnswer(false, false);
      if (!d.cont)
      chatUser(d.name, d.hidden ? "asked something (the host keeps the chat private)" : d.text);
      chatBotStart(d.hidden ? null : d.mid);
      setBusyUI(true, d.asker === peer.id);
      mascot(`${d.name} asked something. Thinking…`);
      break;
    case "ai-token": chatBotPiece(d.text, d.d); break;
    case "ai-gendone":
      chatBotEnd(d.hidden ? "answer hidden by the host" : null, d.stats);
      setBusyUI(false);
      setAfterAnswer(!!d.capped && !d.hidden, !d.failed && !d.hidden);
      if (d.ctx) setCtx(d.ctx.used, d.ctx.max);
      mascot("Your turn. Ask anything.");
      break;
    case "ai-history":
      for (const it of d.items || []) {
        chatUser(it.name, it.text);
        chatBotStart(it.mid);
        botEl.pieces = [{ t: it.reply || "", d: 0 }];
        chatBotEnd(null, it.stats);
      }
      break;
    case "ai-ready-all":
      aiLoading(false);
      $("ai-panel").classList.add("online");
      if (ai.role !== "host" && ai.role !== "worker") ai.role = "guest";
      if (ai.role !== "host") ai.hostId = from;
      if (MODELS[d.model]) { $("ai-model").value = d.model; ai.model = d.model; }
      $("ai-row").style.display = "flex";
      $("chat-tools").hidden = false;
      $("ai-empty").textContent = "cluster online. ask anything.";
      aiStatus(ai.range ? `cluster online · serving layers ${ai.range[0]}–${ai.range[1] - 1}` : "cluster online · this device asks, the others think");
      mascot("Cluster online! Type a question, the whole room answers.");
      break;
    case "ai-ask":
      if (ai.role !== "host") break;
      aiAsk(String(d.text || "").slice(0, 8000), d.name, from);
      break;
    case "ai-queued":
      toast(d.pos === 1 ? "queued: yours is next" : `queued: ${d.pos - 1} question${d.pos > 2 ? "s" : ""} ahead of yours`);
      break;
    case "ai-queue": showQueue(d.n); break;
    case "ai-stop":
      if (ai.role === "host" && ai.busy === "gen" && from === ai.askerId) { ai.abort = true; aiStatus(`${e?.name || "the asker"} pressed stop…`); }
      break;
    case "ai-busy": toast(d.why || "the swarm is still answering, try again in a moment"); break;
  }
}

$("ai-start").addEventListener("click", aiStartAnywhere);
$("ai-redeal").addEventListener("click", aiRedeal);
$("ai-visibility").addEventListener("change", (e) => {
  ai.visibility = e.target.value;
  broadcastAll({ t: "ai-visibility", mode: ai.visibility });
  toast(ai.visibility === "all" ? "everyone sees the chat" : ai.visibility === "host" ? "only you see the chat" : "each answer goes to whoever asked");
});
// answer style: persona, sampling, thinking. Takes effect on the next question; a new system
// prompt changes the conversation's first tokens, so that question re-prefills from scratch.
for (const [k, v] of Object.entries(PERSONAS)) $("ai-persona").add(new Option(v.label, k));
for (const [k, v] of Object.entries(SAMPLING)) $("ai-sampling").add(new Option(v.label, k));
function styleChanged() {
  ai.settings = { persona: $("ai-persona").value, sampling: $("ai-sampling").value, thinking: $("ai-thinking").checked, length: $("ai-length").value };
  broadcastAll({ t: "ai-style", ...ai.settings });
}
for (const id of ["ai-persona", "ai-sampling", "ai-thinking", "ai-length"]) $(id).addEventListener("change", styleChanged);
$("cache-clear").addEventListener("click", async (ev) => {
  ev.preventDefault();
  try { await caches.delete("swarmllm-weights-v1"); weightCache = null; toast("cached weights cleared"); } catch { toast("could not clear the cache"); }
});
$("new-chat").addEventListener("click", aiNewChat);
$("draft-view").addEventListener("click", () => setDraftView(!draftView));
$("export-chat").addEventListener("click", exportChat);
for (const [id, cmd] of [["continue-btn", "continue"], ["regen-btn", "regen"]])
  $(id).addEventListener("click", () => { setAfterAnswer(false, false); if (ai.role === "host") aiCommand(cmd, peer.id); else if (ai.hostId) sendTo(ai.hostId, { t: "ai-cmd", cmd }); });
// Questions asked while the swarm is answering wait in the host's queue and run in order, one
// generation at a time (every device is busy with every token). At most QUEUE_MAX waiting, two
// per device.
const QUEUE_MAX = 10;
function aiAsk(text, name, from) {
  if (!text) return;
  if (ai.busy !== "gen" && !ai.queue?.length && !ai.degraded) { aiGenerate(text, name, from); return; }
  ai.queue ||= [];
  if (ai.queue.length >= QUEUE_MAX || ai.queue.filter((q) => q.from === from).length >= 2) {
    const why = "the queue is full, try again after this answer";
    if (from === peer.id) toast(why); else sendTo(from, { t: "ai-busy", why });
    return;
  }
  ai.queue.push({ text, name, from });
  const pos = ai.queue.length;
  if (from === peer.id) toast(pos === 1 ? "queued: yours is next" : `queued: ${pos - 1} ahead of yours`);
  else sendTo(from, { t: "ai-queued", pos });
  broadcastAll({ t: "ai-queue", n: ai.queue.length }); showQueue(ai.queue.length);
}
function nextQueued() {
  if (ai.role !== "host" || ai.busy || ai.degraded || !ai.engine || !ai.queue?.length) return;
  const q = ai.queue.shift();
  broadcastAll({ t: "ai-queue", n: ai.queue.length }); showQueue(ai.queue.length);
  aiGenerate(q.text, q.name, q.from);
}
function showQueue(n) { $("queue-note").textContent = n ? `${n} queued` : ""; }
function aiSubmit() {
  const text = $("ai-prompt").value.trim();
  if (!text) return;
  if (ai.role === "host") { $("ai-prompt").value = ""; growPrompt(); aiAsk(text, myName, peer.id); return; }
  const hostId = ai.hostId;
  if (!conns.has(hostId)) { toast("not connected to the host"); return; }
  $("ai-prompt").value = ""; growPrompt();
  sendTo(hostId, { t: "ai-ask", text, name: myName });
}
function aiStop() {
  if (ai.role === "host") { if (ai.busy === "gen") { ai.abort = true; aiStatus("stopping after this lap…"); } }
  else if (ai.hostId) sendTo(ai.hostId, { t: "ai-stop" });
  $("ai-send").disabled = true;
}
// the prompt box grows with its text; Enter sends and Shift+Enter is a new line (phones: the
// keyboard's return key is a new line, the Send button sends)
function growPrompt() {
  if (!$("ai-prompt").value) queueMicrotask(sendLabel); const p = $("ai-prompt"); p.style.height = "auto"; p.style.height = Math.min(p.scrollHeight, 160) + "px"; }
// the button stops while it says Stop; Enter always sends (or queues) the text, never stops
$("ai-send").addEventListener("click", () => { if ($("ai-send").classList.contains("stop")) aiStop(); else aiSubmit(); });
$("ai-prompt").addEventListener("input", () => { growPrompt(); sendLabel(); noteTyping(); });
if (!("speechSynthesis" in window)) $("read-aloud").hidden = true;
$("read-aloud").addEventListener("click", () => {
  readAloud = !readAloud;
  $("read-aloud").classList.toggle("on", readAloud);
  if (!readAloud) try { speechSynthesis.cancel(); } catch {}
  toast(readAloud ? "answers are read aloud by this device's own voice (nothing leaves the room)" : "read aloud off");
});
$("ai-prompt").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing && !myMeta?.phone) { e.preventDefault(); aiSubmit(); }
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && $("ai-send").classList.contains("stop")) aiStop(); });
mascot("Hi! I'm Swarmy. Create a room, or type a friend's code to join one.");
