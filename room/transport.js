// Hidden-state transport: a dedicated data channel per peer link that sends activation frames
// as small slices, optionally striped over several peer connections.
//
// Why: Chrome's SCTP stack (dcSCTP) releases at most 4 packets per send opportunity and starts
// with a ~12 KB congestion window, so a single 10 KB message pays an extra round trip and a 50 KB
// speculative verify block pays three. Measured on a 100 ms link: 1 KB = 51 ms one-way,
// 5 KB = 153 ms, 20 KB = 254 ms (docs/bench-log.md). Slicing every send under four packets and
// spreading a block across several associations brings a hop back to one one-way trip.
//
// Exact by construction: only the packaging of the bytes changes.

export const WIRE_ID = 77;                 // negotiated channel id, same on both ends
export const SLICE_BYTES = 4600;           // ~4 packets of 1150 B payload
export const HDR = 26;
const MAGIC = 0x5357;                      // "SW"
const KINDS = ["ai-hidden", "ai-hidden-b", "ai-hiddenret", "ai-hiddenret-b"];

// Per-link state: { chans: [RTCDataChannel], rr: number, rx: Map<msgId, {parts, got, n, meta}> }
export function makeLink() { return { chans: [], rr: 0, rx: new Map(), nextId: 1, sent: 0, recv: 0 }; }

// Open the wire channel on a PeerJS DataConnection's RTCPeerConnection. Both sides call this with
// the same id, so no ondatachannel event fires and PeerJS never sees the channel.
export function attachWire(link, conn, onFrame, { ordered = true } = {}) {
  const pc = conn.peerConnection;
  if (!pc) return null;
  const ch = pc.createDataChannel("swarm-wire", { negotiated: true, id: WIRE_ID, ordered, ...(ordered ? {} : { maxRetransmits: 0 }) });
  ch.binaryType = "arraybuffer";
  ch.onmessage = (ev) => receive(link, ev.data, onFrame);
  ch.onclose = () => { link.chans = link.chans.filter((c) => c !== ch); };
  link.chans.push(ch);
  return ch;
}

export function wireReady(link) { return link.chans.some((c) => c.readyState === "open"); }

// msg: { t, pos|basePos, n?, spec?, hops?, data: Uint16Array (f16) }
//
// hops[] (roadmap 25 · A4) rides as a small JSON trailer on the final slice only — a
// handful of peers' {peer, computeMs, encodeMs, bytes} is a few hundred bytes at most,
// nowhere near worth slicing of its own. The f16 payload's byte range is computed the
// same way on both ends (from `total`, `per`, `k`), so the trailer never lands inside it.
export function sendFrame(link, msg) {
  const kind = KINDS.indexOf(msg.t);
  if (kind < 0) throw new Error("not a wire kind: " + msg.t);
  const open = link.chans.filter((c) => c.readyState === "open");
  if (!open.length) return false;
  link.sent++;
  const u16 = msg.data;
  const bytes = new Uint8Array(u16.buffer, u16.byteOffset, u16.byteLength);
  const per = SLICE_BYTES - HDR;
  const nSlices = Math.max(1, Math.ceil(bytes.length / per));
  const id = link.nextId++ >>> 0;
  const pos = msg.t === "ai-hidden" || msg.t === "ai-hiddenret" ? msg.pos : msg.basePos;
  const hopsBytes = msg.hops && msg.hops.length ? new TextEncoder().encode(JSON.stringify(msg.hops)) : null;
  for (let k = 0, off = 0; k < nSlices; k++) {
    const len = Math.min(per, bytes.length - off);
    const trailer = k === nSlices - 1 ? hopsBytes : null;
    const buf = new ArrayBuffer(HDR + len + (trailer ? trailer.length : 0)), dv = new DataView(buf);
    dv.setUint16(0, MAGIC); dv.setUint8(2, kind); dv.setUint8(3, msg.spec ? 1 : 0);
    dv.setUint32(4, id); dv.setUint32(8, pos >>> 0); dv.setUint16(12, msg.n || 1);
    dv.setUint16(14, k); dv.setUint16(16, nSlices); dv.setUint32(20, bytes.length);
    dv.setUint16(24, trailer ? trailer.length : 0);
    const u8 = new Uint8Array(buf);
    u8.set(bytes.subarray(off, off + len), HDR);
    if (trailer) u8.set(trailer, HDR + len);
    off += len;
    // round-robin over associations so a block never waits on one congestion window
    const ch = open[(link.rr++) % open.length];
    ch.send(buf);
  }
  return true;
}

function receive(link, buf, onFrame) {
  if (!(buf instanceof ArrayBuffer) || buf.byteLength < HDR) return;
  const dv = new DataView(buf);
  if (dv.getUint16(0) !== MAGIC) return;
  const kind = dv.getUint8(2), spec = dv.getUint8(3), id = dv.getUint32(4), pos = dv.getUint32(8), n = dv.getUint16(12);
  const k = dv.getUint16(14), nSlices = dv.getUint16(16), total = dv.getUint32(20), hopsLen = dv.getUint16(24);
  let r = link.rx.get(id);
  if (!r) { r = { parts: new Array(nSlices), got: 0, n: nSlices, buf: new Uint8Array(total), hops: null, t: performance.now() }; link.rx.set(id, r); }
  if (r.parts[k]) return;   // duplicate
  r.parts[k] = true; r.got++;
  const per = SLICE_BYTES - HDR;
  const len = Math.min(per, total - k * per);
  r.buf.set(new Uint8Array(buf, HDR, len), k * per);
  if (hopsLen) r.hops = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, HDR + len, hopsLen)));
  if (r.got < r.n) return;
  link.rx.delete(id);
  link.recv++;
  const data = new Uint16Array(r.buf.buffer, 0, total >> 1);
  const t = KINDS[kind];
  const msg = { t, enc: "f16", data, n, spec: spec ? 1 : 0, hops: r.hops || [] };
  if (t === "ai-hidden" || t === "ai-hiddenret") msg.pos = pos; else msg.basePos = pos;
  onFrame(msg);
  // drop half-received frames older than 30 s so a lost slice cannot leak memory
  if (link.rx.size > 64) for (const [i, v] of link.rx) if (performance.now() - v.t > 30000) link.rx.delete(i);
}
