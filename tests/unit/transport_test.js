// room/transport.js: slicing and reassembly are byte-exact, tolerate reordering and duplicates,
// and keep every send under SLICE_BYTES.
import { makeLink, sendFrame, SLICE_BYTES } from "../../room/transport.js";

function fakeChannels(link, n, sink) {
  for (let i = 0; i < n; i++) link.chans.push({ readyState: "open", send: (buf) => sink.push({ i, buf }) });
}
// receive() is module-private: attach a stub peer connection and drive its onmessage
import { attachWire } from "../../room/transport.js";
function receiver(onFrame) {
  const link = makeLink(); let handler = null;
  const pc = { createDataChannel: () => ({ set onmessage(f) { handler = f; }, set onclose(_) {}, readyState: "open" }) };
  attachWire(link, { peerConnection: pc }, onFrame);
  return (buf) => handler({ data: buf });
}

const dim = 5120;
const shapes = [
  { t: "ai-hidden", pos: 17, n: 1, cols: 1 },
  { t: "ai-hidden-b", basePos: 240, n: 16, cols: 16 },
  { t: "ai-hiddenret-b", basePos: 5, n: 6, spec: 1, cols: 6 },
];
for (const sh of shapes) {
  const data = new Uint16Array(dim * sh.cols); for (let i = 0; i < data.length; i++) data[i] = (i * 2654435761) >>> 16;
  Deno.test(`transport round trip ${sh.t} x${sh.cols}`, () => {
    const link = makeLink(), out = []; fakeChannels(link, 3, out);
    if (!sendFrame(link, { ...sh, data })) throw new Error("send refused");
    for (const { buf } of out) if (buf.byteLength > SLICE_BYTES) throw new Error("slice too big: " + buf.byteLength);
    const expectSlices = Math.ceil(data.byteLength / (SLICE_BYTES - 24));
    if (out.length !== expectSlices) throw new Error(`expected ${expectSlices} slices, got ${out.length}`);
    // stripes: consecutive slices land on different channels
    if (out.length > 1 && out[0].i === out[1].i) throw new Error("slices not striped");
    let got = null; const deliver = receiver((m) => { got = m; });
    // deliver reversed, with a duplicate in the middle
    const order = [...out].reverse(); order.splice(1, 0, out[Math.floor(out.length / 2)]);
    for (const { buf } of order) deliver(buf);
    if (!got) throw new Error("frame not reassembled");
    if (got.t !== sh.t) throw new Error("kind mismatch " + got.t);
    if ((sh.pos ?? sh.basePos) !== (got.pos ?? got.basePos)) throw new Error("pos mismatch");
    if (got.n !== sh.n || !!got.spec !== !!sh.spec) throw new Error("meta mismatch");
    if (got.data.length !== data.length) throw new Error("length mismatch");
    for (let i = 0; i < data.length; i++) if (got.data[i] !== data[i]) throw new Error("byte mismatch at " + i);
  });
}
Deno.test("transport refuses when no channel is open", () => {
  const link = makeLink(); link.chans.push({ readyState: "connecting", send() {} });
  if (sendFrame(link, { t: "ai-hidden", pos: 0, data: new Uint16Array(8) })) throw new Error("should refuse");
});
