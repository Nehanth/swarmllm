// The swarm card: a 1200x630 PNG of what this room just did (model, devices and their layers,
// tok/s, draft acceptance) to download or share. Canvas only, no dependencies.

const C = { bg: "#f5f4ee", dot: "#e1ded2", panel: "#fbfaf5", border: "#e1ded2", text: "#16171c", muted: "#8b877a", accent: "#2b4eff" };
const SANS = '"Space Grotesk", -apple-system, BlinkMacSystemFont, sans-serif';
const MONO = '"JetBrains Mono", ui-monospace, monospace';

function roundRect(g, x, y, w, h, r) { g.beginPath(); g.roundRect ? g.roundRect(x, y, w, h, r) : g.rect(x, y, w, h); }
function fit(g, text, max) { let t = String(text); while (t.length > 1 && g.measureText(t).width > max) t = t.slice(0, -2) + "…"; return t; }

// info: { model, code, nodes: [{ name, layers, host }], tps, acc, lap, date }
export function drawCard(canvas, info) {
  const W = 1200, H = 630;
  canvas.width = W; canvas.height = H;
  const g = canvas.getContext("2d");
  g.fillStyle = C.bg; g.fillRect(0, 0, W, H);
  g.fillStyle = C.dot;
  for (let y = 13; y < H; y += 26) for (let x = 13; x < W; x += 26) { g.beginPath(); g.arc(x, y, 1.2, 0, 7); g.fill(); }
  // header
  g.fillStyle = C.text; g.font = `700 34px ${SANS}`; g.textBaseline = "alphabetic";
  g.fillText("swarm", 64, 92);
  const sw = g.measureText("swarm").width;
  g.fillStyle = C.accent; g.fillText("LLM", 64 + sw, 92);
  g.fillStyle = C.muted; g.font = `500 16px ${MONO}`;
  g.fillText(`ROOM ${info.code || ""} · ${info.date || ""}`.toUpperCase(), 64, 124);
  // the number
  g.fillStyle = C.text; g.font = `700 150px ${SANS}`;
  const tps = info.tps ? info.tps.toFixed(1) : "—";
  g.fillText(tps, 60, 300);
  const tw = g.measureText(tps).width;
  g.fillStyle = C.accent; g.font = `500 34px ${MONO}`; g.fillText("tok/s", 76 + tw, 300);
  g.fillStyle = C.text; g.font = `500 30px ${SANS}`;
  g.fillText(fit(g, `${info.model || "a model"} on ${info.nodes.length} device${info.nodes.length > 1 ? "s" : ""}, in browser tabs`, W - 128), 64, 356);
  g.fillStyle = C.muted; g.font = `400 18px ${MONO}`;
  const bits = [];
  if (info.acc != null) bits.push(`${Math.round(info.acc * 100)}% of drafts accepted`);
  if (info.lap) bits.push(`${info.lap} ms per lap round the room`);
  bits.push("no server did any of the thinking");
  g.fillText(fit(g, bits.join(" · "), W - 128), 64, 394);
  // the chain
  const n = info.nodes.length, gap = 26, y = 440, h = 92;
  const w = Math.min(250, (W - 128 - gap * (n - 1)) / n);
  info.nodes.forEach((d, i) => {
    const x = 64 + i * (w + gap);
    g.fillStyle = C.panel; roundRect(g, x, y, w, h, 16); g.fill();
    g.strokeStyle = d.host ? C.text : C.border; g.lineWidth = 2; roundRect(g, x, y, w, h, 16); g.stroke();
    g.fillStyle = C.accent; g.beginPath(); g.arc(x + 22, y + 32, 6, 0, 7); g.fill();
    g.fillStyle = C.text; g.font = `500 20px ${SANS}`; g.fillText(fit(g, d.name, w - 50), x + 38, y + 39);
    g.fillStyle = C.muted; g.font = `400 14px ${MONO}`;
    g.fillText(fit(g, `${d.host ? "embed · " : ""}${d.layers ? "layers " + d.layers : ""}${d.host ? " · head" : ""}`, w - 36), x + 18, y + 72);
    if (i < n - 1) { g.strokeStyle = C.accent; g.lineWidth = 3; g.beginPath(); g.moveTo(x + w + 4, y + h / 2); g.lineTo(x + w + gap - 4, y + h / 2); g.stroke(); }
  });
  g.fillStyle = C.muted; g.font = `400 16px ${MONO}`;
  g.fillText("no servers · nothing to install · one room code · swarmllm.ai", 64, H - 34);
  return canvas;
}
