// Minimal markdown renderer for the chat transcript (escapes first; no raw HTML).

export function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

export function md(src) {
  let s = esc(src);
  s = s.replace(/```([\s\S]*?)```/g, (_, c) => `<pre>${c.trim()}</pre>`);
  s = s.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>").replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<i>$2</i>");
  const lines = s.split("\n");
  let out = "", list = null, para = [];
  const flushP = () => { if (para.length) { out += `<p>${para.join("<br>")}</p>`; para = []; } };
  const flushL = () => { if (list) { out += `</${list}>`; list = null; } };
  for (const ln of lines) {
    const h = ln.match(/^(#{1,4})\s+(.*)$/);
    const ul = ln.match(/^\s*[-*]\s+(.*)$/);
    const ol = ln.match(/^\s*\d+[.)]\s+(.*)$/);
    if (ln.startsWith("<pre>")) { flushP(); flushL(); out += ln; continue; }
    if (h) { flushP(); flushL(); const lv = Math.min(4, h[1].length + 1); out += `<h${lv}>${h[2]}</h${lv}>`; continue; }
    if (ul) { flushP(); if (list !== "ul") { flushL(); list = "ul"; out += "<ul>"; } out += `<li>${ul[1]}</li>`; continue; }
    if (ol) { flushP(); if (list !== "ol") { flushL(); list = "ol"; out += "<ol>"; } out += `<li>${ol[1]}</li>`; continue; }
    if (!ln.trim()) { flushP(); flushL(); continue; }
    flushL(); para.push(ln);
  }
  flushP(); flushL();
  return out;
}
