// Tiny static server with HTTP Range support (python's http.server has none), for loading GGUF shards in a tab.
import http from "node:http"; import fs from "node:fs"; import path from "node:path";
const root = path.resolve(process.argv[2] || "."), port = +(process.argv[3] || 8791);
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json" };
http.createServer((req, res) => {
  const p = path.join(root, decodeURIComponent(new URL(req.url, "http://x").pathname)); if (!p.startsWith(root)) { res.writeHead(403).end(); return; }
  fs.stat(p, (err, st) => { if (err || !st.isFile()) { res.writeHead(404).end(); return; }
    const h = { "Content-Type": types[path.extname(p)] || "application/octet-stream", "Accept-Ranges": "bytes", "Cache-Control": "no-store" };
    const m = /bytes=(\d+)-(\d*)/.exec(req.headers.range || "");
    if (m) { const a = +m[1], b = m[2] ? +m[2] : st.size - 1; res.writeHead(206, { ...h, "Content-Range": `bytes ${a}-${b}/${st.size}`, "Content-Length": b - a + 1 }); fs.createReadStream(p, { start: a, end: b }).pipe(res); }
    else { res.writeHead(200, { ...h, "Content-Length": st.size }); fs.createReadStream(p).pipe(res); } });
}).listen(port, "127.0.0.1", () => console.log(`serving ${root} on http://127.0.0.1:${port}`));
