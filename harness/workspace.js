// The files a Tabby agent works on. Two implementations with one interface:
//   MemoryWorkspace   - a Map of path -> text (tests, scratch projects)
//   DirWorkspace      - a folder the user picked with showDirectoryPicker() (File System Access
//                       API): reads and writes go to the real files on their disk, nothing leaves
//                       the machine
// Paths are relative, "/"-separated, and may not climb out of the root ("..").
//
// interface: list(dir) -> [{ name, dir: bool }], read(path) -> string, write(path, text),
//            exists(path) -> bool, walk() -> [path] (every file, for search)

export function normPath(p) {
  const parts = [];
  for (const s of String(p || "").replace(/\\/g, "/").split("/")) {
    if (!s || s === ".") continue;
    if (s === "..") throw new Error(`path leaves the workspace: ${p}`);
    parts.push(s);
  }
  return parts.join("/");
}

export class MemoryWorkspace {
  constructor(files = {}) { this.files = new Map(Object.entries(files).map(([k, v]) => [normPath(k), v])); }
  async read(p) {
    const k = normPath(p);
    if (!this.files.has(k)) throw new Error(`no such file: ${k}`);
    return this.files.get(k);
  }
  async write(p, text) { this.files.set(normPath(p), text); }
  async exists(p) { const k = normPath(p); return this.files.has(k) || [...this.files.keys()].some((f) => f.startsWith(k + "/")); }
  async list(dir = "") {
    const d = normPath(dir), pre = d ? d + "/" : "", seen = new Map();
    for (const f of this.files.keys()) {
      if (!f.startsWith(pre)) continue;
      const rest = f.slice(pre.length), i = rest.indexOf("/");
      if (i < 0) seen.set(rest, false); else seen.set(rest.slice(0, i), true);
    }
    if (d && !seen.size) throw new Error(`no such directory: ${d}`);
    return [...seen].map(([name, dir]) => ({ name, dir })).sort((a, b) => a.name.localeCompare(b.name));
  }
  async walk() { return [...this.files.keys()].sort(); }
}

// Folders nobody wants an agent to read through.
export const SKIP_DIRS = new Set([".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".next", "target"]);

export class DirWorkspace {
  constructor(handle) { this.root = handle; }   // a FileSystemDirectoryHandle
  async _dir(parts, create = false) {
    let h = this.root;
    for (const s of parts) h = await h.getDirectoryHandle(s, { create });
    return h;
  }
  async read(p) {
    const parts = normPath(p).split("/");
    const f = await (await this._dir(parts.slice(0, -1))).getFileHandle(parts[parts.length - 1]);
    return (await f.getFile()).text();
  }
  async write(p, text) {
    const parts = normPath(p).split("/");
    const f = await (await this._dir(parts.slice(0, -1), true)).getFileHandle(parts[parts.length - 1], { create: true });
    const w = await f.createWritable(); await w.write(text); await w.close();
  }
  async exists(p) {
    const parts = normPath(p).split("/");
    try {
      const d = await this._dir(parts.slice(0, -1));
      try { await d.getFileHandle(parts[parts.length - 1]); } catch { await d.getDirectoryHandle(parts[parts.length - 1]); }
      return true;
    } catch { return false; }
  }
  async list(dir = "") {
    const d = await this._dir(normPath(dir) ? normPath(dir).split("/") : []), out = [];
    for await (const [name, h] of d.entries()) out.push({ name, dir: h.kind === "directory" });
    return out.sort((a, b) => a.name.localeCompare(b.name));
  }
  async walk(limit = 5000) {
    const out = [];
    const go = async (h, pre) => {
      for await (const [name, c] of h.entries()) {
        if (out.length >= limit) return;
        if (c.kind === "directory") { if (!SKIP_DIRS.has(name)) await go(c, pre + name + "/"); }
        else out.push(pre + name);
      }
    };
    await go(this.root, "");
    return out.sort();
  }
}
