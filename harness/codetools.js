// Coding tools for a Tabby agent over a workspace (harness/workspace.js). Each tool is
// { name, description, parameters (JSON schema), mutates, run(args) -> string }. Results are
// plain text written for the model: short, with line numbers, and with errors that say how to
// recover. Tools that change files are marked `mutates` so the agent can ask the user first.

const MAX_LINES = 400, MAX_CHARS = 24000, MAX_HITS = 60;

const numbered = (lines, from) => lines.map((l, i) => `${String(from + i).padStart(5)}\t${l}`).join("\n");

export function codingTools(ws) {
  return [
    {
      name: "list_dir", mutates: false,
      description: "List the files and folders in a directory of the project (\"\" or \".\" for the root). Folders end with /.",
      parameters: { type: "object", properties: { path: { type: "string", description: "directory, relative to the project root" } } },
      async run({ path = "" } = {}) {
        const es = await ws.list(path);
        return es.length ? es.map((e) => e.name + (e.dir ? "/" : "")).join("\n") : "(empty)";
      },
    },
    {
      name: "read_file", mutates: false,
      description: `Read a text file with line numbers. Long files come in pieces of ${MAX_LINES} lines: pass start_line / end_line for the rest.`,
      parameters: { type: "object", properties: { path: { type: "string" }, start_line: { type: "integer" }, end_line: { type: "integer" } }, required: ["path"] },
      async run({ path, start_line, end_line }) {
        const lines = (await ws.read(path)).split("\n");
        const from = Math.max(1, start_line || 1);
        let to = Math.min(lines.length, end_line || from + MAX_LINES - 1, from + MAX_LINES - 1);
        let body = numbered(lines.slice(from - 1, to), from);
        while (body.length > MAX_CHARS && to > from) { to = from + Math.floor((to - from) / 2); body = numbered(lines.slice(from - 1, to), from); }
        const more = to < lines.length ? `\n(lines ${from}-${to} of ${lines.length}; read on with start_line=${to + 1})` : "";
        return body + more;
      },
    },
    {
      name: "search", mutates: false,
      description: "Search the project's files for a regular expression (JavaScript syntax). Returns path:line: text for each match.",
      parameters: { type: "object", properties: { pattern: { type: "string" }, path: { type: "string", description: "only files under this folder" }, ignore_case: { type: "boolean" } }, required: ["pattern"] },
      async run({ pattern, path = "", ignore_case = false }) {
        let re;
        try { re = new RegExp(pattern, ignore_case ? "i" : ""); } catch (e) { return `error: bad pattern: ${e.message}`; }
        const pre = path && path !== "." ? path.replace(/\/+$/, "") + "/" : "";
        const hits = [];
        for (const f of await ws.walk()) {
          if (pre && !f.startsWith(pre)) continue;
          let text;
          try { text = await ws.read(f); } catch { continue; }
          if (text.includes("\u0000")) continue;   // binary
          const lines = text.split("\n");
          for (let i = 0; i < lines.length && hits.length < MAX_HITS; i++) if (re.test(lines[i])) hits.push(`${f}:${i + 1}: ${lines[i].trim().slice(0, 200)}`);
          if (hits.length >= MAX_HITS) break;
        }
        return hits.length ? hits.join("\n") + (hits.length >= MAX_HITS ? `\n(first ${MAX_HITS} matches)` : "") : "no matches";
      },
    },
    {
      name: "edit_file", mutates: true,
      description: "Replace one exact piece of text in a file. old_string must appear exactly once (include a few surrounding lines to make it unique); new_string replaces it.",
      parameters: { type: "object", properties: { path: { type: "string" }, old_string: { type: "string" }, new_string: { type: "string" } }, required: ["path", "old_string", "new_string"] },
      async run({ path, old_string, new_string }) {
        const text = await ws.read(path);
        if (!old_string) return "error: old_string is empty; use write_file to create a file";
        const n = text.split(old_string).length - 1;
        if (n === 0) return `error: old_string not found in ${path}; read_file it again and copy the text exactly (whitespace included)`;
        if (n > 1) return `error: old_string appears ${n} times in ${path}; include more surrounding lines so it is unique`;
        const at = text.indexOf(old_string);
        await ws.write(path, text.slice(0, at) + new_string + text.slice(at + old_string.length));
        const line = text.slice(0, at).split("\n").length;
        return `edited ${path} at line ${line}: -${old_string.split("\n").length} +${new_string.split("\n").length} lines`;
      },
    },
    {
      name: "write_file", mutates: true,
      description: "Create a file, or replace a whole file, with the given content.",
      parameters: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] },
      async run({ path, content }) {
        const existed = await ws.exists(path);
        await ws.write(path, content);
        return `${existed ? "replaced" : "created"} ${path} (${content.split("\n").length} lines)`;
      },
    },
  ];
}
