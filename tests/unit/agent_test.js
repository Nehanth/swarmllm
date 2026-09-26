// harness/workspace.js, codetools.js, agent.js: the agent loop over an in-memory project with a
// scripted model.
import { MemoryWorkspace, normPath } from "../../harness/workspace.js";
import { codingTools } from "../../harness/codetools.js";
import { Agent } from "../../harness/agent.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

const project = () => new MemoryWorkspace({
  "src/add.js": "export function add(a, b) {\n  return a - b;\n}\n",
  "src/util/str.js": "export const up = (s) => s.toUpperCase();\n",
  "README.md": "# demo\n",
});
const tool = (ws, n) => codingTools(ws).find((t) => t.name === n);

Deno.test("paths stay inside the workspace", () => {
  eq(normPath("./src//a.js"), "src/a.js");
  let threw = false; try { normPath("../etc/passwd"); } catch { threw = true; }
  ok(threw);
});

Deno.test("list_dir, read_file, search", async () => {
  const ws = project();
  eq(await tool(ws, "list_dir").run({}), "README.md\nsrc/");
  eq(await tool(ws, "list_dir").run({ path: "src" }), "add.js\nutil/");
  ok((await tool(ws, "read_file").run({ path: "src/add.js" })).includes("    2\t  return a - b;"));
  eq(await tool(ws, "search").run({ pattern: "return" }), "src/add.js:2: return a - b;");
  eq(await tool(ws, "search").run({ pattern: "UPPER", ignore_case: true, path: "src/util" }), "src/util/str.js:1: export const up = (s) => s.toUpperCase();");
  ok((await tool(ws, "search").run({ pattern: "(" })).startsWith("error: bad pattern"));
});

Deno.test("read_file pages long files", async () => {
  const ws = new MemoryWorkspace({ "big.txt": Array.from({ length: 1000 }, (_, i) => "line " + (i + 1)).join("\n") });
  const r = await tool(ws, "read_file").run({ path: "big.txt" });
  ok(r.includes("(lines 1-400 of 1000; read on with start_line=401)"));
  ok((await tool(ws, "read_file").run({ path: "big.txt", start_line: 990 })).trim().endsWith("line 1000"));
});

Deno.test("edit_file needs one exact match", async () => {
  const ws = project();
  const ed = tool(ws, "edit_file");
  ok((await ed.run({ path: "src/add.js", old_string: "a + b", new_string: "x" })).includes("not found"));
  await ws.write("dup.js", "x\nx\n");
  ok((await ed.run({ path: "dup.js", old_string: "x", new_string: "y" })).includes("appears 2 times"));
  eq(await ed.run({ path: "src/add.js", old_string: "a - b", new_string: "a + b" }), "edited src/add.js at line 2: -1 +1 lines");
  eq(await ws.read("src/add.js"), "export function add(a, b) {\n  return a + b;\n}\n");
});

// a scripted model: each call yields the next reply in small pieces
function scripted(replies, seen) {
  let i = 0;
  return async function* ({ system, turns }) {
    seen.push({ system, turns: turns.map((t) => ({ ...t })) });
    const r = replies[i++] ?? "done";
    for (let k = 0; k < r.length; k += 5) yield r.slice(k, k + 5);
  };
}

Deno.test("agent: read, edit, answer", async () => {
  const ws = project(), seen = [], events = [];
  const replies = [
    "I'll look at the file.\n<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/add.js\n</parameter>\n</function>\n</tool_call>",
    "<tool_call>\n<function=edit_file>\n<parameter=path>\nsrc/add.js\n</parameter>\n<parameter=old_string>\n  return a - b;\n</parameter>\n<parameter=new_string>\n  return a + b;\n</parameter>\n</function>\n</tool_call>",
    "Fixed: add() subtracted instead of adding.",
  ];
  const A = new Agent({ generate: scripted(replies, seen), tools: codingTools(ws), system: "You are Tabby.", onEvent: (e) => events.push(e.type) });
  const r = await A.run("add() is broken, fix it");
  eq(r, { text: "Fixed: add() subtracted instead of adding.", steps: 3, calls: 2 });
  eq(await ws.read("src/add.js"), "export function add(a, b) {\n  return a + b;\n}\n");
  ok(seen[0].system.includes("<tools>") && seen[0].system.endsWith("You are Tabby."), "tools in the system prompt");
  const t1 = seen[1].turns;
  eq(t1.map((t) => t.role), ["user", "assistant", "user"]);
  ok(t1[2].text.startsWith("<tool_response>\n    1\texport function add"), "results go back as <tool_response>");
  ok(events.includes("tool") && events[events.length - 1] === "done");
});

Deno.test("agent: declined edits, unknown tools and malformed calls are reported, not thrown", async () => {
  const ws = project(), seen = [];
  const replies = [
    "<tool_call>\n<function=write_file>\n<parameter=path>\nx.js\n</parameter>\n<parameter=content>\nhi\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=rm_rf>\n</function>\n</tool_call>\n<tool_call>\n{\"arguments\": {}}\n</tool_call>",
    "ok",
  ];
  const A = new Agent({ generate: scripted(replies, seen), tools: codingTools(ws), approve: async () => false });
  const r = await A.run("make x.js");
  eq(r.text, "ok");
  ok(!(await ws.exists("x.js")), "declined write did not happen");
  const back = seen[1].turns[2].text;
  ok(back.includes("the user declined this change") && back.includes("there is no tool called rm_rf") && back.includes("error: tool call has no name"), back);
});

Deno.test("agent stops at maxSteps", async () => {
  const ws = project();
  const loop = "<tool_call>\n<function=list_dir>\n</function>\n</tool_call>";
  const A = new Agent({ generate: scripted(Array(10).fill(loop), []), tools: codingTools(ws), maxSteps: 3 });
  eq((await A.run("go")).steps, 3);
});

Deno.test("agent trims old tool outputs past its budget, oldest first, keeping the latest", async () => {
  const big = "x".repeat(3000);
  const ws = new MemoryWorkspace({ "a.txt": big, "b.txt": big, "c.txt": big });
  const call = (f) => `<tool_call>\n<function=read_file>\n<parameter=path>\n${f}\n</parameter>\n</function>\n</tool_call>`;
  const seen = [];
  const A = new Agent({ generate: scripted([call("a.txt"), call("b.txt"), call("c.txt"), "done"], seen), tools: codingTools(ws), budget: 2500 });
  const r = await A.run("read them");
  eq(r.text, "done");
  const last = seen[seen.length - 1].turns;
  const res = last.filter((t) => t.text.startsWith("<tool_response>"));
  eq(res.length, 3);
  ok(res[0].text.includes("[output removed"), "oldest trimmed");
  ok(res[2].text.includes("xxxx"), "latest kept");
  const size = (seen[seen.length - 1].system.length + last.reduce((n, t) => n + t.text.length, 0)) / 3.5;
  ok(size < 2500 * 1.2, `size after trimming ~${Math.round(size)} tokens`);
});
