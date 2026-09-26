// harness/tools.js: tool prompts, streaming tool-call parsing (both Qwen formats), results.
import { detectStyle, toolsSystemPrompt, toolResponses, renderCalls, parseCallBody, ToolCallParser } from "../../harness/tools.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

const TOOLS = [
  { name: "read_file", description: "Read a file", parameters: { type: "object", properties: { path: { type: "string" }, max_lines: { type: "integer" } }, required: ["path"] } },
  { name: "run", description: "Run a command", parameters: { type: "object", properties: { cmd: { type: "string" }, dry: { type: "boolean" } } } },
];
const schemaFor = (n) => TOOLS.find((t) => t.name === n)?.parameters;

Deno.test("detectStyle reads the chat template", () => {
  eq(detectStyle("{{- '<tool_call>\\n<function=' + tool_call.name }}"), "xml");
  eq(detectStyle("{{- '<tool_call>\\n{\"name\": \"' }}"), "json");
  eq(detectStyle(undefined), "json");
});

Deno.test("toolsSystemPrompt keeps the system text and lists every tool", () => {
  const p = toolsSystemPrompt(TOOLS, { system: "You are Tabby." });
  ok(p.startsWith("You are Tabby.\n\n# Tools"), p.slice(0, 40));
  ok(p.includes('"name":"read_file"') && p.includes('"name":"run"'));
  ok(p.includes("<tool_call>\n{\"name\": <function-name>"));
  const x = toolsSystemPrompt(TOOLS, { style: "xml", system: "You are Tabby." });
  ok(x.startsWith("# Tools") && x.endsWith("\n\nYou are Tabby.") && x.includes("<function=example_function_name>"), "xml: tools first, system appended");
  eq(toolsSystemPrompt([], { system: "x" }), "x");
});

Deno.test("parseCallBody: JSON and XML", () => {
  eq(parseCallBody('\n{"name": "read_file", "arguments": {"path": "a.js"}}\n'), { name: "read_file", arguments: { path: "a.js" } });
  eq(parseCallBody('{"name": "run", "arguments": "{\\"cmd\\": \\"ls\\"}"}'), { name: "run", arguments: { cmd: "ls" } });
  eq(parseCallBody("\n<function=read_file>\n<parameter=path>\nsrc/a b.js\n</parameter>\n<parameter=max_lines>\n40\n</parameter>\n</function>\n", schemaFor),
    { name: "read_file", arguments: { path: "src/a b.js", max_lines: 40 } });
  eq(parseCallBody("<function=run>\n<parameter=cmd>\necho 'a\nb'\n</parameter>\n<parameter=dry>\ntrue\n</parameter>\n</function>", schemaFor),
    { name: "run", arguments: { cmd: "echo 'a\nb'", dry: true } });
  eq(parseCallBody("<function=read_file>\n<parameter=path>\na.js\n<parameter=max_lines>\n3\n</function>", schemaFor),
    { name: "read_file", arguments: { path: "a.js", max_lines: 3 } }, "missing </parameter> tolerated");
  ok(parseCallBody("{not json").error, "malformed JSON is an error, not a throw");
  ok(parseCallBody('{"arguments": {}}').error, "a call needs a name");
});

Deno.test("ToolCallParser streams text and calls, never showing half a tag", () => {
  const full = "Let me look.\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.js\"}}\n</tool_call>\n<tool_call>\n<function=run>\n<parameter=cmd>\nls\n</parameter>\n</function>\n</tool_call>";
  for (const step of [1, 3, 7, 1000]) {
    const P = new ToolCallParser({ schemaFor });
    let text = "", calls = [];
    for (let i = 0; i < full.length; i += step) {
      const r = P.feed(full.slice(i, i + step));
      ok(!r.text.includes("<"), `step ${step}: leaked a tag fragment: ${JSON.stringify(r.text)}`);
      text += r.text; calls.push(...r.calls);
    }
    const e = P.end(); text += e.text; calls.push(...e.calls);
    eq(text, "Let me look.\n", `step ${step} text`);
    eq(calls, [{ name: "read_file", arguments: { path: "a.js" } }, { name: "run", arguments: { cmd: "ls" } }], `step ${step} calls`);
  }
});

Deno.test("ToolCallParser: plain answers pass through; unterminated calls are reported", () => {
  const P = new ToolCallParser();
  eq(P.feed("a < b and <tool").text, "a < b and ");
  eq(P.feed("s> done").text, "<tools> done");
  eq(P.end().text, "");
  const Q = new ToolCallParser({ schemaFor });
  Q.feed("<tool_call>\n<function=run>\n<parameter=cmd>\nls\n</parameter>\n</function>");
  eq(Q.end().calls, [{ name: "run", arguments: { cmd: "ls" } }], "closing tag missing at EOS still yields the call");
  const R = new ToolCallParser();
  R.feed("<tool_call>\n{\"name\": \"x\"");
  ok(R.end().calls[0].error);
});

Deno.test("toolResponses and renderCalls round-trip through the parser", () => {
  eq(toolResponses(["ok", { lines: 3 }]), "<tool_response>\nok\n</tool_response>\n<tool_response>\n{\"lines\":3}\n</tool_response>");
  const calls = [{ name: "read_file", arguments: { path: "a.js", max_lines: 5 } }];
  for (const style of ["json", "xml"]) {
    const P = new ToolCallParser({ schemaFor });
    const r = P.feed(renderCalls(calls, style)); const e = P.end();
    eq([...r.calls, ...e.calls], calls, style);
  }
});
