// harness/constrain.js: tool-call name masking.
import { ToolCallConstraint } from "../../harness/constrain.js";
const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

const VOCAB = ["<tool_call>", "\n", "<function=", "read", "_file", "run", "rm", ">", ">\n", "<parameter=", "path", "cmd", "x", "hello", "</parameter>", "</function>", "</tool_call>", "re", "ad_file>", "\"", "{\"name\": \"", "run\""];
const tools = [
  { name: "read_file", parameters: { properties: { path: {}, max_lines: {} } } },
  { name: "run", parameters: { properties: { cmd: {} } } },
];
const words = (m) => m ? VOCAB.filter((_, i) => m[i]) : null;
const mk = (style = "xml") => new ToolCallConstraint(tools, { vocabSize: VOCAB.length, tokenText: (i) => VOCAB[i], style });

Deno.test("free text outside tool calls", () => {
  const C = mk();
  eq(C.allowed(), null);
  C.push("hello "); eq(C.allowed(), null);
});

Deno.test("xml: function names limited to declared tools, step by step", () => {
  const C = mk();
  C.push("<tool_call>"); C.push("\n"); eq(C.allowed(), null, "between tags is free");
  C.push("<function=");
  eq(words(C.allowed()), ["read", "run", "re", "run\""].filter((w) => ["read", "run", "re"].includes(w)), "only prefixes of read_file / run");
  C.push("re");
  eq(words(C.allowed()), ["ad_file>"]);
  C.push("ad_file>");
  eq(C.allowed(), null, "name done");
  C.push("\n"); C.push("<parameter=");
  eq(words(C.allowed()), ["path"], "read_file's params only (max_lines has no token here)");
  C.push("path"); eq(words(C.allowed()), [">", ">\n"]);
  C.push(">\n"); C.push("hello"); eq(C.allowed(), null, "values are free");
  C.push("</parameter>"); C.push("</function>"); C.push("</tool_call>");
  eq(C.allowed(), null, "after the call: free again");
});

Deno.test("json: the name string is limited", () => {
  const C = mk("json");
  C.push("<tool_call>\n{\"name\": \"");
  eq(words(C.allowed()), ["read", "run", "re", "run\""]);
  C.push("run\"");
  eq(C.allowed(), null);
});

Deno.test("mask() sets disallowed logits to -Infinity and caches the mask", () => {
  const C = mk();
  C.push("<tool_call>\n<function=");
  const lg = new Float32Array(VOCAB.length).fill(1);
  C.mask(lg);
  eq([...lg].map((x, i) => (x === 1 ? VOCAB[i] : null)).filter(Boolean), ["read", "run", "re"]);
  ok(C.allowed() === C.allowed(), "cached");
});
