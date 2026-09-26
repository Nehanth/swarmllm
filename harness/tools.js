// Tool calling for the Qwen chat template: how tools are described to the model, how its tool
// calls are recognised in the streamed answer, and how results go back. DOM-free and
// engine-free so the harness (and the unit tests) can use it anywhere.
//
// Two formats exist in the Qwen family, and a model is trained on exactly one of them:
//   "json" (Qwen2.5 / Qwen3, Hermes-style):
//       <tool_call>
//       {"name": "read_file", "arguments": {"path": "src/a.js"}}
//       </tool_call>
//   "xml" (Qwen3-Coder, Qwen3.5 and later):
//       <tool_call>
//       <function=read_file>
//       <parameter=path>
//       src/a.js
//       </parameter>
//       </function>
//       </tool_call>
// Results go back in a user turn as <tool_response> ... </tool_response> blocks, one per call,
// in call order. detectStyle() reads the GGUF's own chat template (tokenizer.chat_template) so the
// prompt matches what the model was trained on; the parser accepts both formats either way.

// tools: [{ name, description, parameters: JSON schema object }]

export function detectStyle(chatTemplate = "") {
  return /<function=|<parameter=/.test(chatTemplate) ? "xml" : "json";
}

const fnJSON = (t) => JSON.stringify({ type: "function", function: { name: t.name, description: t.description || "", parameters: t.parameters || { type: "object", properties: {} } } });

// The system prompt with the tool block appended, as the Qwen templates render it.
export function toolsSystemPrompt(tools, { style = "json", system = "" } = {}) {
  if (!tools || !tools.length) return system;
  const list = "# Tools\n\nYou have access to the following functions:\n\n<tools>\n" + tools.map(fnJSON).join("\n") + "\n</tools>";
  if (style === "xml") {   // Qwen3.5+ templates: tool block first, the caller's system text appended
    return list + "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
      + "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
      + "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
      + "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
      + "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
      + "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"
      + (system ? "\n\n" + system : "");
  }
  return (system ? system + "\n\n" : "") + list + "\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
    + "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
}

// The text of a tool-results user turn: one <tool_response> block per result, in call order.
export function toolResponses(results) {
  return results.map((r) => "<tool_response>\n" + (typeof r === "string" ? r : JSON.stringify(r)) + "\n</tool_response>").join("\n");
}

// An assistant turn's tool calls rendered back in the model's format (for history rebuilt from
// structured calls; a live session should keep the sampled ids instead, see room/conversation.js).
export function renderCalls(calls, style = "json") {
  return calls.map((c) => style === "xml"
    ? "<tool_call>\n<function=" + c.name + ">\n" + Object.entries(c.arguments || {}).map(([k, v]) =>
      "<parameter=" + k + ">\n" + (typeof v === "string" ? v : JSON.stringify(v)) + "\n</parameter>\n").join("") + "</function>\n</tool_call>"
    : "<tool_call>\n" + JSON.stringify({ name: c.name, arguments: c.arguments || {} }) + "\n</tool_call>").join("\n");
}

// Parse one <tool_call> body (the text between the tags) in either format.
// Returns { name, arguments } or { error, raw } when the model produced something malformed.
export function parseCallBody(body, schemaFor = () => null) {
  const raw = body;
  const b = body.trim();
  const fm = /^<function=([^>\s]+)>([\s\S]*?)(?:<\/function>\s*)?$/.exec(b);
  if (fm) {
    const name = fm[1], args = {};
    const props = schemaFor(name)?.properties || {};
    // a missing </parameter> before the next parameter or </function> is tolerated (seen in the wild)
    const re = /<parameter=([^>\s]+)>\n?([\s\S]*?)(?:\n?<\/parameter>|\n?(?=<parameter=)|\n?$)/g;
    let m;
    while ((m = re.exec(fm[2]))) args[m[1]] = coerce(m[2], props[m[1]]);
    return { name, arguments: args };
  }
  try {
    const o = JSON.parse(b);
    if (!o || typeof o.name !== "string") return { error: "tool call has no name", raw };
    let a = o.arguments ?? o.parameters ?? {};
    if (typeof a === "string") { try { a = JSON.parse(a); } catch { /* leave as text */ } }
    return { name: o.name, arguments: a };
  } catch (e) {
    return { error: "tool call is not valid JSON: " + e.message, raw };
  }
}

// XML parameters are text; turn them into the schema's type when it says number / boolean /
// object / array (the Qwen3-Coder parser does the same), else keep the string.
function coerce(text, schema) {
  const t = schema?.type;
  if (t === "string" || !t) {
    if (!t) { try { const v = JSON.parse(text); if (typeof v !== "string") return v; } catch { /* text */ } }
    return text;
  }
  if (t === "integer" || t === "number") { const n = Number(text.trim()); return Number.isFinite(n) ? n : text; }
  if (t === "boolean") { const s = text.trim().toLowerCase(); return s === "true" ? true : s === "false" ? false : text; }
  try { return JSON.parse(text); } catch { return text; }
}

// Streaming recogniser: feed() the decoded answer text as it grows (deltas); get back the text
// that is safe to show (never a half-typed "<tool_c") and any tool calls completed so far.
// end() flushes the rest; an unterminated <tool_call> at the end is reported as an error call.
export class ToolCallParser {
  constructor({ schemaFor = () => null } = {}) {
    this.buf = ""; this.inCall = false; this.calls = []; this.schemaFor = schemaFor;
  }
  feed(delta) {
    this.buf += delta;
    let text = "";
    const calls = [];
    for (;;) {
      if (!this.inCall) {
        if (this.eatNL && this.buf) { if (this.buf[0] === "\n") this.buf = this.buf.slice(1); this.eatNL = false; }   // the newline between calls
        const i = this.buf.indexOf("<tool_call>");
        if (i < 0) {
          // hold back a tail that could be the start of the tag
          const keep = partialTagTail(this.buf, "<tool_call>");
          text += this.buf.slice(0, this.buf.length - keep);
          this.buf = this.buf.slice(this.buf.length - keep);
          break;
        }
        text += this.buf.slice(0, i);
        this.buf = this.buf.slice(i + "<tool_call>".length);
        this.inCall = true;
      } else {
        const j = this.buf.indexOf("</tool_call>");
        if (j < 0) break;
        const c = parseCallBody(this.buf.slice(0, j), this.schemaFor);
        calls.push(c); this.calls.push(c);
        this.buf = this.buf.slice(j + "</tool_call>".length);
        this.inCall = false; this.eatNL = true;
      }
    }
    return { text, calls };
  }
  end() {
    const r = { text: "", calls: [] };
    if (this.inCall) {
      // a model that stops right after </function> without closing the call still meant it
      const c = /<function=/.test(this.buf) ? parseCallBody(this.buf, this.schemaFor) : { error: "unterminated <tool_call>", raw: this.buf };
      r.calls.push(c); this.calls.push(c);
    } else r.text = this.buf;
    this.buf = ""; this.inCall = false;
    return r;
  }
}

function partialTagTail(s, tag) {
  for (let k = Math.min(tag.length - 1, s.length); k > 0; k--) if (tag.startsWith(s.slice(s.length - k))) return k;
  return 0;
}

