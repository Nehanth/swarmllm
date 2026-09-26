// The Tabby agent loop: ask the model, run the tools it calls, hand back the results, repeat
// until it answers without calling a tool. The model is any function
//   generate({ system, turns }) -> async iterable of text deltas
// (the swarm, a single engine, or a script in tests), so the loop knows nothing about GPUs.
//
// turns: [{ role: "user" | "assistant", text }]. Tool results go back as one user turn of
// <tool_response> blocks, as the Qwen templates expect. Assistant turns keep the model's raw
// text (tool-call markup included) so the history the model sees is exactly what it wrote.
import { toolsSystemPrompt, toolResponses, ToolCallParser } from "./tools.js";

export class Agent {
  // budget: tokens the conversation may take (the context minus room for an answer); count:
  // text -> tokens (a tokenizer's encode().length; default ~3.5 characters per token). Past the
  // budget, the oldest tool outputs are cut to a stub first: they are most of an agent's context
  // and the model can run the tool again. Cutting changes the prompt, so the next step prefills
  // from the first cut turn: done only when needed, oldest first, a whole batch at a time.
  constructor({ generate, tools, style = "xml", system = "", maxSteps = 24, approve = async () => true, onEvent = () => {}, budget = Infinity, count = null }) {
    this.generate = generate; this.tools = tools; this.style = style; this.maxSteps = maxSteps;
    this.budget = budget; this.count = count || ((t) => Math.ceil(t.length / 3.5));
    this.approve = approve; this.onEvent = onEvent;
    this.system = toolsSystemPrompt(tools.map(({ name, description, parameters }) => ({ name, description, parameters })), { style, system });
    this.turns = [];
    this.byName = new Map(tools.map((t) => [t.name, t]));
  }
  // Run one user request to the end. Returns { text, steps, calls } (text: the final answer).
  async run(userText, { signal } = {}) {
    this.turns.push({ role: "user", text: userText });
    let calls = 0;
    for (let step = 1; step <= this.maxSteps; step++) {
      if (signal?.aborted) throw new Error("stopped");
      this._fit();
      const P = new ToolCallParser({ schemaFor: (n) => this.byName.get(n)?.parameters });
      let raw = "", shown = "";
      const found = [];
      for await (const d of this.generate({ system: this.system, turns: this.turns, signal })) {
        raw += d;
        const r = P.feed(d);
        shown += r.text; found.push(...r.calls);
        if (r.text) this.onEvent({ type: "text", text: r.text, step });
      }
      const e = P.end();
      shown += e.text; found.push(...e.calls);
      if (e.text) this.onEvent({ type: "text", text: e.text, step });
      this.turns.push({ role: "assistant", text: raw });
      if (!found.length) { this.onEvent({ type: "done", step }); return { text: shown.trim(), steps: step, calls }; }
      const results = [];
      for (const c of found) {
        calls++;
        results.push(await this._runCall(c, step));
      }
      this.turns.push({ role: "user", text: toolResponses(results) });
    }
    this.onEvent({ type: "limit", steps: this.maxSteps });
    return { text: `(stopped after ${this.maxSteps} steps)`, steps: this.maxSteps, calls };
  }
  _size() { return this.count(this.system) + this.turns.reduce((n, t) => n + this.count(t.text) + 4, 0); }
  _fit() {
    if (this._size() <= this.budget) return;
    const stub = "[output removed to save context; run the tool again if you need it]";
    // tool-result turns, oldest first, never the latest one (the model is about to read it)
    const res = this.turns.map((t, i) => i).filter((i) => this.turns[i].role === "user" && this.turns[i].text.startsWith("<tool_response>") && i < this.turns.length - 1);
    let cut = 0;
    for (const i of res) {
      if (this._size() <= this.budget * 0.75) break;   // leave headroom so the next steps do not cut again
      const t = this.turns[i];
      const trimmed = t.text.replace(/<tool_response>\n([\s\S]*?)\n<\/tool_response>/g, (m, body) => (body.length > 200 ? `<tool_response>\n${stub}\n</tool_response>` : m));
      if (trimmed !== t.text) { t.text = trimmed; cut++; }
    }
    if (cut) this.onEvent({ type: "trimmed", turns: cut });
  }
  async _runCall(c, step) {
    let result;
    if (c.error) result = `error: ${c.error}. Write the call again in the format the system prompt shows.`;
    else {
      const t = this.byName.get(c.name);
      if (!t) result = `error: there is no tool called ${c.name}; the tools are ${[...this.byName.keys()].join(", ")}`;
      else if (t.mutates && !(await this.approve(c))) result = "the user declined this change";
      else {
        try { result = String(await t.run(c.arguments || {})); }
        catch (err) { result = `error: ${err.message}`; }
      }
    }
    this.onEvent({ type: "tool", call: c, result, step });
    return result;
  }
}
