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
  constructor({ generate, tools, style = "xml", system = "", maxSteps = 24, approve = async () => true, onEvent = () => {} }) {
    this.generate = generate; this.tools = tools; this.style = style; this.maxSteps = maxSteps;
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
