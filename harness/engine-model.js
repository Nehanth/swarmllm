// The agent's model interface (harness/agent.js: generate({ system, turns }) -> async iterable of
// text) over one Qwen35Engine and its tokenizer. Every request re-renders the whole conversation
// with the chat template, but only the tokens after what the engine already holds are prefilled:
// an agent resends ~96% of its input every step (docs/research/tabby-2026-09.md §3).
//
// The model's own turns are kept as the exact ids it sampled (keyed by their text), so rendering
// never re-tokenizes them differently and the prefix stays reusable. Decoding is greedy by
// default; with the draft head, speculative steps give the same tokens faster.
import { buildIds, reusablePrefix, specials } from "../room/conversation.js";
import { ToolCallConstraint } from "./constrain.js";

// tools (optional): the agent's tool list; inside a tool call the function and parameter names
// are then limited to declared ones (harness/constrain.js), on every sampled position including
// the ones a speculative step checks, so accepted tokens always satisfy it.
export function engineModel(engine, tok, { thinking = false, maxNew = 1024, K = 3, spec = true, sample = null, tools = null, style = "xml" } = {}) {
  const S = specials(tok);
  const pick0 = sample || ((lg) => { let b = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[b]) b = i; return b; });
  const texts = [];
  const tokText = (id) => (texts[id] ??= tok.decode([id]));
  const C = tools?.length ? new ToolCallConstraint(tools, { vocabSize: engine.dims?.vocab ?? Object.keys(tok.vocab).length, tokenText: tokText, style }) : null;
  let base = "", pend = "";   // text of the answer so far / of tokens sampled in the current step
  const pick = (lg) => {
    if (C) { C.text = base + pend; C.mask(lg); }
    const t = pick0(lg);
    if (C) pend += tokText(t);
    return t;
  };
  const stop = new Set([S.imEnd, S.eot].filter(Number.isInteger));
  const own = new Map();   // assistant text -> the ids it was sampled as
  let fed = [];            // exactly the tokens the engine's caches hold
  const stats = { calls: 0, reused: 0, prefilled: 0, generated: 0 };

  async function* generate({ system = "", turns, signal } = {}) {
    stats.calls++;
    const T = turns.map((t) => (t.role === "assistant" ? { role: "assistant", ids: own.get(t.text) || tok.encode(t.text) } : { role: "user", text: t.text }));
    const ids = buildIds(tok, { system, turns: T, thinking });
    if (ids.length + 2 > engine.maxSeq) throw new Error(`conversation is ${ids.length} tokens; the context is ${engine.maxSeq}`);
    const reused = reusablePrefix(fed, ids);
    if (!reused) { engine.reset(); fed = []; }
    stats.reused += reused; stats.prefilled += ids.length - reused;
    const rest = ids.slice(reused);
    if (rest.length > 1) await engine.prefillTokens(rest.slice(0, -1));
    fed = ids.slice(0, -1);   // (prefillTokens wrote all but the last prompt token)
    base = ""; pend = "";
    let next = pick(await engine.forwardToken(ids[ids.length - 1]));
    fed.push(ids[ids.length - 1]);
    // `next` is sampled, not yet written. A plain step writes it; a speculative step writes it
    // and the drafts it accepts, and returns the tokens sampled after it (the last one is the
    // new `next`), exactly as the room does.
    const out = [];
    let text = "", done = stop.has(next);
    if (!done) out.push(next);
    const room = () => Math.min(maxNew - out.length, engine.maxSeq - engine.pos - 2);
    base = tok.decode(out); pend = "";
    while (!done && room() > 0 && !signal?.aborted) {
      let toks;
      if (spec && engine.mtp && room() > K + 1) { toks = await engine.specStep(next, pick, K); fed.push(next, ...toks.slice(0, -1)); }
      else { toks = [pick(await engine.forwardToken(next))]; fed.push(next); }
      for (const t of toks) { if (stop.has(t)) { done = true; break; } out.push(t); }
      next = toks[toks.length - 1];
      const now = tok.decode(out);
      base = now; pend = "";   // the constraint sees exactly the answer so far
      if (now.length > text.length && !now.endsWith("\uFFFD")) { yield now.slice(text.length); text = now; }
    }
    const all = tok.decode(out);
    if (all.length > text.length) yield all.slice(text.length);
    own.set(all, out.slice());
    stats.generated += out.length;
  }
  return { generate, stats, get fed() { return fed; } };
}
