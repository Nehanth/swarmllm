// Conversation state for the room host: the chat template, the context budget, and how much of
// the conversation the swarm's caches already hold, so a new turn prefills only what is new.
// DOM-free so it can be unit tested.
//
// Every device keeps its KV caches and recurrent (DeltaNet) state between questions. That state
// is a function of exactly the tokens written so far (`fed`), in order. A new turn can continue
// from it only when `fed` is a strict prefix of the new conversation's ids; anything else (the
// system prompt changed, older turns were dropped to fit the context, the answer stopped on a
// token the template does not end with) means a reset and a full re-prefill, which is always
// correct, just slower. Correctness never depends on guessing: it depends on `fed` being exact.

// Answer styles the host can pick. The key goes over the wire; the text is the system prompt.
export const PERSONAS = {
  default: { label: "plain", system: "" },
  concise: { label: "concise", system: "Answer in at most three short sentences." },
  eli5: { label: "explain like I'm five", system: "Explain everything as if to a curious five-year-old: short sentences, everyday words, one vivid comparison." },
  pirate: { label: "pirate", system: "You are a cheerful pirate. Answer every question correctly and helpfully, but talk like a pirate." },
  haiku: { label: "haiku", system: "Answer every question as a single haiku (three lines, 5-7-5 syllables). Nothing else." },
  swarm: { label: "the swarm speaks", system: "You are Swarmy, a hive mind whose thoughts are split across several phones and laptops in this room; every word you say takes a lap through all of them. You find this delightful and say so now and then, but you still answer the question well." },
};

// Special-token ids the template needs; throws with a readable message when the tokenizer lacks them.
export function specials(tok) {
  const V = tok.vocab;
  const s = { imStart: V["<|im_start|>"], imEnd: V["<|im_end|>"], eot: V["<|endoftext|>"], think: V["<think>"], thinkEnd: V["</think>"] };
  if (!Number.isInteger(s.imStart) || !Number.isInteger(s.imEnd))
    throw new Error("this model's tokenizer has no chat tokens (<|im_start|>, <|im_end|>)");
  return s;
}

// ChatML ids for a conversation. turns: [{role: "user", text} | {role: "assistant", ids, open?}],
// ending with a user turn, or with an open assistant turn (Continue). Assistant turns carry the exact sampled ids (never re-tokenized text, which
// can split differently) so the history matches what the caches hold token for token.
// thinking=false pre-closes the think block on every assistant turn (Qwen3 family), so answers
// come straight; the first turn's ids are the same as the single-turn template this replaces.
export function buildIds(tok, { system = "", turns, thinking = false }) {
  const S = specials(tok);
  const nl = tok.encode("\n");
  const closeThink = !thinking && S.think !== undefined && S.thinkEnd !== undefined
    ? [S.think, ...tok.encode("\n\n"), S.thinkEnd, ...tok.encode("\n\n")] : [];
  const ids = [];
  if (system) ids.push(S.imStart, ...tok.encode("system\n" + system), S.imEnd, ...nl);
  for (const t of turns) {
    if (t.role === "user") ids.push(S.imStart, ...tok.encode("user\n" + t.text), S.imEnd, ...nl, S.imStart, ...tok.encode("assistant\n"), ...closeThink);
    else if (t.open) ids.push(...t.ids);   // an answer being continued: left open, no end token
    else ids.push(...t.ids, S.imEnd, ...nl);
  }
  if (ids.some((t) => !Number.isInteger(t)))
    throw new Error("tokenizer produced an invalid token id (special tokens missing) — " + JSON.stringify(ids.slice(0, 6)));
  return ids;
}

// Fit the conversation into the context: drop the oldest whole exchanges until the prompt leaves
// `reserve` tokens for the answer. Returns { ids, turns, dropped } or throws when even the last
// question alone does not fit.
export function fitContext(tok, { system, turns, thinking }, maxSeq, reserve) {
  let t = turns.slice(), dropped = 0;
  for (;;) {
    const ids = buildIds(tok, { system, turns: t, thinking });
    if (ids.length <= maxSeq - reserve) return { ids, turns: t, dropped };
    // drop the oldest user turn and the answer that followed it; never the question being asked
    const nextUser = t.findIndex((x, i) => i > 0 && x.role === "user");
    if (nextUser < 0) {
      throw new Error(`this question is ${ids.length} tokens; the room's context is ${maxSeq} tokens and an answer needs at least ${reserve}. Shorten it.`);
    }
    t = t.slice(nextUser);
    dropped++;
  }
}

// How many leading tokens of `ids` the caches already hold: fed.length when fed is a strict
// prefix of ids (at least one new token is left to prefill, since the last prompt token must run
// through the head), else 0, meaning reset and prefill everything.
export function reusablePrefix(fed, ids) {
  if (!fed || !fed.length || fed.length >= ids.length) return 0;
  for (let i = 0; i < fed.length; i++) if (fed[i] !== ids[i]) return 0;
  return fed.length;
}

// Split raw answer text into the thinking part and the answer part. A think block that has not
// closed yet (still streaming) is all thinking.
export function splitThink(raw) {
  const a = raw.indexOf("<think>");
  if (a < 0) return { think: null, answer: raw };
  const b = raw.indexOf("</think>", a);
  if (b < 0) return { think: raw.slice(a + 7).trim(), answer: raw.slice(0, a), open: true };
  return { think: raw.slice(a + 7, b).trim(), answer: raw.slice(0, a) + raw.slice(b + 8).replace(/^\s+/, "") };
}
