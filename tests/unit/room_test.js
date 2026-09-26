// room/conversation.js, room/plan.js and room/qr.js: DOM-free room logic.
import { buildIds, fitContext, reusablePrefix, splitThink, PERSONAS } from "../../room/conversation.js";
import { planSplit, ladder, bestFit, codeFromLocation } from "../../room/plan.js";
import { qrMatrix, qrSVG, rsEncode } from "../../room/qr.js";
import { packFlags, unpackFlags } from "../../room/transport.js";

const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ok = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };

// a toy tokenizer: one id per character (1000 + code), plus the chat specials
const tok = {
  vocab: { "<|im_start|>": 1, "<|im_end|>": 2, "<|endoftext|>": 3, "<think>": 4, "</think>": 5 },
  encode: (s) => [...s].map((c) => 1000 + c.codePointAt(0)),
};
const E = (s) => tok.encode(s);

Deno.test("conversation: the first turn is the single-turn template it replaces", () => {
  const ids = buildIds(tok, { turns: [{ role: "user", text: "hi" }] });
  eq(ids, [1, ...E("user\nhi"), 2, ...E("\n"), 1, ...E("assistant\n"), 4, ...E("\n\n"), 5, ...E("\n\n")]);
});
Deno.test("conversation: thinking leaves the think block open; a system prompt comes first", () => {
  const ids = buildIds(tok, { system: "be brief", turns: [{ role: "user", text: "q" }], thinking: true });
  eq(ids, [1, ...E("system\nbe brief"), 2, ...E("\n"), 1, ...E("user\nq"), 2, ...E("\n"), 1, ...E("assistant\n")]);
});
Deno.test("conversation: a second turn extends the first turn's ids exactly (answer ids verbatim)", () => {
  const t1 = [{ role: "user", text: "a" }];
  const first = buildIds(tok, { turns: t1 });
  const answer = [77, 78, 79];
  const second = buildIds(tok, { turns: [...t1, { role: "assistant", ids: answer }, { role: "user", text: "b" }] });
  const fed = [...first, ...answer, 2];   // what the plain path writes when the answer ends on <|im_end|>
  eq(second.slice(0, fed.length), fed, "history prefix");
  eq(reusablePrefix(fed, second), fed.length);
  // the speculative path does not write the final <|im_end|>: still a prefix
  eq(reusablePrefix([...first, ...answer], second), first.length + answer.length);
});
Deno.test("conversation: reusablePrefix refuses anything but a strict prefix", () => {
  eq(reusablePrefix([], [1, 2]), 0);
  eq(reusablePrefix(null, [1, 2]), 0);
  eq(reusablePrefix([1, 2], [1, 2]), 0, "nothing new to prefill");
  eq(reusablePrefix([1, 2, 3], [1, 2]), 0);
  eq(reusablePrefix([1, 9], [1, 2, 3]), 0, "diverged");
  eq(reusablePrefix([1, 2], [1, 2, 3]), 2);
});
Deno.test("conversation: fitContext drops the oldest exchanges, never the question", () => {
  const turns = [{ role: "user", text: "x".repeat(40) }, { role: "assistant", ids: new Array(40).fill(9) },
    { role: "user", text: "y".repeat(10) }, { role: "assistant", ids: [9, 9] }, { role: "user", text: "z" }];
  const all = buildIds(tok, { turns }).length;
  const r0 = fitContext(tok, { turns }, all + 10, 10);
  eq(r0.dropped, 0);
  const r1 = fitContext(tok, { turns }, all, 10);
  eq(r1.dropped, 1); eq(r1.turns[0].text, "y".repeat(10));
  let threw = false;
  try { fitContext(tok, { turns: [{ role: "user", text: "w".repeat(100) }] }, 64, 16); } catch (e) { threw = /Shorten/.test(e.message); }
  ok(threw, "an over-long question is refused, not truncated");
});
Deno.test("conversation: splitThink", () => {
  eq(splitThink("plain"), { think: null, answer: "plain" });
  eq(splitThink("<think>\nhmm\n</think>\n\nyes"), { think: "hmm", answer: "yes" });
  eq(splitThink("<think>still going"), { think: "still going", answer: "", open: true });
  ok(PERSONAS.default.system === "", "the default persona adds no system prompt");
});

Deno.test("plan: planSplit gives every device a layer and covers every layer once", () => {
  const GB = 2 ** 30;
  for (const caps of [[10 * GB, 1 * GB], [0.1 * GB, 14 * GB, 0.5 * GB], [1, 1, 1, 1, 1], [5 * GB]]) {
    const { assigned, ranges } = planSplit(64, caps);
    eq(assigned.reduce((a, b) => a + b, 0), 64, "total");
    ok(assigned.every((a) => a >= 1), "at least one layer each");
    eq(ranges[0][0], 0); eq(ranges[ranges.length - 1][1], 64);
    for (let i = 1; i < ranges.length; i++) eq(ranges[i][0], ranges[i - 1][1], "contiguous");
  }
  eq(planSplit(64, [62, 2]).assigned, [62, 2], "the MacBook + iPhone demo split");
});
Deno.test("plan: ladder and bestFit", () => {
  const need = { big: 16.5, small: 0.6, mid: 4.6 };
  eq(ladder(need, 5).map((x) => [x.key, x.ok, x.short]), [["small", true, 0], ["mid", true, 0], ["big", false, 11.5]]);
  eq(bestFit(need, 5), "mid");
  eq(bestFit(need, 0.1), "small", "nothing fits: smallest");
  eq(bestFit(need, 99), "big");
});
Deno.test("plan: codeFromLocation reads /r/CODE, ?code= and #CODE", () => {
  eq(codeFromLocation("/r/ABCD"), "ABCD");
  eq(codeFromLocation("/r/abcd/"), "ABCD");
  eq(codeFromLocation("/room", "?code=xy23"), "XY23");
  eq(codeFromLocation("/room", "", "#K7MP"), "K7MP");
  eq(codeFromLocation("/room", "", "#debug"), "");
  eq(codeFromLocation("/r/IO01"), "", "letters the room never generates");
  eq(codeFromLocation("/room"), "");
});

Deno.test("qr: Reed-Solomon matches the ISO 18004 annex example", () => {
  // version 1-M "01234567" example codewords and their 10 ECC bytes
  eq(rsEncode([16, 32, 12, 86, 97, 128, 236, 17, 236, 17, 236, 17, 236, 17, 236, 17], 10), [165, 36, 212, 193, 237, 54, 199, 135, 44, 85]);
});
Deno.test("qr: a room link encodes to the matrix jsQR decoded (fingerprint)", async () => {
  const m = qrMatrix("https://swarmllm.ai/r/ABCD");
  eq(m.length, 25, "version 2");
  const bytes = new TextEncoder().encode(m.map((r) => r.join("")).join("\n"));
  const h = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 16);
  eq(h, "a687d8c6cc00e73f");
  // finder patterns in three corners
  for (const [r, c] of [[0, 0], [0, 18], [18, 0]]) ok(m[r][c] && m[r + 6][c + 6] && m[r + 3][c + 3] && !m[r + 1][c + 1], "finder at " + r + "," + c);
  ok(qrSVG("x").startsWith("<svg"), "svg");
});
Deno.test("transport: header flags round trip", () => {
  for (const f of [{}, { spec: 1 }, { reset: 1 }, { rb: 0 }, { rb: 7, spec: 1 }, { reset: 1, rb: 62 }]) {
    const back = unpackFlags(packFlags(f));
    eq(!!back.spec, !!f.spec); eq(!!back.reset, !!f.reset); eq(back.rb, f.rb);
  }
});

import { verdict, deviceKind } from "../../room/preflight.js";
const UA = {
  mac: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
  ipad: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
  iphone: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1",
  firefox: "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0",
  linux: "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
};
Deno.test("preflight: device kinds (an iPad is not a Mac)", () => {
  eq(deviceKind({ ua: UA.ipad, touchPoints: 5 }), "iPad");
  eq(deviceKind({ ua: UA.mac, touchPoints: 0 }), "Mac");
  eq(deviceKind({ ua: UA.iphone }), "iPhone");
});
Deno.test("preflight: every browser without WebGPU gets a specific remedy", () => {
  const no = (ua, extra = {}) => verdict({ ua, secure: true, hasGpuApi: false, adapter: null, ...extra });
  ok(/Chrome or Edge/.test(no(UA.firefox).line), "firefox");
  ok(/enable-unsafe-webgpu/.test(no(UA.linux).line), "linux chrome flag");
  ok(/Safari 26/.test(no(UA.iphone).line), "ios update");
  ok(/https/.test(no(UA.mac, { secure: false }).line), "insecure page");
  ok(/blocklisted/.test(no(UA.mac, { hasGpuApi: true }).line), "api but no adapter");
  const yes = verdict({ ua: UA.mac, secure: true, hasGpuApi: true, adapter: { vendor: "apple", architecture: "metal-3" } });
  ok(yes.ok && /apple metal-3/.test(yes.line), "ok line names the GPU");
  ok(!no(UA.firefox).ok && /ask questions/.test(no(UA.firefox).line), "guests are told they can still ask");
});
Deno.test("conversation: an open assistant turn (Continue) extends the caches without an end token", () => {
  const t1 = [{ role: "user", text: "a" }];
  const answer = [77, 78];
  const closed = buildIds(tok, { turns: [...t1, { role: "assistant", ids: answer }] });
  const open = buildIds(tok, { turns: [...t1, { role: "assistant", ids: answer, open: true }] });
  eq(open, closed.slice(0, open.length), "open is the closed turn minus <|im_end|>\\n");
  eq(closed.length - open.length, 2);
  // what a capped speculative answer leaves in the caches (all but the last emitted token) is a prefix
  eq(reusablePrefix(open.slice(0, -1), open), open.length - 1);
});
