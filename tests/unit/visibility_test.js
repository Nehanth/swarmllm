import { chatRecipients, VISIBILITY } from "../../room/visibility.js";
const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };
const ids = ["p1", "p2", "p3"];

Deno.test("visibility: all — everyone gets the text", () => {
  eq(chatRecipients("all", "p2", ids), { full: ["p1", "p2", "p3"], hidden: [] });
});
Deno.test("visibility: host — nobody but the host's own screen", () => {
  eq(chatRecipients("host", "p2", ids), { full: [], hidden: ["p1", "p2", "p3"] });
});
Deno.test("visibility: asker — the asking peer by id, the rest hidden", () => {
  eq(chatRecipients("asker", "p2", ids), { full: ["p2"], hidden: ["p1", "p3"] });
});
Deno.test("visibility: asker is the host itself — nobody else sees it", () => {
  eq(chatRecipients("asker", "host-id", ids), { full: [], hidden: ["p1", "p2", "p3"] });
});
Deno.test("visibility: modes are the three the dropdown offers", () => { eq(VISIBILITY, ["all", "host", "asker"]); });
