// harness/prefix.js: longest reusable checkpoint.
import { PrefixIndex } from "../../harness/prefix.js";
const eq = (a, b, m) => { const ja = JSON.stringify(a), jb = JSON.stringify(b); if (ja !== jb) throw new Error((m || "mismatch") + ": " + ja + " != " + jb); };

Deno.test("PrefixIndex picks the longest strict prefix", () => {
  const P = new PrefixIndex();
  P.add([1, 2, 3], "sys");
  P.add([1, 2, 3, 4, 5, 6], "turn1");
  P.add([1, 2, 9, 9], "other");
  eq(P.best([1, 2, 3, 4, 5, 6, 7, 8]), { key: "turn1", n: 6 });
  eq(P.best([1, 2, 3, 4, 5]), { key: "sys", n: 3 }, "turn1 is longer than the prompt");
  eq(P.best([1, 2, 3, 4, 5, 6]), { key: "sys", n: 3 }, "an exact match leaves nothing to run: not usable");
  eq(P.best([7, 1, 2, 3]), null);
  P.remove("turn1");
  eq(P.best([1, 2, 3, 4, 5, 6, 7]), { key: "sys", n: 3 });
});

Deno.test("PrefixIndex keeps the most recently used entries", () => {
  const P = new PrefixIndex(2);
  P.add([1], "a"); P.add([1, 2], "b"); P.add([1, 2, 3], "c");
  eq(P.items.length, 2);
  P.add([1, 2], "b");
  eq(P.items.map((x) => x.key).sort(), ["b", "c"]);
});
