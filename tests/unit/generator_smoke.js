// Generates the full WGSL for a few shapes and checks it is non-trivial and balanced.
import { WGSL, coopWGSL } from "../../engine/engine.js";
for (const [wg, rows, cols, rowsB] of [[256, 4, 4, 4], [128, 4, 8, 2], [64, 4, 8, 4]]) {
  const src = WGSL + coopWGSL(wg, rows, 64, cols, rowsB, true);
  const open = (src.match(/\{/g) || []).length, close = (src.match(/\}/g) || []).length;
  if (open !== close) { console.error(`unbalanced braces for ${wg}/${rows}/${cols}/${rowsB}`); Deno.exit(1); }
  console.log(`ok ${wg}/${rows}/${cols}/${rowsB}: ${src.length} chars, ${(src.match(/@compute/g) || []).length} entry points`);
}
