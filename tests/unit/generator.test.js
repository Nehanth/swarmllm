// No-GPU tests for the WGSL generator and helpers.
import { coopWGSL } from "../../engine/engine.js";
import { f32ToF16 } from "../../engine/gguf.js";
import { assert, assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";

Deno.test("coopWGSL emits every kernel family for the default shape", () => {
  const src = coopWGSL(256, 4, 64, 4, 4, true);
  for (const name of ["matvec_coop", "matvec_q8_coop", "matvec_q4_coop", "matvec_q4_coop_acc",
    "matvec_q4_coop_b", "matvec_q4_coop_b_acc", "matvec_q4_gu", "matvec_q4_gu_b"]) {
    assert(src.includes(`fn ${name}(`), `missing ${name}`);
  }
  assert(!src.includes("_b4("), "4-column twins only exist when COLS > 4");
});

Deno.test("coopWGSL emits 4-column twins when COLS > 4", () => {
  const src = coopWGSL(256, 4, 64, 8, 2, true);
  for (const name of ["matvec_q4_coop_b4", "matvec_q4_coop_b4_acc", "matvec_q4_gu_b4"]) assert(src.includes(`fn ${name}(`), `missing ${name}`);
});

Deno.test("dequant snippets follow the UNPACK switch", () => {
  const withUnpack = coopWGSL(256, 4, 64, 4, 4, true), without = coopWGSL(256, 4, 64, 4, 4, false);
  assert(withUnpack.includes("unpack4xU8(") && !without.includes("unpack4xU8("));
  assert(without.includes(">> 24u)"));
});

Deno.test("no dynamically indexed local accumulators in generated kernels", () => {
  const src = coopWGSL(256, 4, 64, 8, 2, true);
  assert(!/var acc: array<f32/.test(src), "accumulator arrays spill to scratch memory");
});

Deno.test("f32ToF16 round-trips simple values", () => {
  assertEquals(f32ToF16(1.0), 0x3c00);
  assertEquals(f32ToF16(-2.0), 0xc000);
  assertEquals(f32ToF16(0.0), 0);
});
