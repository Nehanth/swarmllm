# Kernel research round 3 (Sep 1 night, 61 agents)

# Spec: feature-gated subgroup reduction for the coop Q4_0/Q8_0 GEMV family

Status: implementation-ready. Everything below is derived from the verified findings and from the current generator (`engine/engine.js` `coopWGSL` L411-826, `autotuneCoop` L1460-1511, pipeline table L900-921; `engine/qwen35.js` L501, L610-624; `p2p.html` L1052, L1078, L1141). Items marked UNCERTAIN are not backed by a verified finding and must be checked before relying on them.

## 0. Expectations (so nobody builds this for the wrong reason)

Measured on the GB10 (Deno 2.9.5 / wgpu / Vulkan, subgroup size 32), same sweep code, only the reduction changed:

| shape | tree (coop WG256 R4) | best subgroup variant |
|---|---|---|
| 17408x5120 (gate/up) | 0.350 ms, 143 GB/s | x1.03-1.04 (two-stage A, WG256 R4) |
| 5120x5120 (attn proj) | 0.087 ms, 166-169 GB/s | x1.08 (two-stage A, WG128 R4) |
| 248320x5120 (LM head, Q4 stand-in) | 3.39-3.45 ms, 207-211 GB/s | x1.02 (A, WG256 R4); do not cite variant B here (re-run: x0.54) |
| 512x5120 / 1024x5120 (DeltaNet/KV size) | 0.051 ms = launch floor | x1.01 / x1.05, variant B x0.77-0.87 |

Numerics: max abs diff vs tree 1.8e-7..4.8e-7 (f32 accumulate, reordering only). GB10 DRAM peak is 273 GB/s and the LM head already runs at ~210 GB/s, so a 2-3x gain is physically impossible; the shared-memory tree is not the bottleneck. Ship this as an autotune candidate worth 0-8%, with the tree kept as the canonical Safari/fallback path. Bigger levers (dispatch fusion against the ~0.05 ms x 640 floor, bytes in flight per thread) are out of scope here.

## 1. Runtime gate (browser and Deno)

Facts the gate must accommodate:

- Chrome/Dawn (134+): `requestDevice({requiredFeatures:['subgroups']})`, shader must start with `enable subgroups;`.
- Deno 2.9.5 / wgpu on the GB10: `adapter.features` does NOT list `subgroups` (Deno masks with `all_webgpu_mask()`), but `requestDevice` with it SUCCEEDS and `device.features.has('subgroups')` is true. naga REJECTS `enable subgroups;` ("enable-extension is not yet supported", wgpu#5555) but compiles the builtins without the directive. A future naga may flip to requiring the directive (wgpu#8202, open).
- Safari 26.0-26.4: no `subgroups` feature; `requestDevice` with an unknown feature name throws. (Safari TP 249 added subgroups, so never hard-code a Chromium check.)
- f16 subgroup ops need both `shader-f16` and `subgroups`; not needed here (f32 accumulation is mandatory).

```js
// engine/gpu-features.js (new) -----------------------------------------------
// Do NOT trust adapter.features for 'subgroups' (Deno masks it). Try, then fall back.
export async function requestDeviceWithSubgroups(adapter, desc = {}) {
  const base = desc.requiredFeatures || [];
  try {
    const d = await adapter.requestDevice({ ...desc, requiredFeatures: [...base, "subgroups"] });
    return d;
  } catch {
    return adapter.requestDevice({ ...desc, requiredFeatures: base });
  }
}

// Returns the module prefix that makes subgroup builtins compile on THIS runtime:
//   'enable subgroups;\n'  Chrome/Dawn (and any future naga that requires it)
//   ''                     Deno/wgpu today (naga rejects the directive)
//   null                   no subgroups (Safari 26.x, old drivers) -> tree only
export async function subgroupPrefix(device) {
  if (!device.features.has("subgroups")) return null;
  const body = `@compute @workgroup_size(32)
fn p(@builtin(subgroup_size) s: u32, @builtin(subgroup_invocation_id) l: u32) {
  let x = subgroupAdd(f32(s) + f32(l));
}`;
  for (const pre of ["enable subgroups;\n", ""]) {
    device.pushErrorScope("validation");
    const m = device.createShaderModule({ code: pre + body });
    const info = await m.getCompilationInfo();
    const err = await device.popErrorScope();
    if (!err && !info.messages.some((x) => x.type === "error")) return pre;
  }
  return null;
}

// Optional: gate for the barrier-free rows-per-subgroup layout (section 4.4).
// Only sound when the subgroup width is fixed for this adapter (NVIDIA, Apple).
export function fixedSubgroupSize(adapter) {
  const i = adapter.info;
  if (!i || !i.subgroupMinSize || i.subgroupMinSize !== i.subgroupMaxSize) return 0;
  return i.subgroupMinSize;
}
```

Notes:
- `pushErrorScope` around `createShaderModule` is belt-and-braces; the verified probe used `getCompilationInfo()` alone. UNCERTAIN: whether Deno's `getCompilationInfo()` reports naga's directive error as `type === "error"` versus surfacing it only via the error scope; the double check covers both.
- The prefix MUST be the first text of the whole module (before `WGSL`, `coopWGSL(...)`, `WGSL2`): enable directives precede all declarations. It goes on the concatenation, not on `coopWGSL`'s output.
- `adapter.info.subgroupMinSize/MaxSize` was observed on Deno (reported 32/32) and is spec'd on Chrome. UNCERTAIN on Safari once it ships subgroups.

Call sites to change:
- `p2p.html:1052` (`ai.device = await adapter.requestDevice({requiredLimits: ...})`) -> `requestDeviceWithSubgroups(adapter, {requiredLimits: ...})`, then `ai.sgPrefix = await subgroupPrefix(ai.device)`.
- `p2p.html:1068` throwaway test device: same wrapper if the micro/self tests should exercise the sg kernels (recommended, section 6).
- Every Deno script that does `adapter.requestDevice({requiredLimits: ...})` (bench_deno.js:30, bench_single_deno.js:8, test_q38_deno.js:30, test_mtp_deno.js:15, bench_breakdown_deno.js:7, ...): same wrapper. A shared `engine/deno-device.js` helper is the least churn.

## 2. WGSL design rules for every subgroup entry point

1. Every `subgroupAdd` is called from ALL invocations at top level of the entry point (after the sweep loop, never inside `if (full)`, never inside the row-tail guards, never inside `if (t < N)`). Chrome 134-144 uniformity analysis is workgroup-scoped; Deno/naga and Chrome 145+ are more relaxed, but the workgroup-uniform form compiles everywhere. Gate only the STORE.
2. No `subgroup_id` / `num_subgroups` builtins (separate `requires subgroup_id;` extension, Chrome 144+, absent on Safari). No `subgroupBroadcast` with runtime lane ids. No `subgroupElect` (missing in naga).
3. Size-agnostic: rely only on `subgroupMinSize <= subgroup_size <= subgroupMaxSize` and on `subgroup_size` being workgroup-uniform in compute. Never bake 32 into the two-stage kernel.
4. Mapping-free partials: the WGSL spec defines no relationship between `local_invocation_index` and `subgroup_invocation_id`/`subgroup_id`, and the WebGPU spec sets no lower bound on `subgroupMinSize`. So the kernels below do NOT index partials by `t / sg_size`. Instead every thread writes its own slot `part[r*WG + t]` with `select(0.0, sum, sg_lane == 0u)`, and stage 2 sums the whole WG-wide row. This reuses the existing `mvc_part` array (no new workgroup memory) and costs `WG / sg_size` extra loads per lane (8 at WG=256/SG=32, 64 at SG=4). Remaining assumption: every subgroup contains an invocation with `subgroup_invocation_id == 0` (true for hardware lanes on Vulkan/Metal; UNCERTAIN for Chrome's D3D subgroup emulation, where "subgroups may not be full"). The load-time diff check in section 6 catches a violation and falls back to the tree. If it ever fails, the fix is `let leader = subgroupBroadcastFirst(t); ... select(0.0, s, t == leader)` (UNCERTAIN whether naga implements `subgroupBroadcastFirst`; not probed).
5. f32 only. `subgroupAdd(f32)` reorders the sum; measured diff <= 5e-7 vs tree.
6. `workgroupBarrier()` stays outside all conditionals (existing engine rule).
7. Dispatch geometry, bindings, uniforms and buffer layouts are identical to the tree kernels: `ceil(dOut / ROWS)` workgroups, group(1) bindings `[qs, sc, x, y, shape]`. Correctness never depends on subgroups being present.

## 3. Generator changes (`coopWGSL`)

Signature becomes:

```js
export function coopWGSL(WG = 256, ROWS = 4, WGB = 64, COLS = 4, ROWSB = ROWS, opts = {}) {
  const { sg = false, sgRow = 0 } = opts;   // sg: emit two-stage *_sg entry points; sgRow: SG width for _sgrow (0 = off)
  ...
```

Add this helper inside `coopWGSL` next to `reduce`. It emits a two-stage reduction of `N` named scalar accumulators into `N` named finals, using a WG-wide partial array `arr` (must hold at least `N * WGT` floats):

```js
  // Two-stage subgroup reduction, size-agnostic and mapping-free (section 2).
  //   accs[i] : name of the per-thread f32 accumulator i
  //   outs[i] : name to bind the workgroup-wide total i to (declared by this snippet)
  //   arr     : var<workgroup> array<f32, >= accs.length * WGT>
  //   WGT     : workgroup size of the calling entry point
  // Assumes `t`, `sg_size`, `sg_lane` are in scope. Every subgroupAdd is at top level.
  const sgReduce = (accs, outs, arr, WGT) => `
  // stage 1: in-register subgroup sums, written by lane 0 of each subgroup; other lanes write 0
${accs.map((a, i) => `  { let s = subgroupAdd(${a}); ${arr}[${i * WGT}u + t] = select(0.0, s, sg_lane == 0u); }`).join("\n")}
  workgroupBarrier();
  // stage 2: every subgroup redundantly sums the WG-wide row (only WG/sg_size entries are non-zero),
  // so the second subgroupAdd is also in workgroup-uniform control flow (Chrome 134-144 analysis).
${accs.map((_, i) => `  var p${i} = 0.0;`).join("\n")}
  for (var k: u32 = sg_lane; k < ${WGT}u; k += sg_size) {
${accs.map((_, i) => `    p${i} += ${arr}[${i * WGT}u + k];`).join("\n")}
  }
${outs.map((o, i) => `  let ${o} = subgroupAdd(p${i});`).join("\n")}`;
```

The `for` loop has a lane-dependent trip count but contains no subgroup builtin; the `subgroupAdd(p_i)` after it is reached by all invocations. This is the same structure llama.cpp's `mul_mat_vec.wgsl` stage 2 uses under Dawn (`for (k = subgroup_invocation_id; k < num_subgroups; k += subgroup_size)` then `subgroupAdd`), and the equivalent loop compiled on naga in `verify_claim_sg.js`.

### 3.1 Complete single-column Q4_0 and Q8_0 entry points

Appended to the returned template string when `sg` is true. They reference `q4_row`, `q8_row`, `q4s`, `q8s`, `q4_x4`, `q8_x4`, `mvc_part` already declared in the same module, so the sweep is bit-for-bit the existing one. Shown fully expanded for WG=256, ROWS=4 (the generator produces them from `WG`/`ROWS` exactly like the tree kernels; the generator form follows the listing).

```wgsl
// ===== matvec_q4_coop_sg : Q4_0 GEMV, coop sweep + two-stage subgroup reduction =====
// bindings: group(1) 0 q4_qs: array<u32>, 1 q4_sc: array<u32> (f16 pairs), 2 q4_x4: array<vec4<f32>>,
//           3 q4_y: array<f32>, 4 q4_shape: Shape {dOut, dIn}     (identical to matvec_q4_coop)
// dispatch: ceil(dOut / 4) workgroups
@compute @workgroup_size(256)
fn matvec_q4_coop_sg(@builtin(workgroup_id) wg: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>,
                     @builtin(subgroup_size) sg_size: u32,            // workgroup-uniform in compute
                     @builtin(subgroup_invocation_id) sg_lane: u32) {
  let t = lid.x;
  let qt = t & 3u;                 // which u32 word of the 32-weight block this thread owns
  let bl = t >> 2u;                // block lane: 64 lanes sweep the row's blocks
  let dIn = q4_shape.dIn;
  let nb = dIn / 32u;
  let rowWords = dIn / 8u;
  let row0 = wg.x * 4u;
  let full = row0 + 3u < q4_shape.dOut;    // uniform: workgroup_id + uniform buffer
  var acc0 = 0.0; var acc1 = 0.0; var acc2 = 0.0; var acc3 = 0.0;   // scalars: dynamic local arrays spill 3x
  for (var b: u32 = bl; b < nb; b += 64u) {
    let xlo = q4_x4[b * 8u + qt];          // x[j..j+3]   (low nibbles)
    let xhi = q4_x4[b * 8u + qt + 4u];     // x[j+16..j+19] (high nibbles)
    let wIdx = row0 * rowWords + b * 4u + qt;   // consecutive threads -> consecutive words
    let scBase = row0 * nb + b;
    if (full) {
      acc0 += q4_row(wIdx,                 q4s(scBase),           xlo, xhi);
      acc1 += q4_row(wIdx + rowWords,      q4s(scBase + nb),      xlo, xhi);
      acc2 += q4_row(wIdx + 2u * rowWords, q4s(scBase + 2u * nb), xlo, xhi);
      acc3 += q4_row(wIdx + 3u * rowWords, q4s(scBase + 3u * nb), xlo, xhi);
    } else {
      if (row0      < q4_shape.dOut) { acc0 += q4_row(wIdx,                 q4s(scBase),           xlo, xhi); }
      if (row0 + 1u < q4_shape.dOut) { acc1 += q4_row(wIdx + rowWords,      q4s(scBase + nb),      xlo, xhi); }
      if (row0 + 2u < q4_shape.dOut) { acc2 += q4_row(wIdx + 2u * rowWords, q4s(scBase + 2u * nb), xlo, xhi); }
    }
  }
  // ---- stage 1: subgroup sums. Called by EVERY invocation; only the write is lane-gated (via select). ----
  { let s = subgroupAdd(acc0); mvc_part[0u   + t] = select(0.0, s, sg_lane == 0u); }
  { let s = subgroupAdd(acc1); mvc_part[256u + t] = select(0.0, s, sg_lane == 0u); }
  { let s = subgroupAdd(acc2); mvc_part[512u + t] = select(0.0, s, sg_lane == 0u); }
  { let s = subgroupAdd(acc3); mvc_part[768u + t] = select(0.0, s, sg_lane == 0u); }
  workgroupBarrier();                      // the only barrier (tree: 9)
  // ---- stage 2: sum the WG-wide partial rows; every subgroup does it redundantly (uniform control flow). ----
  var p0 = 0.0; var p1 = 0.0; var p2 = 0.0; var p3 = 0.0;
  for (var k: u32 = sg_lane; k < 256u; k += sg_size) {
    p0 += mvc_part[k]; p1 += mvc_part[256u + k]; p2 += mvc_part[512u + k]; p3 += mvc_part[768u + k];
  }
  let f0 = subgroupAdd(p0); let f1 = subgroupAdd(p1); let f2 = subgroupAdd(p2); let f3 = subgroupAdd(p3);
  if (t == 0u) {
    q4_y[row0] = f0;
    if (row0 + 1u < q4_shape.dOut) { q4_y[row0 + 1u] = f1; }
    if (row0 + 2u < q4_shape.dOut) { q4_y[row0 + 2u] = f2; }
    if (row0 + 3u < q4_shape.dOut) { q4_y[row0 + 3u] = f3; }
  }
}

// ===== matvec_q8_coop_sg : Q8_0 GEMV (int8 packed 4/u32, f16 scale pairs), same reduction =====
// bindings identical to matvec_q8_coop: 0 q8_qs, 1 q8_sc, 2 q8_x4: array<vec4<f32>>, 3 q8_y, 4 q8_shape
@compute @workgroup_size(256)
fn matvec_q8_coop_sg(@builtin(workgroup_id) wg: vec3<u32>,
                     @builtin(local_invocation_id) lid: vec3<u32>,
                     @builtin(subgroup_size) sg_size: u32,
                     @builtin(subgroup_invocation_id) sg_lane: u32) {
  let t = lid.x;
  let qt = t & 3u;
  let bl = t >> 2u;
  let dIn = q8_shape.dIn;
  let nb = dIn / 32u;
  let rowWords = dIn / 4u;
  let row0 = wg.x * 4u;
  let full = row0 + 3u < q8_shape.dOut;
  var acc0 = 0.0; var acc1 = 0.0; var acc2 = 0.0; var acc3 = 0.0;
  for (var b: u32 = bl; b < nb; b += 64u) {
    let x4 = b * 8u + qt * 2u;
    let xa = q8_x4[x4];
    let xb = q8_x4[x4 + 1u];
    let wBase = row0 * rowWords + b * 8u + qt * 2u;   // 2 words (8 int8) per thread per block
    let scBase = row0 * nb + b;
    if (full) {
      acc0 += q8_row(wBase,                 q8s(scBase),           xa, xb);
      acc1 += q8_row(wBase + rowWords,      q8s(scBase + nb),      xa, xb);
      acc2 += q8_row(wBase + 2u * rowWords, q8s(scBase + 2u * nb), xa, xb);
      acc3 += q8_row(wBase + 3u * rowWords, q8s(scBase + 3u * nb), xa, xb);
    } else {
      if (row0      < q8_shape.dOut) { acc0 += q8_row(wBase,                 q8s(scBase),           xa, xb); }
      if (row0 + 1u < q8_shape.dOut) { acc1 += q8_row(wBase + rowWords,      q8s(scBase + nb),      xa, xb); }
      if (row0 + 2u < q8_shape.dOut) { acc2 += q8_row(wBase + 2u * rowWords, q8s(scBase + 2u * nb), xa, xb); }
    }
  }
  { let s = subgroupAdd(acc0); mvc_part[0u   + t] = select(0.0, s, sg_lane == 0u); }
  { let s = subgroupAdd(acc1); mvc_part[256u + t] = select(0.0, s, sg_lane == 0u); }
  { let s = subgroupAdd(acc2); mvc_part[512u + t] = select(0.0, s, sg_lane == 0u); }
  { let s = subgroupAdd(acc3); mvc_part[768u + t] = select(0.0, s, sg_lane == 0u); }
  workgroupBarrier();
  var p0 = 0.0; var p1 = 0.0; var p2 = 0.0; var p3 = 0.0;
  for (var k: u32 = sg_lane; k < 256u; k += sg_size) {
    p0 += mvc_part[k]; p1 += mvc_part[256u + k]; p2 += mvc_part[512u + k]; p3 += mvc_part[768u + k];
  }
  let f0 = subgroupAdd(p0); let f1 = subgroupAdd(p1); let f2 = subgroupAdd(p2); let f3 = subgroupAdd(p3);
  if (t == 0u) {
    q8_y[row0] = f0;
    if (row0 + 1u < q8_shape.dOut) { q8_y[row0 + 1u] = f1; }
    if (row0 + 2u < q8_shape.dOut) { q8_y[row0 + 2u] = f2; }
    if (row0 + 3u < q8_shape.dOut) { q8_y[row0 + 3u] = f3; }
  }
}
```

Generator form (replaces the hand-expanded text above; place after the `matvec_q4_coop` template, guarded by `${sg ? ... : ""}`):

```js
  const sgArgs = `@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_size) sg_size: u32, @builtin(subgroup_invocation_id) sg_lane: u32`;
  const accNames = Array.from({ length: ROWS }, (_, r) => `acc${r}`);
  const finNames = Array.from({ length: ROWS }, (_, r) => `f${r}`);
  const sgStore = (ybuf, shp) => `
  if (t == 0u) {
    ${ybuf}[row0] = f0;
${Array.from({ length: ROWS - 1 }, (_, r) => `    if (row0 + ${r + 1}u < ${shp}.dOut) { ${ybuf}[row0 + ${r + 1}u] = f${r + 1}; }`).join("\n")}
  }`;
  const sgSingle = (kind) => {   // kind: "q4" | "q8"
    const shp = `${kind}_shape`, ybuf = `${kind}_y`;
    const loads = kind === "q4" ? `
    let xlo = q4_x4[b * 8u + qt];
    let xhi = q4_x4[b * 8u + qt + 4u];
    let wIdx = row0 * rowWords + b * 4u + qt;
    let scBase = row0 * nb + b;` : `
    let x4 = b * 8u + qt * 2u;
    let xa = q8_x4[x4];
    let xb = q8_x4[x4 + 1u];
    let wBase = row0 * rowWords + b * 8u + qt * 2u;
    let scBase = row0 * nb + b;`;
    const term = kind === "q4"
      ? (r) => `q4_row(wIdx + ${r}u * rowWords, q4s(scBase + ${r}u * nb), xlo, xhi)`
      : (r) => `q8_row(wBase + ${r}u * rowWords, q8s(scBase + ${r}u * nb), xa, xb)`;
    return `
@compute @workgroup_size(${WG})
fn matvec_${kind}_coop_sg(${sgArgs}) {
  let t = lid.x;
  let qt = t & 3u;
  let bl = t >> 2u;
  let dIn = ${shp}.dIn;
  let nb = dIn / 32u;
  let rowWords = ${kind === "q4" ? "dIn / 8u" : "dIn / 4u"};
  let row0 = wg.x * ${ROWS}u;
  let full = row0 + ${ROWS - 1}u < ${shp}.dOut;
  ${accDecl}
  for (var b: u32 = bl; b < nb; b += ${LANES}u) {${loads}
    if (full) {
${fullBody(term)}
    } else {
${tailBody(term, `${shp}.dOut`)}
    }
  }
${sgReduce(accNames, finNames, "mvc_part", WG)}
${sgStore(ybuf, shp)}
}`;
  };
```

`mvc_part` is `array<f32, WG * ROWS>` (L628), exactly the `ROWS * WG` floats `sgReduce` needs. The f32 `matvec_coop_sg` is optional (f32 weights are not on the hot path); if generated, use `mvf_row` and `mv_shape` the same way.

### 3.2 Batched `_coop_b_sg` (verify/prefill, COLS=4, ROWSB rows, WGB threads)

Today (L804-821) the batched kernel does `COLS/2` rounds, each a 2*ROWSB-wide tree over `mvb_part` (2*ROWSB*WGB floats) = 2 x (log2(WGB)+1) = 14 barriers at WGB=64. The subgroup version keeps the two-columns-per-round structure and `mvb_part` (no new workgroup memory), 2 barriers per round:

```js
  // inside the ["", "_q8", "_q4"].map((kind) => ...) block, emitted in addition to the tree kernel when sg:
  const sgBatched = () => {
    const rounds = Array.from({ length: COLS / 2 }, (_, h) => {
      const accs = [], stores = [];
      for (const c of [2 * h, 2 * h + 1]) for (const r of rows) {
        accs.push(`a${r}_${c}`);
        stores.push(`    if (row0 + ${r}u < dOut) { ${ybuf}[${c}u * ys + row0 + ${r}u] = f_${r}_${c}; }`);
      }
      const outs = accs.map((a) => "f_" + a.slice(1));
      return `${sgReduce(accs, outs, "mvb_part", WGB)}
  if (t == 0u) {
${stores.join("\n")}
  }
  workgroupBarrier();   // mvb_part is reused by the next round`;
    });
    return `
@compute @workgroup_size(${WGB})
fn matvec${kind}_coop_b_sg(${sgArgs}) {
  let t = lid.x;
  let qt = t & 3u;
  let bl = t >> 2u;
  let dIn = ${shp}.dIn;
  let dOut = ${shp}.dOut;
  let xs4 = ${shp}.xs4;
  let ys = ${shp}.ys;
  let nb = dIn / 32u;
  let rowWords = ${kind === "_q4" ? "dIn / 8u" : "dIn / 4u"};
  let row0 = wg.x * ${ROWSB}u;
  let full = row0 + ${ROWSB - 1}u < dOut;
  ${rows.map((r) => cols.map((m) => `var a${r}_${m} = 0.0;`).join(" ")).join("\n  ")}
  for (var b: u32 = bl; b < nb; b += ${LANESB}u) {
    ${xLoads}
    ${idx}
    ${rows.map(rowBlock).join("\n    ")}
  }
${rounds.join("\n")}
}`;
  };
```

`xLoads`, `idx`, `rowBlock`, `cols`, `rows`, `shp`, `xbuf`, `ybuf` are the existing locals of that block (L757-783), so the sweep is unchanged. The trailing barrier after the last round is redundant but harmless; drop it for the final round if desired.

### 3.3 Fused gate/up `_gu_sg` and `_gu_b_sg`

Single-column gu (L489-521) runs two full tree rounds (`ag*` then `au*`) exchanging through `gu_res`: 2 x 9 = 18 barriers. Subgroup version: reduce both sets in one shot using a 2*ROWS*WG partial array and one barrier. Declare once in `guAll`:

```js
var<workgroup> gusg_part: array<f32, ${2 * ROWS * WG}>;    // 8 KB at WG=256, ROWS=4 (per-entry-point accounting)
```

and in `guKernel(kind, batched, sg)` replace the two `reduceTo(...)` calls of `colBody(m)` with:

```js
    const gAcc = Array.from({ length: ROWS }, (_, r) => `ag${r}`), uAcc = Array.from({ length: ROWS }, (_, r) => `au${r}`);
    const gFin = Array.from({ length: ROWS }, (_, r) => `fg${r}`), uFin = Array.from({ length: ROWS }, (_, r) => `fu${r}`);
    const sgEpilogue = `
${sgReduce([...gAcc, ...uAcc], [...gFin, ...uFin], "gusg_part", WG)}
  if (t == 0u) {
${Array.from({ length: ROWS }, (_, r) => `    if (row0 + ${r}u < dOut) { let g = fg${r}; ${P}_y[${batched ? `${m}u * ${shape}.ys + ` : ""}row0 + ${r}u] = (g / (1.0 + exp(-g))) * fu${r}; }`).join("\n")}
  }
  workgroupBarrier();`;
```

(entry point name `matvec_${kind}_gu_sg` / `..._gu_b_sg`, signature `sgArgs`, `let xcol` block unchanged). For the batched `guKernelB` (L601-619: one tree round per column, 4 x 7 = 28 barriers at WGB=64), reduce per column with `sgReduce([...ag{r}_{m} for r], [...au{r}_{m} for r] , ..., "gub_part", WGB)` — `gub_part` is `2*ROWSB*WGB` floats which is exactly `2*ROWSB` accumulators per column — followed by the `t == 0u` SiLU store and a barrier before the next column: 2 barriers per column, 8 total.

### 3.4 Optional: barrier-free rows-per-subgroup `matvec_q4_coop_sgrow` (SG baked in)

Only emitted when `opts.sgRow > 0`, which the caller sets from `fixedSubgroupSize(adapter)` (subgroupMinSize == subgroupMaxSize, true on NVIDIA/Apple, false on many Intel/Qualcomm). On AMD wave64 or Intel with a floating width this layout silently drops rows, which is why it is gated, and why the dispatch count depends on `sgRow`. Measured: x1.01-1.10 at 5120x5120 depending on run, x1.04 at 2048x5120, but x0.77-0.87 at 512/1024 rows and unstable on the LM head (x1.04 vs x0.54 across runs). Autotune candidate only; not the default.

```wgsl
// SG substituted by JS (e.g. 32). rows per WG = WG / SG. dispatch: ceil(dOut / (WG / SG)).
@compute @workgroup_size(${WG})
fn matvec_q4_coop_sgrow(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
                        @builtin(subgroup_invocation_id) sg_lane: u32) {
  let t = lid.x;
  let sg = t / ${SG}u;              // which subgroup (valid only because SG is the fixed hardware width)
  let l = t % ${SG}u;
  let qt = l & 3u;
  let bl = l >> 2u;                 // SG/4 block lanes per row
  let dIn = q4_shape.dIn;
  let nb = dIn / 32u;
  let rowWords = dIn / 8u;
  let row = wg.x * ${WG / SG}u + sg;
  let inb = row < q4_shape.dOut;    // rows past dOut do zero-work loads of row dOut-1 (in-bounds), store is gated
  let rr = select(q4_shape.dOut - 1u, row, inb);
  var acc = 0.0;
  for (var b: u32 = bl; b < nb; b += ${SG / 4}u) {
    let xlo = q4_x4[b * 8u + qt];
    let xhi = q4_x4[b * 8u + qt + 4u];
    acc += q4_row(rr * rowWords + b * 4u + qt, q4s(rr * nb + b), xlo, xhi);
  }
  let s = subgroupAdd(acc);         // uniform: every invocation executes the loop and this call
  if (sg_lane == 0u && inb) { q4_y[row] = s; }
}
```

Note `t / SG == subgroup index` is a shipping-implementation convention (Chrome 144 blog: "should be safe"), not a spec guarantee; the load-time diff check in section 6 is the safety net. This is the one place the assumption is made, and it is confined to a gated, optional candidate. Q8 variant: same with `q8_row`/`wBase = rr*rowWords + b*8u + qt*2u`.

## 4. Autotuner changes (`autotuneCoop`)

Candidates become `(wg, rows, variant)` with `variant in {"tree", "sg", "sgrow"}`; `sg` only when `sgPrefix !== null`, `sgrow` only when `sgRow > 0`. The same 3% noise guard applies, and the tree default is preferred unless a candidate wins by more than 3%. Every subgroup candidate is also checked numerically against the tree on random data before it is allowed to win.

```js
export async function autotuneCoop(device, { dIn = 5120, dOut = 17408, kind = "q4", sgPrefix = null, sgRow = 0 } = {}) {
  const shapes = [[256, 4], [128, 4], [256, 8], [128, 8], [64, 4]];
  const variants = ["tree"];
  if (sgPrefix !== null) variants.push("sg");
  if (sgPrefix !== null && sgRow > 0) variants.push("sgrow");
  const nb = dIn / 32;
  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  // random-ish data so the sg-vs-tree diff check is meaningful (uninitialized buffers are all zero)
  let seed = 12345; const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const qsBytes = dOut * (kind === "q4" ? dIn / 2 : dIn);
  const qsData = new Uint8Array(qsBytes); for (let i = 0; i < qsBytes; i++) qsData[i] = (rnd() * 256) | 0;
  const scData = new Uint16Array(Math.ceil(dOut * nb / 2) * 2); for (let i = 0; i < scData.length; i++) scData[i] = f32ToF16(0.01 + rnd() * 0.02);
  const xData = new Float32Array(dIn); for (let i = 0; i < dIn; i++) xData[i] = rnd() - 0.5;
  const qs = device.createBuffer({ size: qsBytes, usage: S });
  const sc = device.createBuffer({ size: scData.byteLength, usage: S });
  const x = device.createBuffer({ size: dIn * 4, usage: S });
  const y = device.createBuffer({ size: dOut * 4, usage: S | GPUBufferUsage.COPY_SRC });
  const stage = device.createBuffer({ size: dOut * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  device.queue.writeBuffer(qs, 0, qsData); device.queue.writeBuffer(sc, 0, scData); device.queue.writeBuffer(x, 0, xData);
  const shape = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(shape, 0, new Uint32Array([dOut, dIn, 0, 0]));
  const cfgB = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM });
  const frameB = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM });
  const C = GPUShaderStage.COMPUTE;
  const readY = async () => {
    const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(y, 0, stage, 0, dOut * 4);
    device.queue.submit([enc.finish()]); await stage.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(stage.getMappedRange().slice(0)); stage.unmap(); return out;
  };
  const results = [];
  let ref = null;   // tree output at the first shape; every subgroup candidate must match it
  for (const variant of variants) for (const [wg, rows] of shapes) {
    if (variant === "sgrow" && wg % sgRow !== 0) continue;
    try {
      const suffix = variant === "tree" ? "" : "_" + variant;
      const entry = (kind === "q4" ? "matvec_q4_coop" : "matvec_q8_coop") + suffix;
      const code = (sgPrefix || "") + WGSL + coopWGSL(wg, rows, 64, 4, rows, { sg: variant !== "tree", sgRow: variant === "sgrow" ? sgRow : 0 });
      const mod = device.createShaderModule({ code });
      const l0 = device.createBindGroupLayout({ entries: [0, 1].map((b) => ({ binding: b, visibility: C, buffer: { type: "uniform" } })) });
      const l1 = device.createBindGroupLayout({ entries: ["read-only-storage", "read-only-storage", "read-only-storage", "storage", "uniform"].map((t, i) => ({ binding: i, visibility: C, buffer: { type: t } })) });
      const pipe = await device.createComputePipelineAsync({
        layout: device.createPipelineLayout({ bindGroupLayouts: [l0, l1] }),
        compute: { module: mod, entryPoint: entry },
      });
      const bg0 = device.createBindGroup({ layout: l0, entries: [{ binding: 0, resource: { buffer: cfgB } }, { binding: 1, resource: { buffer: frameB } }] });
      const bg1 = device.createBindGroup({ layout: l1, entries: [qs, sc, x, y, shape].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
      const wgs = variant === "sgrow" ? Math.ceil(dOut / (wg / sgRow)) : Math.ceil(dOut / rows);
      const run = (n) => {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipe); pass.setBindGroup(0, bg0); pass.setBindGroup(1, bg1);
        for (let i = 0; i < n; i++) pass.dispatchWorkgroups(wgs);
        pass.end();
        device.queue.submit([enc.finish()]);
        return device.queue.onSubmittedWorkDone();
      };
      // correctness gate: subgroup candidates must reproduce the tree (measured diff <= 5e-7 abs on O(1) outputs)
      await run(1);
      const out = await readY();
      if (!ref) ref = out;
      else {
        let maxd = 0, maxr = 0;
        for (let i = 0; i < dOut; i++) { maxd = Math.max(maxd, Math.abs(out[i] - ref[i])); maxr = Math.max(maxr, Math.abs(ref[i])); }
        if (!(maxd <= 1e-5 * Math.max(1, maxr))) { results.push({ wg, rows, variant, ms: Infinity, maxd }); continue; }
      }
      const tw = performance.now();
      while (performance.now() - tw < (results.length ? 40 : 250)) await run(20);   // clock ramp
      const t0 = performance.now();
      await run(100);
      results.push({ wg, rows, variant, ms: (performance.now() - t0) / 100 });
    } catch { /* not supported on this device (e.g. sg entry missing); skip */ }
  }
  for (const b of [qs, sc, x, y, stage]) b.destroy();
  const ok = results.filter((r) => isFinite(r.ms));
  if (!ok.length) return { wg: 256, rows: 4, variant: "tree", results };
  ok.sort((a, b) => a.ms - b.ms);
  const best = ok[0];
  const def = ok.find((r) => r.wg === 256 && r.rows === 4 && r.variant === "tree");
  const pick = def && def.ms <= best.ms * 1.03 ? def : best;
  return { wg: pick.wg, rows: pick.rows, variant: pick.variant, results };
}
```

Notes:
- `f32ToF16` already exists in engine.js (used by `quantizeQ4`).
- Tolerance 1e-5 relative to max|y| is ~20x the measured 4.8e-7 abs diff on O(1)-magnitude outputs; tighten after observing real numbers on Chrome/Apple.
- Runtime: 5 shapes x up to 3 variants = 15 candidates at ~0.15 s each = ~2 s at load (was ~1 s). Acceptable; if not, drop `[64,4]` for the sg variants.
- The batched/gu `_sg` entry points are not autotuned separately; they follow the single-column pick (same reduction structure). If `variant === "sgrow"` wins, the batched/gu kernels still use the two-stage `_sg` form (there is no batched sgrow).
- `autotuneCoop` is always called on the real `ai.device` (p2p.html:1078), so `sgPrefix` from that device is the right input.

## 5. Engine wiring

`engine.js` `_init` (L868) and `qwen35.js` `_init` (L470) gain two options: `sgPrefix = null` and `coopVariant = "tree"` (`"tree" | "sg" | "sgrow"`), plus `sgRow = 0`.

Module assembly (engine.js:890, qwen35.js:501):

```js
const useSg = sgPrefix !== null && coopVariant !== "tree";
const mod = device.createShaderModule({
  code: (sgPrefix || "") + WGSL + coopWGSL(coopWG, coopRows, 64, batchCols, coopRowsB, { sg: sgPrefix !== null, sgRow: coopVariant === "sgrow" ? sgRow : 0 }) + WGSL2,
});
this.sgSuffix = useSg ? "_sg" : "";                       // batched/gu/single all use the two-stage form
this.sgSingle = coopVariant === "sgrow" ? "_sgrow" : this.sgSuffix;   // single-column may use rows-per-subgroup
this.sgRow = sgRow;
```

(For `DenseEngine`, `coopWGSL(coopWG, coopRows, 64, 4, coopRows, {...})`.)

Pipeline table `G1` (engine.js:900-921, qwen35.js:509-530): add entries only when the entry points exist, otherwise `createComputePipelineAsync` fails on a missing entry point:

```js
if (sgPrefix !== null) Object.assign(G1, {
  matvec_q8_coop_sg: ["ro", "ro", "ro", "rw", "u"], matvec_q4_coop_sg: ["ro", "ro", "ro", "rw", "u"],
  matvec_q8_coop_b_sg: ["ro", "ro", "ro", "rw", "u"], matvec_q4_coop_b_sg: ["ro", "ro", "ro", "rw", "u"],
  matvec_q8_gu_sg: ["ro", "ro", "ro", "ro", "ro", "rw", "u"], matvec_q4_gu_sg: ["ro", "ro", "ro", "ro", "ro", "rw", "u"],
  matvec_q8_gu_b_sg: ["ro", "ro", "ro", "ro", "ro", "rw", "u"], matvec_q4_gu_b_sg: ["ro", "ro", "ro", "ro", "ro", "rw", "u"],
});
if (sgPrefix !== null && sgRow > 0 && coopVariant === "sgrow") Object.assign(G1, {
  matvec_q4_coop_sgrow: ["ro", "ro", "ro", "rw", "u"], matvec_q8_coop_sgrow: ["ro", "ro", "ro", "rw", "u"],
});
```

Pipeline selection (qwen35.js `mv` L610-615, `guOp` L616-624, and the `_coop_b` site at engine.js:1235 / its qwen35 equivalent):

```js
const mv = (w, x, y, dOut, dIn) => {
  const base = w.kind === "q8" ? "matvec_q8" : w.kind === "q4" ? "matvec_q4" : "matvec";
  const sfx = w.kind === "f32" ? "" : this.sgSingle;                       // f32 kernels stay tree-only
  const pipe = coop ? base + "_coop" + sfx : base;
  const wgs = !coop ? Math.ceil(dOut / 64)
            : sfx === "_sgrow" ? Math.ceil(dOut / (this.coopWG / this.sgRow))
            : Math.ceil(dOut / this.coopRows);
  const bufs = w.kind === "f32" ? [w.buf, x, y, this._shape(dOut, dIn)] : [w.qs, w.sc, x, y, this._shape(dOut, dIn)];
  return { pipe, wgs, bg: this._bg(this.pipes[pipe], 1, bufs) };
};
// guOp: pipe = base + (xB ? "_b" : "") + (wg2.kind === "f32" ? "" : this.sgSuffix)
// batched mv (_coop_b): pipe = base + "_coop_b" + (kind === "f32" ? "" : this.sgSuffix); wgs unchanged
```

Dispatch geometry is unchanged for every `_sg` kernel; only `_sgrow` changes it. The `_shape`/`_shapeB` uniforms and bind groups are identical, so `_bg` caching keys still work.

`p2p.html` (L1052, L1078, L1141/1154/1163):

```js
ai.device = await requestDeviceWithSubgroups(adapter, { requiredLimits: { ... } });
ai.sgPrefix = await subgroupPrefix(ai.device);
ai.sgRow = ai.sgPrefix === null ? 0 : fixedSubgroupSize(adapter);
...
ai.tune = await autotuneCoop(ai.device, { sgPrefix: ai.sgPrefix, sgRow: ai.sgRow }).catch(() => ({ wg: 256, rows: 4, variant: "tree" }));
crumb(`autotune: WG=${ai.tune.wg} ROWS=${ai.tune.rows} variant=${ai.tune.variant} sg=${ai.sgPrefix === null ? "none" : ai.sgPrefix ? "enable" : "bare"}`);
...
Qwen35Engine.create({ ..., coopWG: ai.tune?.wg, coopRows: ai.tune?.rows, coopVariant: ai.tune?.variant, sgPrefix: ai.sgPrefix, sgRow: ai.sgRow });
```

Deno scripts: same three lines with the shared helper. Add an env override (`SGV=tree|sg|sgrow`) in `bench_deno.js` / `bench_single_deno.js` so the variants can be A/B'd on the real model without waiting for autotune.

P2P peers are heterogeneous: each device runs its own gate and autotune; the choice is local and never affects the wire format or dispatch semantics, so mixed tree/sg rooms are fine.

## 6. Verification

1. Unit: extend `kernelMicroTests(device)` (engine.js:1650) to run the q8 (and q4) single-column, `_coop_b` and `_gu`/`_gu_b` kernels in both `tree` and `_sg` forms on its 128-dim synthetic model and assert max-abs-diff <= 1e-5 (measured 1.8e-7..4.8e-7 on the big shapes). Run it on the throwaway test device (p2p.html:1068) created with `requestDeviceWithSubgroups` and the same `sgPrefix`.
2. Load-time: the diff gate inside `autotuneCoop` (section 4) rejects any subgroup candidate that does not reproduce the tree, which covers the two non-spec assumptions (lane-0 presence in every subgroup; `t / SG` mapping for `_sgrow`) on whatever hardware the page lands on.
3. Golden: `test_q38_deno.js` / `test_mtp_deno.js` with `SGV=sg` must match the tree run's tokens under greedy decoding (bench-log reference is bit-identical greedy). Because f32 sums are reordered, logits differ at ~1e-7; a greedy tie flip is possible but was not observed in the verification runs. If a flip appears, compare argmax on the logits with a 1e-5 tolerance rather than exact tokens.
4. Perf: re-run `bench_sg.js` / `verify_claim_sg.js` style A/B per shape on each target (GB10/Deno, Chrome on the Mac, Chrome on NVIDIA/AMD), not just the autotune shape; numbers vary run to run, so report ratios from repeated runs with a clock warm-up and discard the first slot.
5. Chrome-specific compile check (UNCERTAIN until run): the kernels compile on naga today; on Chrome 134-144 the workgroup-scoped `subgroup_uniformity` analysis must accept `subgroupAdd(p_i)` after the `for (k = sg_lane; ...)` loop. llama.cpp ships the same structure under Dawn, so failure is unlikely; if Dawn rejects it, add `diagnostic(off, subgroup_uniformity);` after the enable directive for that module (safe here because every call really is workgroup-uniform).

## 7. Open items / UNCERTAIN

- Chrome D3D12 subgroup emulation: whether every emulated subgroup has an invocation with `subgroup_invocation_id == 0` and whether emulated widths are ever below 4 — the diff gate handles both by falling back to the tree.
- `subgroupBroadcastFirst` availability in naga (would make the partial write fully mapping-independent); not probed.
- Safari once subgroups ship (TP 249): whether `adapter.info.subgroupMinSize/MaxSize` are populated and whether Metal accepts the bare-builtin form; the probe handles the directive question automatically.
- wgpu flipping to require `enable subgroups;` (wgpu#8202 proposal): handled by the probe order (`enable` first, bare second).
- `getCompilationInfo()` semantics on Deno for naga parse errors (error scope covers it).
- Gains on Apple/Intel (where shared memory + barriers are relatively costlier) are unmeasured; the GB10 numbers (0-8%) are the only data.

---

# Spec: f16 activation (x) storage for the cooperative GEMV family

## 0. Verdict first (read before implementing)

Measured on the target GB10 (Deno 2.9.5 / wgpu / Vulkan, shader-f16 present), with a micro-bench that reproduces `matvec_q4_coop` exactly and only swaps how x is stored/loaded, **f16 x storage gives no speedup at any shape**:

| Shape (Q4_0, L2-cold weights) | f32 x (today) | f16 x, f32 math (variant B) | notes |
|---|---|---|---|
| 5120x5120 (attn/DN proj) | 0.154 ms | 0.95x  (re-run: 1.05x, reversed order 0.96x) | pure run-order noise |
| 17408x5120 (gate/up) | 0.419 ms | ~0.99x (re-run 0.97x-1.03x) | noise |
| 248320x5120 (LM head) | 4.57 ms | 0.90x | slightly slower |

Why: x is 20 KB and is re-read by every workgroup from L1/L2 (25.6 MB of cache hits per 5120x5120 dispatch vs 14.7 MB of DRAM weight stream); halving it changed nothing, so neither DRAM nor L1/L2 bandwidth on x is the limiter. The kernel is latency/occupancy-bound on the weight stream (92-166 GB/s of 273 GB/s peak). f16 math (variant D) was also 0.92-1.03x and 3x less accurate; NVIDIA non-tensor FP16 throughput equals FP32 and every f16->f32 cvt is an extra half-rate instruction.

**Expected decode gain on GB10: 0% (within +/-5% noise; the LM head measured -10%).** Precision cost: relL2 1.4e-4..2.4e-4 per GEMV, maxabs ~5e-3 (the 2^-11 rounding of x; same class as the f16 P2P hidden-state hop), which breaks bit-identical greedy golden tests.

The rest of this document is the implementation-ready spec you asked for, so it can be built as a feature-gated, off-by-default autotune candidate (the only place it might pay is Apple/Metal, which is UNCERTAIN, see section 8). If the goal is tok/s on GB10, do not build this; the measured levers are memory-level parallelism in the weight sweep, dispatch fusion, and DP4a for the batched kernels.

## 1. Scope and design decisions

1. **Storage-only f16.** x buffers become `array<vec4<f16>>`; every load is immediately upcast with `vec4<f32>(...)`; dequant, dot, accumulators, shared-memory partials and outputs stay f32. Rule (a)-(d) from the precision finding: no `dot(vec4<f16>, vec4<f16>)`, no `f16*f16`, no f16 partials, residual stream / embeddings / norm weights stay f32.
2. **Phase 1 converts one tensor only: `xn` (the RMSNorm output).** It feeds q/k/v (full-attn layers), qkv/z/beta/alpha (DeltaNet in-proj), gate/up (fused `_gu`), and the LM head (Q8) - i.e. the majority of GEMV bytes. `g` (down-proj input), `attnOut` (o-proj input), `gated` (DeltaNet out-proj input) and MTP `ehIn` stay f32 in phase 1. Consequence: both an f32-x and an f16-x kernel family must coexist in the same module, selected per op by the x buffer's dtype.
3. **Separate entry points, suffix `_h`**, not a module-wide switch: `matvec_q4_coop_h`, `matvec_q8_coop_h`, `matvec_coop_h`, `matvec_{q4,q8,}_gu_h`, `matvec_{q4,q8,}_gu_b_h`, `matvec_{q4,q8,}_coop_b_h`. They bind the same @group/@binding slots with a second `array<vec4<f16>>` view (legal as long as no single entry point references both views - same trick the file already uses at engine.js:630-636).
4. **No `bitcast<vec2<f16>>(u32)` anywhere.** naga (Deno) lowers it as a scalar u32->f32 reinterpretation and splats (verified: 0xC0003E00 -> [-2.0038, -2.0038]). Native `array<vec4<f16>>` loads and `unpack2x16float` are both correct on Deno.
5. **Index math is unchanged.** `vec4<f16>` has align 8 / size 8, so `array<vec4<f16>>` element i covers x[4i..4i+3] exactly like `array<vec4<f32>>` element i today; only the byte stride halves. The batched `BShape.xs4` uniform must therefore be passed in units of the x element type: `stride/16` for f32 kernels, `stride/8` for `_h` kernels.
6. **Feature gate:** `shader-f16` requested on the device; `enable f16;` emitted as the first line of the module only when the device has it; `_h` entry points emitted only then. Never put `enable f16;` in a module that must compile on a non-f16 adapter.
7. **Producers:** `rmsnorm` and `rmsnorm_mc` get `_h` variants writing `array<f16>`; the residual `x` input of those kernels stays f32.

## 2. Device request and module prefix

### 2.1 p2p.html:1052 (browser) and every `engine/*_deno.js` `requestDevice` call

```js
const f16 = adapter.features.has("shader-f16");   // Chrome 120+, Safari 26 (all Apple GPUs), Deno 2.9.5/GB10: true
ai.device = await adapter.requestDevice({
  requiredFeatures: f16 ? ["shader-f16"] : [],
  requiredLimits: { /* unchanged */ },
});
```

`adapter.features.has("shader-f16")` is reliable for this feature (unlike `subgroups` on Deno, which is filtered by `all_webgpu_mask`; that quirk does not apply to shader-f16, which is a WebGPU-mask feature and was observed exposed on the GB10 adapter).

### 2.2 Module assembly (qwen35.js:501, engine.js:890, engine.js:1477)

```js
const hasF16 = device.features.has("shader-f16") && this.mvVariant === "coop";  // scalar matvec_q4/q8 read array<f32>; keep them f32-only
this.hasF16 = hasF16;
const code = (hasF16 ? "enable f16;\n" : "") + WGSL + coopWGSL(coopWG, coopRows, 64, batchCols, coopRowsB, { f16x: hasF16 }) + WGSL2 + (hasF16 ? WGSL_F16_PRODUCERS : "");
const mod = device.createShaderModule({ code });
```

`enable f16;` must precede every declaration in the module, so it goes before `WGSL` (whose first line is `struct Config`).

### 2.3 Load-time self-test (mandatory on Deno; Deno issue #23125 "f16 compute produces zeros" is still open)

Run once after pipeline creation, before trusting any `_h` pipeline:

```wgsl
// appended only when hasF16
@group(1) @binding(0) var<storage, read> f16t_in: array<vec4<f16>>;
@group(1) @binding(1) var<storage, read_write> f16t_out: array<vec4<f32>>;
@compute @workgroup_size(1)
fn f16_selftest() { f16t_out[0] = vec4<f32>(f16t_in[0]); }
```

```js
async _f16SelfTest() {
  // bits: 1.5h=0x3E00, -2.0h=0xC000, 0.5h=0x3800, 65504h=0x7BFF  -> two u32 words, low half first
  const inBuf = this._buf(new Uint32Array([0xC0003E00, 0x7BFF3800]), GPUBufferUsage.STORAGE);
  const out = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const stage = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = this.device.createCommandEncoder(); const p = enc.beginComputePass();
  p.setPipeline(this.pipes.f16_selftest); p.setBindGroup(0, this.bg0);
  p.setBindGroup(1, this._bg(this.pipes.f16_selftest, 1, [inBuf, out])); p.dispatchWorkgroups(1); p.end();
  enc.copyBufferToBuffer(out, 0, stage, 0, 16); this.device.queue.submit([enc.finish()]);
  await stage.mapAsync(GPUMapMode.READ);
  const v = Array.from(new Float32Array(stage.getMappedRange())); stage.unmap();
  const ok = v[0] === 1.5 && v[1] === -2 && v[2] === 0.5 && v[3] === 65504;
  if (!ok) { console.warn("shader-f16 self-test failed", v, "-> f16 x path disabled"); this.hasF16 = false; }
  return ok;
}
```

If it fails, `hasF16=false` and all op selection below falls back to the f32 family (the module still contains the `_h` entry points; they are simply never dispatched).

## 3. coopWGSL changes (engine.js:411-826)

### 3.1 Signature and x-view descriptor

```js
export function coopWGSL(WG = 256, ROWS = 4, WGB = 64, COLS = 4, ROWSB = ROWS, { f16x = false } = {}) {
  // x views: every kernel is emitted once per view. cv() is applied to EVERY x load; nothing else changes.
  const XV = [{ sfx: "", ty: "vec4<f32>", cv: (e) => e }];
  if (f16x) XV.push({ sfx: "_h", ty: "vec4<f16>", cv: (e) => `vec4<f32>(${e})` });
```

### 3.2 Bindings (engine.js:633-636 and the gu decl at 442/449)

Add, after the existing three f32 views, when `f16x`:

```wgsl
@group(1) @binding(1) var<storage, read> mv_x4_h: array<vec4<f16>>;
@group(1) @binding(2) var<storage, read> q8_x4_h: array<vec4<f16>>;
@group(1) @binding(2) var<storage, read> q4_x4_h: array<vec4<f16>>;
```

and in `guKernel`'s `decl` (which is emitted per kind once): for the f32 kind at @binding(2), for q8/q4 at @binding(4):

```wgsl
@group(1) @binding(${kind === "f32" ? 2 : 4}) var<storage, read> ${P}_x_h: array<vec4<f16>>;
```

Because `decl` is emitted inside `guKernel(kind, false)`, restructure so the decl+helpers block is emitted once per `kind` (not per view) and the entry points are emitted per view. Concretely: `["f32","q8","q4"].map((k) => guDecl(k) + XV.map((X) => guKernel(k, false, X)).join("\n"))`.

### 3.3 Load-site replacements (the only lines that change inside kernel bodies)

Every kernel emitter gains an `X` parameter; its name gets `${X.sfx}`; the x buffer name gets `${X.sfx}`; each load is wrapped in `X.cv`. Exact sites:

| engine.js line | today | replacement |
|---|---|---|
| 472-473 (gu q4 loads) | `let xa = ${P}_x[xcol + b * 8u + qt];` | `let xa = ${X.cv(`${P}_x${X.sfx}[xcol + b * 8u + qt]`)};` (same for xb) |
| 476-477, 480-481 (gu q8/f32 loads) | `let xa = ${P}_x[xcol + b * 8u + qt * 2u];` | `let xa = ${X.cv(`${P}_x${X.sfx}[xcol + b * 8u + qt * 2u]`)};` (same for xb) |
| 524 (gu entry name) | `fn matvec${...}_gu${batched ? "_b" : ""}(` | append `${X.sfx}` |
| 546-547 (gu_b xLoads) | `let xa${m} = ${P}_x[...]` | `let xa${m} = ${X.cv(`${P}_x${X.sfx}[...]`)}` |
| 582 (gu_b entry name) | `..._gu_b(` | `..._gu_b${X.sfx}(` |
| 654-655 (matvec_coop) | `let xa = mv_x4[c4];` | `let xa = ${X.cv(`mv_x4${X.sfx}[c4]`)};` |
| 642 | `fn matvec_coop(` | `fn matvec_coop${X.sfx}(` |
| 693-694 (q8) | `let xa = q8_x4[x4];` | `let xa = ${X.cv(`q8_x4${X.sfx}[x4]`)};` |
| 681 | `fn matvec_q8_coop(` | `fn matvec_q8_coop${X.sfx}(` |
| 731-732 (q4) | `let xlo = q4_x4[b * 8u + qt];` | `let xlo = ${X.cv(`q4_x4${X.sfx}[b * 8u + qt]`)};` |
| 719 | `fn matvec_q4_coop(` | `fn matvec_q4_coop${X.sfx}(` |
| 758 (coop_b xbuf) | `const xbuf = ... "q4_x4";` | `const xbuf = (... ) + X.sfx;` and wrap 762-763 loads in `X.cv` |
| 786 | `fn matvec${kind}_coop_b(` | `fn matvec${kind}_coop_b${X.sfx}(` |

Everything else (helper fns `q4_row/q8_row/mvf_row`, accumulators, `reduce`, shared arrays, stores, tail guards) is untouched: they already take `vec4<f32>` parameters.

### 3.4 Complete resulting single-column Q4 kernel (the `_h` instance, WG=256, ROWS=4, for review)

```wgsl
// module starts with:  enable f16;
@group(1) @binding(2) var<storage, read> q4_x4_h: array<vec4<f16>>;   // 8-byte stride; element i == x[4i..4i+3]

@compute @workgroup_size(256)
fn matvec_q4_coop_h(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let qt = t & 3u;
  let bl = t >> 2u;
  let dIn = q4_shape.dIn;
  let nb = dIn / 32u;
  let rowWords = dIn / 8u;
  let row0 = wg.x * 4u;
  let full = row0 + 3u < q4_shape.dOut;
  var acc0 = 0.0; var acc1 = 0.0; var acc2 = 0.0; var acc3 = 0.0;   // f32, scalar (arrays spill)
  for (var b: u32 = bl; b < nb; b += 64u) {
    let xlo = vec4<f32>(q4_x4_h[b * 8u + qt]);        // f16 storage -> f32 immediately; one 64-bit load
    let xhi = vec4<f32>(q4_x4_h[b * 8u + qt + 4u]);
    let wIdx = row0 * rowWords + b * 4u + qt;
    let scBase = row0 * nb + b;
    if (full) {
      acc0 += q4_row(wIdx,                q4s(scBase),          xlo, xhi);
      acc1 += q4_row(wIdx + rowWords,     q4s(scBase + nb),     xlo, xhi);
      acc2 += q4_row(wIdx + 2u*rowWords,  q4s(scBase + 2u*nb),  xlo, xhi);
      acc3 += q4_row(wIdx + 3u*rowWords,  q4s(scBase + 3u*nb),  xlo, xhi);
    } else {
      if (row0      < q4_shape.dOut) { acc0 += q4_row(wIdx,               q4s(scBase),         xlo, xhi); }
      if (row0 + 1u < q4_shape.dOut) { acc1 += q4_row(wIdx + rowWords,    q4s(scBase + nb),    xlo, xhi); }
      if (row0 + 2u < q4_shape.dOut) { acc2 += q4_row(wIdx + 2u*rowWords, q4s(scBase + 2u*nb), xlo, xhi); }
    }
  }
  mvc_part[t] = acc0; mvc_part[256u + t] = acc1; mvc_part[512u + t] = acc2; mvc_part[768u + t] = acc3;
  workgroupBarrier();
  var stride: u32 = 128u;
  while (stride > 0u) {
    if (t < stride) {
      mvc_part[t] += mvc_part[t + stride];
      mvc_part[256u + t] += mvc_part[256u + t + stride];
      mvc_part[512u + t] += mvc_part[512u + t + stride];
      mvc_part[768u + t] += mvc_part[768u + t + stride];
    }
    workgroupBarrier();                       // outside the conditional
    stride = stride >> 1u;
  }
  if (t < 4u) {
    let row = row0 + t;
    if (row < q4_shape.dOut) { q4_y[row] = mvc_part[t * 256u]; }
  }
}
```

This is exactly variant B of the measured micro-bench (`bench_f16x.js`), which reproduced the engine kernel at engine.js:710-746 and was verified numerically (relL2 vs f64 1.4e-4..2.4e-4, i.e. pure f16 rounding of x).

## 4. Producer kernels (RMSNorm -> f16 xn)

Append (only when `hasF16`) as `WGSL_F16_PRODUCERS`; these share `@group(0)` cfg/frame and reuse the existing binding slots with an f16 output view.

```wgsl
// --- rmsnorm_h: y(f16) = x * invRms(x) * w ; x, w, partials all f32 ---
@group(1) @binding(2) var<storage, read_write> rn_yh: array<f16>;      // same slot as rn_y; different entry point
var<workgroup> rnh_partial: array<f32, 256>;
@compute @workgroup_size(256)
fn rmsnorm_h(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  var ss: f32 = 0.0;
  for (var i: u32 = t; i < rn_n; i += 256u) { let v = rn_x[i]; ss += v * v; }
  rnh_partial[t] = ss;
  workgroupBarrier();
  var stride: u32 = 128u;
  while (stride > 0u) {
    if (t < stride) { rnh_partial[t] += rnh_partial[t + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let inv = inverseSqrt(rnh_partial[0] / f32(rn_n) + cfg.eps);
  for (var i: u32 = t; i < rn_n; i += 256u) {
    rn_yh[i] = f16(clamp(rn_x[i] * inv * rn_w[i], -65504.0, 65504.0));   // clamp: never write inf (see 8.2)
  }
}

// --- rmsnorm_mc_h: multi-column variant (qwen35.js:190-211 layout). rnm_mc.s1 is now in f16 elements ---
@group(1) @binding(2) var<storage, read_write> rnm_yh: array<f16>;
var<workgroup> rnmh_partial: array<f32, 256>;
@compute @workgroup_size(256)
fn rmsnorm_mc_h(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x; let n = rnm_mc.n;
  let xo = wg.y * rnm_mc.s0; let yo = wg.y * rnm_mc.s1;
  var ss: f32 = 0.0;
  for (var i: u32 = t; i < n; i += 256u) { let v = rnm_x[xo + i]; ss += v * v; }
  rnmh_partial[t] = ss;
  workgroupBarrier();
  var stride: u32 = 128u;
  while (stride > 0u) {
    if (t < stride) { rnmh_partial[t] += rnmh_partial[t + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let inv = inverseSqrt(rnmh_partial[0] / f32(n) + cfg.eps);
  for (var i: u32 = t; i < n; i += 256u) {
    rnm_yh[yo + i] = f16(clamp(rnm_x[xo + i] * inv * rnm_w[i], -65504.0, 65504.0));
  }
}
```

Note: `array<f16>` in the storage address space requires `enable f16;` (align 2, size 2); a scalar `f16` store is fine here because the producer is a launch-floor kernel (5120 elements), not a bandwidth kernel.

## 5. Host changes (qwen35.js; engine.js `_init` mirrors the same edits)

1. **Pipeline table (qwen35.js:521/530 and the `G1` layout table at engine.js:900-908):** duplicate each coop entry with the `_h` suffix and identical layout arrays; add `rmsnorm_h: ["ro","ro","rw","u"]`, `rmsnorm_mc_h: ["ro","ro","rw","u"]`, `f16_selftest: ["ro","rw"]`. Guard with `if (this.hasF16)`.

2. **xn buffers.** Tag the buffer: `this.xn = device.createBuffer({ size: dim * (hasF16 ? 2 : 4), usage: S }); this.xn.f16 = hasF16;` (qwen35.js:574). For the batched set (qwen35.js:923) keep `mkB(D.dim)` (256-byte-aligned f32-size slices; the extra half is unused when f16) and set `B.xn.f16 = hasF16`. Do not shrink `B.xn.stride` in phase 1; it keeps `_shapeB` keys and `slice()` helpers untouched.

3. **Op selection.** In `mv` (qwen35.js:610), `guOp` (617) and `mvB` (951):

```js
const hs = (x) => (x?.f16 || x?.buf?.f16) ? "_h" : "";        // x is a GPUBuffer (mv/guOp) or {buf,stride} (mvB)
// mv:   const pipe = coop ? base + "_coop" + hs(x) : base;
// guOp: const pipe = (xB ? base + "_b" : base) + hs(xB ?? x);
// mvB:  pipe = base + "_coop_b" + hs(xB);
```

and pass `BShape.xs4` in element units:

```js
const xs4 = (xB) => xB.stride / (xB.f16 ? 8 : 16);
// guOp: this._shapeB(dOut, dIn, xs4(xB), yB.stride / 4)      (line 621)
// mvB:  this._shapeB(dOut, dIn, xs4(xB), yB.stride / 4)      (line 954)
```

Only ops whose x is `xn`/`B.xn` resolve to `_h`: `mvQ/mvK/mvV`, `mvQKV/mvZ/mvBeta/mvAlpha`, `gu`, `mvGate/mvUp`, `headOp` (633-707), and the batched `qkvOps`, `dnOps`, `gateUp`, `gu`, `headB` (960-1032). `mvO`, `mvDown`, `mvOut`, `M2.proj` and their batched twins stay f32 automatically because their x buffers carry no tag.

4. **RMSNorm dispatch.** Every `rmsnorm`/`rmsnorm_mc` dispatch whose output is xn switches to the `_h` pipeline: qwen35.js:793, 818, 836, 1065, 1093, 1106, 1170, 1196 (`bgHNorm*` -> xn), plus `bgFinalNorm`/`bgFinalNormMC`, `M2.bgHeadNorm`. Line 1195 (`bgENorm`) writes the MTP `ehIn` half, not xn - leave it f32. Bind groups: `bgNorm(...)` at 626 and the `_bg2res(this.pipes.rmsnorm_mc, ...)` calls at 974-981 must use `this.pipes.rmsnorm_h` / `rmsnorm_mc_h`, and the `mcU(D.dim, st(B.x), st(B.xn))` uniform must pass the xn stride in f16 elements: `st16 = (b) => b.stride / 2`.

5. **Autotune (engine.js:1460):** add `xF16` as a candidate axis only if you want the numbers on other hardware; on the GB10 it will lose or tie (section 7). Default: `hasF16 && autotuneSaysF16` with the default `false` on NVIDIA/Vulkan.

6. **Readback / P2P:** unaffected. `stageX`/`stageXB` read `this.x`/`B.x` (residual, f32). The P2P hop already sends f16 of the residual; nothing about xn leaves the GPU.

## 6. Tests

1. `f16_selftest` (section 2.3) at load; disable the path on failure (Deno #23125 guard).
2. Kernel A/B at load or in `test_*_deno.js`: run `matvec_q4_coop` vs `matvec_q4_coop_h` on the same random x (f32 and its f16 rounding) and assert `maxabs <= 1e-2` on outputs of magnitude ~3 and `relL2 <= 5e-4` (measured 1.4e-4..2.4e-4; the f16-math variant D at 4.2e-4 is the "too far" reference).
3. Golden tests (`test_q38_*_deno.js`) are bit-identical greedy today. With f16 xn they will not be bit-identical. Add a logit-tolerance harness (max |dlogit| and top-1 agreement rate over the golden prompt) before flipping any default; treat argmax flips on near-ties as expected, not as failures.
4. Overflow scan: in the golden run, read xn back once per layer for the test prompt and assert `max|xn| < 6e4` (section 8.2). The clamp in the producer prevents inf propagation but would silently distort if ever hit.
5. Timing: wall-clock, best-of-N, alternate order (A, B, A) and discard the first slot - the measured first-slot clock-ramp penalty was ~55%, far larger than the effect being measured. No timestamp queries (3x slowdown when merely enabled).

## 7. Expected gain

- **GB10 decode (9 tok/s plain, 16 MTP):** 0%. Predicted range -1%..+1% per token, from measured per-kernel ratios 0.90x-1.06x that flip sign with run order. The LM head (7-8 ms/token, Q8) measured 0.90x with f16 x - if that reproduces at the token level it is a ~0.7 ms/token loss; if it is noise, nothing. Weight bytes per token (15.2 GB / 0.11 s = ~137 GB/s effective) are unchanged by this work.
- **GB10 prefill/verify (batched `_b`, `_gu_b`):** not measured with f16 x. Those kernels are x-load (L1) + FMA-issue bound at N>=8 (hold layout: 4 vec4 x loads per weight word per thread); halving x bytes in L1 might matter there but the N=1 result says L1 bandwidth was not the limiter. UNCERTAIN; must be measured with `bench_ncols.js`-style runs before claiming anything.
- **Memory:** xn 20 KB -> 10 KB per column. Irrelevant.
- **Apple/Safari 26 (M-series):** UNCERTAIN. f16 storage does not change ALU rate on M1+ (FP16 FMA rate == FP32), but halved register/register-cache footprint and lower dependent-FMA latency could help the batched kernels that hold 2*ROWS*4 accumulators plus 8 x vectors. No measurement exists; this is the only platform where the `_h` autotune candidate might be selected.

## 8. UNCERTAIN / flagged items

1. **Batched-kernel gain on GB10** - unmeasured (only single-column measured).
2. **xn overflow risk.** f16 max is 65504. xn = x*inv*w has unit-rms scale times the norm weight; Qwen residual outliers are normalized away by the rms, so overflow is not expected, but max|xn| over Qwen3.8 was never measured. Hence the clamp plus the test-time scan in section 6.4. If the scan ever trips, use bf16-in-u32 (`bits>>16`, reconstruct `bits<<16`, no shader-f16 needed) instead - same bytes, full range.
3. **Deno f16 correctness** was verified for `array<vec2<f16>>` loads in the bitcast test and for `array<vec4<f16>>` in the bench (variant B numerically correct); `array<f16>` scalar *stores* (the producer side) were not separately verified on Deno - the self-test covers loads only. Extend the self-test to a store if you want to be strict.
4. **Golden-test flip rate** after f16 xn is unknown until the tolerance harness exists.
5. **Safari 26 `enable f16;` + `array<f16>` storage stores**: Safari exposes shader-f16 on all Apple GPUs as an optional feature; no engine test has run there with f16 storage. Runtime gate + self-test handle the "feature missing" case; a Safari-specific miscompile would only be caught by the A/B test in 6.2.
6. **Adreno**: some devices now expose shader-f16 in current Chrome (only `storageBuffer16BitAccess` required since Oct 2025); older ones do not. The gate handles both; no perf expectation either way.

## 9. Recommendation

Implement only if you want the autotune knob for non-NVIDIA devices; ship it default-off. Portable zero-gate alternative with identical measured performance and no `enable f16;` at all: keep xn as f32 and skip this entirely (variant C - u32 pairs + `unpack2x16float` - also measured at parity, so it buys nothing either). For GB10 tok/s, the measured order of levers is: (1) more weight bytes in flight per thread (2x-4x unrolled block loop, vec2/vec4<u32> weight loads; target 17408x5120 from ~120 to >=180 GB/s, worth ~+40% decode), (2) dispatch fusion against the ~0.05 ms x 640 per-token floor, (3) DP4a Q8_1 activations for the batched `_b`/`_gu_b` verify/prefill kernels, (4) f16/bf16 KV cache for the 16 attention layers at long context. f16 GEMV input is not on that list.

Files referenced: `engine/engine.js` (coopWGSL 411-826, module 890, autotune 1460), `engine/qwen35.js` (rmsnorm_mc 190-211, module 501, mv/guOp 610-624, xn alloc 574, batched mvB 951, norm bind groups 974-981, dispatch sites 793/818/836/1065/1093/1106/1170/1196), `p2p.html` (requestDevice 1052).

---

# Continuous Speculation for the Bello P2P chain — implementation spec

Scope: PipeInfer-style continuous speculation (arXiv 2407.11798) for the layer-split WebRTC chain in `p2p.html` + `engine/qwen35.js`. Host keeps drafting while lap N is on the wire, launches lap N+1 early on a gamble, and cancels it by versioned rollback when the gamble loses. Everything below is derived from the verified findings and the engine source as it is today (`qwen35.js` 1118-1287, `p2p.html` 1339-1509, 1587-1614). Items I could not verify are marked **UNCERTAIN**.

---

## 0. Summary of the design decisions

| Decision | Value | Why (finding) |
|---|---|---|
| Draft depth in continuous mode | K = 3 fixed (one 4-column chunk = `NC`) | q ≈ p^(K+1) collapses at K=7; 2(K+1) = 8 snapshot slots; one chunk per lap per device |
| Laps in flight | L = 2 | L=3 needs 12 slots and its third lap pays with probability q² |
| Gamble hypothesis | all K drafts of lap N accepted **and** bonus token at column K == d_{K+1} (the (K+1)-th chained draft) | only hypothesis that needs no target logits; no attention-mask trees for DeltaNet |
| Cancellation | versioned laps (`lap`, `epoch`) + receiver-side drop + probe between layer groups; rollback message doubles as cancel | ordered DataChannel delivers stale payloads before the rollback; `send()` cannot be recalled |
| Rollback primitive | DeltaNet slot restore (`copyBufferToBuffer`), slot = `(lap & 1) * (K+1) + k` | existing mechanism; KV caches are position-indexed and simply overwritten |
| Gate | draft-confidence product over the K+1 gamble events, PipeInfer cutoff + recovery/decay | Sec IV.B.2 |
| Peer concurrency | per-peer serialized async executor | today's fire-and-forget handlers would interleave GPU submits and double-map `stageXB` |

Expected gain (Sec 8): 1.2–1.5x at lap 300 ms, 1.45–1.8x at lap 600 ms, depending on which reading of p = 0.85 is right; ~1.1–1.2x without cancellation.

---

## 1. Prerequisites (ship first, zero behavior change)

Three latent hazards in the current code become live the moment a second lap is in flight. They are independent of the gamble logic and can ship as a refactor.

1. **Peer executor.** `p2p.html:1587-1604` runs `runHiddenBatch` inside an async message handler; `onData` (line 617) calls `aiOnData` fire-and-forget. With two laps arriving, chunk 0 of lap N+1 would be submitted between lap N's chunks (DeltaNet state applied out of order), and a second `stageXB.mapAsync` while one is pending is rejected with `OperationError` (unhandled → the host hits the 90 s timeout). Fix: every `ai-hidden-b` / `ai-lap` / `ai-rollback` goes through one promise-chain executor per peer (Sec 5.2).
2. **Waiters keyed by lap id.** `p2p.html:1464-1467` keys the verify waiter by `"b"+pos`. After a full-accept/bonus-miss, the corrected lap has the same `basePos` as the stale lap, so `Map.set` overwrites the stale resolver with the corrected lap's, and the stale `ai-hiddenret-b` resolves the corrected lap's promise with hiddens computed from the wrong tokens (`badF32` won't catch it). Fix: key by `"lap"+id`, carry `lap` in the return message.
3. **Rollback sequenced, not applied on receipt.** `p2p.html:1605-1609` calls `restoreDN(d.k)` immediately, which submits a copy at an arbitrary point between chunk submits. Fix: enqueue into the same executor.

---

## 2. Lap, slot and position model

Notation (matches `specStep`, `qwen35.js:1260-1287`): a lap with base position `pos` has columns `[tNext, d1..dK]` at positions `pos..pos+K`; `lgs[k]` predicts position `pos+k+1`; `out[k] = sample(lgs[k])`; draft `d_{k+1}` is accepted iff `out[k] == d_{k+1}`; `out[K]` is the bonus token at `pos+K+1`.

MTP chain (`mtpRun(null, tok, p, ...)` feeds `this.x` = MTP output hidden back in):

```
d1 = mtp(tNext, pos)   d2 = mtp(d1, pos+1) ... dK = mtp(d_{K-1}, pos+K-1)      <- lap N drafts
d_{K+1} = mtp(dK, pos+K)                                                       <- stand-in for lap N's bonus
d_{K+2} = mtp(d_{K+1}, pos+K+1) ... d_{2K+1} = mtp(d_{2K}, pos+2K)             <- lap N+1 drafts
lap N+1 = [d_{K+1}, d_{K+2}, ..., d_{2K+1}] at positions pos+K+1 .. pos+2K+1
```

Chain depth from the last exact hidden is 2K+1 = 7 — exactly today's `Math.min(7, K)` cap. Lap N+1 is valid iff lap N's `a == K` **and** `out[K] == d_{K+1}`.

### 2.1 Snapshot slots

State per DeltaNet layer per slot: S = 48·128·128·4 B = 3.0 MiB (+120 KiB conv). 48 DN layers → 144 MiB/slot full model; a peer with ~16 DN layers → 48 MiB/slot.

| Mode | slots | memory (full model / 16-DN-layer peer) |
|---|---|---|
| today (sequential, K≤7) | 7 | 1.0 GiB / 336 MiB |
| continuous K=3, L=2 | 2·(K+1) = **8** | 1.15 GiB / 384 MiB |
| continuous K=3, L=3 | 12 | 1.7 GiB / 576 MiB (not worth it) |
| continuous K=7, L=2 | 16 | 2.25 GiB (rejected) |

Slot map: `slotOf(lap, k) = (lap & 1) * (K+1) + k`, `k ∈ 0..K`. **Every** column of an in-flight lap is snapshotted (today the final column is excluded because `restoreDN(K)` was never needed; now "lap N fully accepted but bonus missed" restores `slot(N, K)`).

No WGSL change: the kernels (`qwen35.js:263-267, 327-331`) write slot `cvSB-1+col` when `cvSB + col < (frame.snap >> 8)`, `cvSB = base+1`. Passing `{ base: slotBase, total: slotBase + n + 1 }` snapshots all `n` columns into `slotBase..slotBase+n-1`. `frame.snap` packs `base+1` in 8 bits, so `slotBase ≤ 254`.

Restore cases at settle(N) with child N+1 in flight:

| verdict | restore | child |
|---|---|---|
| `a < K` | `slot(N, a)` | cancelled |
| `a == K`, `out[K] != d_{K+1}` | `slot(N, K)` | cancelled |
| `a == K`, `out[K] == d_{K+1}` | none | becomes head of the ring |

Full-attention KV (16 trunk layers + the MTP layer, `gguf.js:483` forceFull) is position-indexed, written by `copyBufferToBuffer` at `(basePos+c)*kvDim*4` (`qwen35.js:1075-1076`) and read only for keys `≤` the query position, so stale entries beyond the frontier are never read before being overwritten. No KV rollback, same as today.

### 2.2 Chain-state buffer (`xChain`)

The refill loop (`qwen35.js:1280-1283`) calls `mtpRun`, which overwrites `this.x`. In continuous mode the next lap's drafts must continue from the MTP output after `d_{2K+1}`, not from the refill. Add a `dim*4`-byte `xChain` buffer: copy `this.x → xChain` after the last draft of a lap; copy `xChain → this.x` before drafting a child lap. On a lost gamble the chain restarts from the exact hidden (`setHidden(hs[a])`) as today.

Refill in the valid-child case covers `j = 1..K+1` (positions `pos+1..pos+K+1`, exact hiddens `hs[0..K]`), i.e. one more MTP-block-only run than today, because the entry at `pos+K+1` was written by the draft chain from an inexact hidden.

---

## 3. Engine changes (`engine/qwen35.js`)

### 3.1 Slot count, `slotOf`, `xChain`

```js
// _init(...) signature: add snapSlots
async _init({ device, meta, weights, layerRange, hasEmbed = true, hasHead = true, maxSeq = 512,
              vocab: vocabOpt, matvecVariant = "coop", coopWG = 256, coopRows = 4, batchCols = 4,
              coopRowsB = coopRows, snapSlots = 8 }) {
  ...
  this.NC = batchCols; this.coopRowsB = coopRowsB;
  this.snapSlots = Math.max(7, snapSlots);      // 7 keeps today's sequential K<=7 path working
  ...
}

// _initBatch(): replace the two 7 * ... allocations
for (const L of this.layers) if (!L.isFull && !L.S_shadow) {
  L.S_shadow    = dev.createBuffer({ size: this.snapSlots * L.S.size,         usage: S });
  L.conv_shadow = dev.createBuffer({ size: this.snapSlots * L.convState.size, usage: S });
}
this.xChain = dev.createBuffer({ size: D.dim * 4, usage: S });   // MTP chain hidden parked across refills

// new methods
slotOf(lap, k) { return (lap & 1) * this.NC + k; }             // NC = K+1 = 4 in continuous mode
saveChain() {  const e = this.device.createCommandEncoder(); e.copyBufferToBuffer(this.x, 0, this.xChain, 0, this.dims.dim * 4); this.device.queue.submit([e.finish()]); }
loadChain() {  const e = this.device.createCommandEncoder(); e.copyBufferToBuffer(this.xChain, 0, this.x, 0, this.dims.dim * 4); this.device.queue.submit([e.finish()]); }
```

`this.x` must carry `COPY_SRC | COPY_DST` (it already receives `_adoptHidden` copies; **UNCERTAIN** whether it currently has `COPY_SRC` — check the allocation and add it).

`_restoreDN(k)` is unchanged; callers now pass `slotOf(lap, k)`. Assert `k < this.snapSlots`.

### 3.2 Cancellable batched pass

Refactor `_runBatchAndRead` so a peer can probe for cancellation between layer groups. Submitting all groups back-to-back would not help (the GPU executes already-submitted work), so each group is awaited with `onSubmittedWorkDone()` before the next is submitted; awaiting lets queued DataChannel messages dispatch. Cost is one GPU idle gap per group (**UNCERTAIN** magnitude; expected sub-ms on Chrome; measure with `groups = 1,2,4`).

```js
// shouldAbort: null (host/solo: one submit, no probe) or () => boolean
// groups: probe points per lap; waste per cancelled lap <= one group's GPU time
async _runBatchAndRead(basePos, n = this.NC, shouldAbort = null, groups = 4) {
  const { dim } = this.dims, nL = this.layers.length;
  const per = shouldAbort ? Math.ceil(nL / Math.max(1, groups)) : nL;
  for (let l0 = 0; l0 < nL; l0 += per) {
    const l1 = Math.min(nL, l0 + per);
    const enc = this.device.createCommandEncoder();
    for (let l = l0; l < l1; l++) this._encodeLayerBatch(enc, l, basePos, n);
    if (l1 === nL)
      for (let c = 0; c < n; c++) enc.copyBufferToBuffer(this.B.x.buf, c * this.B.x.stride, this.stageXB, c * dim * 4, dim * 4);
    this.device.queue.submit([enc.finish()]);
    if (shouldAbort && l1 < nL) {
      await this.device.queue.onSubmittedWorkDone();   // event loop runs here -> ai-rollback can land
      if (shouldAbort()) { this.pos = basePos; return null; }   // state is partial; the restore that follows fixes it
    }
  }
  await this.stageXB.mapAsync(GPUMapMode.READ, 0, n * dim * 4);
  const out = Float32Array.from(new Float32Array(this.stageXB.getMappedRange(0, n * dim * 4), 0, n * dim));
  this.stageXB.unmap();
  this.pos = basePos + n;
  return out;
}

async runHiddenBatch(xs, basePos, snapshot = false, shouldAbort = null, groups = 4) {
  if (!this.B) this._initBatch();
  const { dim } = this.dims;
  const n = xs.length / dim;
  this.pos = basePos;
  const sp = !snapshot ? 0 : typeof snapshot === "object"
    ? ((snapshot.total << 8) | (snapshot.base + 1)) : ((n << 8) | 1);
  for (let c = 0; c < n; c++) {
    this.device.queue.writeBuffer(this.frameBufsB[c], 0, new Uint32Array([basePos + c, basePos + c + 1, n, sp]));
    this.device.queue.writeBuffer(this.B.x.buf, c * this.B.x.stride, xs.subarray(c * dim, (c + 1) * dim));
  }
  return this._runBatchAndRead(basePos, n, shouldAbort, groups);
}
```

`embedRunBatch` is unchanged (host path, no probe).

### 3.3 Draft confidence: `argmax_lse`

Extend the single-workgroup argmax (`qwen35.js:335-359`) to also emit log-sum-exp, so `p_draft = exp(max − lse)` = the MTP head's probability of its own argmax. Second sweep over the 993 KB logits buffer (likely L2-resident; **UNCERTAIN** cost, expect tens of µs). Core WGSL only — runs on Deno/wgpu, Chrome and Safari. Barriers stay outside conditionals.

```wgsl
// --- argmax + log-sum-exp over n floats (single workgroup):
//     out = [index, bitcast(max), bitcast(logsumexp)]  ->  p(argmax) = exp(max - lse) ---
var<workgroup> am_s: array<f32, 256>;
@compute @workgroup_size(256)
fn argmax_lse(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x; let n = am_n.x;
  var bv: f32 = -3.402823e38; var bi: u32 = 0xffffffffu;
  for (var i: u32 = t; i < n; i += 256u) {
    let v = am_x[i];
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
  }
  am_v[t] = bv; am_i[t] = bi;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (t < s) {
      let ov = am_v[t + s]; let oi = am_i[t + s];
      if (ov > am_v[t] || (ov == am_v[t] && oi < am_i[t])) { am_v[t] = ov; am_i[t] = oi; }
    }
    workgroupBarrier();
  }
  let mx = am_v[0];                       // every thread reads the reduced max (am_v[0] is not written again)
  var acc: f32 = 0.0;
  for (var i: u32 = t; i < n; i += 256u) { acc += exp(am_x[i] - mx); }
  am_s[t] = acc;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (t < s) { am_s[t] += am_s[t + s]; }
    workgroupBarrier();
  }
  if (t == 0u) { am_out[0] = am_i[0]; am_out[1] = bitcast<u32>(mx); am_out[2] = bitcast<u32>(mx + log(am_s[0])); }
}
```

Host side: a separate pipeline `pipes.argmax_lse` and its own bind group `bgArgmaxLse` (same bindings: `this.logits`, `this.argBuf`, the `[vocab,0,0,0]` uniform). **UNCERTAIN**: whether `_bg` builds bind groups from `layout: "auto"`; if so a bind group from `pipes.argmax` is not reusable with `pipes.argmax_lse`, hence the separate bind group.

```js
// mtpRun: add wantLogits === "argmaxp"  ->  { id, p }
async mtpRun(srcCol, tNext, pos, wantLogits) {
  const M2 = this.mtp, { dim, vocab } = this.dims;
  const am = wantLogits === "argmax" || wantLogits === "argmaxp";
  this.device.queue.writeBuffer(M2.emb, 0, this._embedRowF32(tNext));
  this._setFrame(pos, pos + 1);
  const enc = this.device.createCommandEncoder();
  { const p = enc.beginComputePass();
    this._d(p, "rmsnorm", M2.bgENorm, 256, 256);
    this._d(p, "rmsnorm", srcCol === null ? M2.bgHNormX : M2.bgHNormB[srcCol], 256, 256);
    this._dop(p, M2.proj);
    p.end(); }
  this._encodeLayerR(enc, this.mtpLayer, pos);
  if (wantLogits) {
    const p = enc.beginComputePass();
    this._d(p, "rmsnorm", M2.bgHeadNorm, 256, 256);
    this._dop(p, this.headOp);
    if (wantLogits === "argmax")  this._d(p, "argmax",     this.bgArgmax,    256, 256);
    if (wantLogits === "argmaxp") this._d(p, "argmax_lse", this.bgArgmaxLse, 256, 256);
    p.end();
  }
  if (am) enc.copyBufferToBuffer(this.argBuf, 0, this.stageArg, 0, 16);
  this.device.queue.submit([enc.finish()]);
  if (!wantLogits) return null;
  if (am) {
    await this.stageArg.mapAsync(GPUMapMode.READ);
    const u = new Uint32Array(this.stageArg.getMappedRange().slice(0));
    this.stageArg.unmap();
    if (wantLogits === "argmax") return u[0];
    const f = new Float32Array(u.buffer);
    return { id: u[0], p: Math.exp(f[1] - f[2]) };
  }
  return await this._readback(this.logits, this.stageLogits, vocab);
}
```

`p_draft` is the drafter's confidence, not the target's acceptance probability. Calibrate: log `(p_draft, accepted)` pairs from `settle()` and fit a monotone map before trusting the gate (Sec 7).

---

## 4. Peer protocol

### 4.1 Messages

All laps and rollbacks carry a monotonic `lap` id (host-assigned) and an `epoch` (incremented by the host on every rollback). A lap is stale on a peer iff `msg.epoch < peer.epoch`.

```
host -> chain[0]           { t:"ai-lap", lap:N, epoch:E, parent:N-1|-1, basePos, n:4, slotBase:(N&1)*4,
                             restore:{afterLap,k,epoch}|null, data, enc:"f16" }     // 4x5120 f16 = 40 KB
peer -> next peer          same object (own hiddens in data)
last peer -> host          { t:"ai-lapret", lap:N, epoch:E, basePos, n, data, enc }
host -> every chain peer   { t:"ai-rollback", afterLap:N, k:a, epoch:E+1 }
```

Semantics:
- `ai-rollback{afterLap, k, epoch}`: truth is the state after column `k` of lap `afterLap`; every lap with a smaller epoch is cancelled. Restore = `restoreDN(slotOf(afterLap, k))`, applied **once per epoch**, sequenced after any lap currently running.
- `restore` on `ai-lap` is the same rollback carried redundantly (idempotent by epoch) on the first lap issued after it. Needed because the host→peer_i rollback travels on a different PeerJS connection than the lap chain, so at peers ≥ 2 the corrected lap can arrive before the direct rollback.
- `ai-hidden-b` (prefill) is unchanged in shape but must go through the executor.
- Ordering facts relied on: one PeerJS `DataConnection` per pair with `reliable:true` → one ordered channel per hop; the host only issues `ai-rollback{afterLap:N}` after `ai-lapret{lap:N}` arrived, which means every peer has already completed lap N (the return travels through the whole chain). A peer forwards laps in executor order or not at all, so along the chain a stale lap always precedes its correction.

Why no `cancel` type: the rollback *is* the cancel; the paper's "forward empty tensors" is unnecessary because ids + the host broadcast make ordering explicit.

Multi-queue (v2, only if measured rollback delay behind payloads matters): a second "control" DataChannel does not remove head-of-line blocking on Chrome (RFC 8260 interleaving is off there); it lets the rollback overtake queued messages only, so the bound is one in-flight ~16 KiB PeerJS chunk (~13 ms at 10 Mbit/s). Not in v1.

### 4.2 Peer executor and handlers (`p2p.html`)

```js
// ---- per-device lap executor (peers only). One queue, laps applied in id order. ----
const px = {
  q: Promise.resolve(),   // serialized async executor
  epoch: 0,               // laps with msg.epoch < epoch are cancelled
  restoredEpoch: 0,       // highest epoch whose restore has been applied
  lastDone: -1,           // highest lap id whose state update finished on this device
  pending: null,          // rollback waiting for lastDone >= afterLap (defensive; see note)
};
function pxReset() { px.q = Promise.resolve(); px.epoch = 0; px.restoredEpoch = 0; px.lastDone = -1; px.pending = null; }
function pxEnqueue(fn) {
  px.q = px.q.then(fn).catch((e) => { aiStatus("lap executor: " + (e?.message || e)); sendTo(ai.hostId, { t: "ai-error", message: String(e?.message || e) }); });
  return px.q;
}
const slotOf = (lap, k) => (lap & 1) * (ai.engine.NC || 4) + k;

function pxApplyRestore(r) {              // r = {afterLap, k, epoch}; idempotent per epoch
  if (!r || r.epoch <= px.restoredEpoch) return;
  px.epoch = Math.max(px.epoch, r.epoch);   // synchronously: a running lap probes this between layer groups
  pxEnqueue(async () => {
    if (r.epoch <= px.restoredEpoch) return;
    // In the linear chain lastDone >= afterLap always holds here (the host settles N only after ret(N)
    // came back through every peer). Kept as a guard for non-chain topologies.
    if (px.lastDone < r.afterLap) { px.pending = r; return; }
    ai.engine.restoreDN(slotOf(r.afterLap, r.k));
    px.restoredEpoch = r.epoch;
  });
}

async function pxRunLap(d) {
  if (d.epoch < px.epoch) return;                       // cancelled while queued: drop
  const xs = unpackWire(d), wdim = ai.engine.dims.dim, n = d.n;
  const snap = { base: d.slotBase, total: d.slotBase + n + 1 };   // snapshot ALL n columns
  const hb = await ai.engine.runHiddenBatch(xs, d.basePos, snap, () => d.epoch < px.epoch, 4);
  if (!hb) return;                                      // aborted between layer groups; restore is queued or rides on the corrected lap
  px.lastDone = d.lap;
  if (px.pending && px.lastDone >= px.pending.afterLap) { const r = px.pending; px.pending = null; ai.engine.restoreDN(slotOf(r.afterLap, r.k)); px.restoredEpoch = r.epoch; }
  if (d.epoch < px.epoch) return;                       // finished but cancelled: don't forward (saves downstream work)
  if (badF32(hb)) { aiStatus("\u26a0 NaN in lap " + d.lap); sendTo(ai.hostId, { t: "ai-error", message: "NaN in lap " + d.lap }); }
  const out = { lap: d.lap, epoch: d.epoch, parent: d.parent, basePos: d.basePos, n, slotBase: d.slotBase, restore: d.restore, ...packWire(hb) };
  if (ai.next === "host") sendTo(ai.hostId, { t: "ai-lapret", ...out });
  else sendTo(ai.next, { t: "ai-lap", ...out });
}

// in aiOnData(from, d):
case "ai-reset": pxReset(); /* existing reset */ break;
case "ai-hidden-b":   // prefill: unchanged math, now serialized
  if (!ai.engine) return;
  pxEnqueue(async () => { /* existing body of the ai-hidden-b case, verbatim */ });
  break;
case "ai-lap":
  if (!ai.engine) return;
  pxApplyRestore(d.restore);                 // sequenced BEFORE this lap in the queue
  if (d.epoch < px.epoch) break;             // stale before it started
  pxEnqueue(() => pxRunLap(d));
  break;
case "ai-rollback":
  pxApplyRestore({ afterLap: d.afterLap, k: d.k, epoch: d.epoch });
  break;
```

Probe granularity: one layer group (default 4 groups per lap → waste ≤ ¼ of a peer's lap on the device that was mid-lap; downstream peers drop the lap outright). A started group always completes.

Arrival-order cases at a peer (all handled):
- rollback before the stale lap: stale lap dropped at receipt; restore applied (lastDone ≥ afterLap).
- rollback mid-stale-lap: aborted at the next group boundary; restore queued behind it; corrected lap queued behind the restore.
- corrected lap (with `restore`) before the direct rollback: `pxApplyRestore` runs from the lap handler, raising `epoch` and queuing the restore ahead of the corrected lap; the late direct rollback is a no-op (`epoch <= restoredEpoch`).
- stale lap completed everywhere: its `ai-lapret` finds no waiter on the host (deleted at discard) and is ignored.

### 4.3 Host transport glue (`p2p.html`)

```js
// waiters keyed by lap id. Discarded laps delete their waiter; a late return is ignored.
const lapIO = {
  sendLap(hdr, hb) {
    const key = "lap" + hdr.lap;
    const p = new Promise((res, rej) => {
      ai.waiters.set(key, res);
      setTimeout(() => { if (ai.waiters.delete(key)) rej(new Error("pipeline timeout (lap " + hdr.lap + ")")); }, 90000);
    });
    p.catch(() => {});                                   // a discarded lap's promise is never awaited
    sendTo(ai.chain[0], { t: "ai-lap", ...hdr, ...packWire(hb) });
    return p;
  },
  discard(lapId) { ai.waiters.delete("lap" + lapId); },
  rollback(r) { for (const id of ai.chain) sendTo(id, { t: "ai-rollback", ...r }); },
};

// in aiOnData:
case "ai-lapret": {
  const w = ai.waiters.get("lap" + d.lap);
  if (w) { ai.waiters.delete("lap" + d.lap); w(unpackWire(d)); }
  break;
}
```

Also change `ai-hiddenret-b` to carry and key by `lap` once prefill is moved onto lap ids (optional; prefill is strictly sequential today so `"b"+basePos` cannot collide there).

Capability negotiation: the host runs continuous mode only if every peer's `ai-ready` advertises `caps: { lap: 1 }`; otherwise it falls back to today's `specStep` loop. **UNCERTAIN**: what `ai-ready` currently carries — add the field.

---

## 5. Host state machine

Two phases per lap: `issue(lap)` (draft, host layer share, send) and `settle(lap)` (wait return, head, sample, verdict, rollback/refill). Ring `inflight` holds at most L = 2 laps, oldest first. The host never blocks on a return while it has issue work available.

```
                 ┌──────────────────────────────────────────────────────────┐
                 │ IDLE: inflight = []                                        │
                 │   issueFresh(next)  [drafts from exact this.x]             │
                 └───────────────┬──────────────────────────────────────────┘
                                 v
      ┌────────────────── ONE_IN_FLIGHT: inflight = [N] ─────────────────────┐
      │ gate(N): draft d_{K+1} from xChain; pp = Π p(N.drafts) · p(d_{K+1})   │
      │   pp >= theta  -> issueChild(N)  -> TWO_IN_FLIGHT                     │
      │   else         -> settle(N)                                            │
      └───────────────┬────────────────────────────────┬──────────────────────┘
                      v                                v
   TWO_IN_FLIGHT: inflight=[N, N+1]           settle(N) with no child:
     settle(N):                                 a<K : restore slot(N,a), rollback; next=out[a]  -> IDLE
       a==K && out[K]==N+1.tok[0]               a==K: next=out[K]                              -> IDLE
         -> emit K+1, refill 1..K+1,
            inflight=[N+1]              -> ONE_IN_FLIGHT (chain continues via xChain)
       else -> restore slot(N,a), epoch++, rollback, discard N+1,
               emit a+1, refill 1..a, setHidden(hs[a]), next=out[a]   -> IDLE
```

Token emission: a lap's tokens are emitted only when it settles, except the child's canonical token (`child.tokens[0]`), which equals the parent's `out[K]` and is emitted at the parent's settle. The canonical column is never rejected (PipeInfer invariant (i)); a lap is valid-or-cancelled as a unit.

### 5.1 Complete host code (`engine/specpipe.js`)

```js
// Continuous speculation driver. Requires engine methods from Sec 3:
//   mtpRun(null, tok, pos, "argmaxp") -> {id, p}; saveChain(); loadChain(); slotOf(lap,k); restoreDN(slot);
//   embedRunBatch(ids, pos, {base,total}); headBatch(hs, n); setHidden(h)
// io: { sendLap(hdr, hb) -> Promise<Float32Array>, discard(lapId), rollback({afterLap,k,epoch}) }
export class SpecPipeline {
  constructor(engine, { K = 3, L = 2, sample, io, gate = {}, log = null }) {
    if (K + 1 !== engine.NC) throw new Error(`continuous mode needs K+1 == NC (K=${K}, NC=${engine.NC})`);
    if (engine.snapSlots < L * (K + 1)) throw new Error(`need ${L * (K + 1)} snapshot slots, have ${engine.snapSlots}`);
    this.E = engine; this.K = K; this.L = L; this.sample = sample; this.io = io; this.log = log;
    // PipeInfer IV.B.2: cutoff theta, recovery added per consecutive gamble, decay when a gamble is skipped
    this.gate = { theta0: 0.35, thetaMin: 0.15, thetaMax: 0.90, recovery: 0.10, decay: 0.05, ...gate };
    this.theta = this.gate.theta0;
    this.lapId = 0; this.epoch = 0; this.inflight = []; this.pendingRestore = null;
    this.stats = { laps: 0, drafts: 0, accepted: 0, fullAccept: 0, bonusMatch: 0, gambled: 0, gambleWon: 0, gateSkipped: 0,
                   t: { draft: 0, trunk: 0, wait: 0, head: 0, refill: 0 } };
    this.calib = [];   // [p_draft, accepted(0/1)] pairs for gate calibration
  }
  get pos() { return this.E.pos; }

  // ---- issue ----------------------------------------------------------------
  async _draft(tok, pos) {
    const t0 = performance.now();
    const r = await this.E.mtpRun(null, tok, pos, "argmaxp");
    this.stats.t.draft += performance.now() - t0;
    return r;
  }
  // lap.tokens[0] is set; draft K more, run host layers, ship. Precondition: this.x holds the hidden to draft from.
  async _issue(lap) {
    const E = this.E, K = this.K;
    for (let k = 0; k < K; k++) {
      const { id, p } = await this._draft(lap.tokens[k], lap.basePos + k);
      lap.tokens.push(id); lap.p.push(p);
    }
    E.saveChain();                                        // MTP output after d_K: where a child lap continues from
    const t0 = performance.now();
    const hb = await E.embedRunBatch(lap.tokens, lap.basePos, { base: lap.slotBase, total: lap.slotBase + K + 2 });
    this.stats.t.trunk += performance.now() - t0;
    if (hb.some((v) => !Number.isFinite(v))) throw new Error(`NaN after HOST layers (lap ${lap.id})`);
    lap.ret = this.io.sendLap({ lap: lap.id, epoch: lap.epoch, parent: lap.parent, basePos: lap.basePos, n: K + 1,
                                slotBase: lap.slotBase, restore: this.pendingRestore }, hb);
    this.pendingRestore = null;
    this.inflight.push(lap);
    this.stats.laps++;
    return lap;
  }
  _newLap(tok0, basePos, parent) {
    const id = this.lapId++;
    return { id, epoch: this.epoch, parent: parent ? parent.id : -1, basePos, tokens: [tok0], p: [],
             slotBase: this.E.slotOf(id, 0), ret: null, gambled: !!parent, tIssue: performance.now() };
  }
  async issueFresh(next) {               // this.x = exact trunk hidden of pos-1 (setHidden), next = sampled token at pos
    return this._issue(this._newLap(next, this.E.pos, null));
  }
  // Gamble: continue the parent's chain. Returns null if the gate refuses.
  async issueChild(parent) {
    const E = this.E, K = this.K;
    E.loadChain();                                        // MTP output after the parent's d_K
    const first = await this._draft(parent.tokens[K], parent.basePos + K);   // d_{K+1}: stand-in for the parent's bonus
    const pp = parent.p.reduce((a, b) => a * b, 1) * first.p;                // the K+1 events the gamble depends on
    if (pp < this.theta) {
      this.stats.gateSkipped++;
      if (!parent.settled) this.theta = Math.max(this.gate.thetaMin, this.theta - this.gate.decay);   // "no logits waiting" -> back off
      return null;
    }
    this.theta = Math.min(this.gate.thetaMax, this.theta + this.gate.recovery);   // successive gambles must be more confident
    const lap = this._newLap(first.id, parent.basePos + K + 1, parent);
    lap.p0 = first.p;
    this.stats.gambled++;
    return this._issue(lap);
  }

  // ---- settle ---------------------------------------------------------------
  // Returns { tokens: emitted, next: token to start a fresh lap with (null if the child took over) }
  async settle() {
    const E = this.E, K = this.K, { dim } = E.dims;
    const lap = this.inflight[0], child = this.inflight[1] || null;
    const t0 = performance.now();
    const hs = await lap.ret;                              // n*dim f32 from the last peer; rejects on timeout
    this.stats.t.wait += performance.now() - t0;
    if (hs.some((v) => !Number.isFinite(v))) throw new Error(`NaN in hidden returned by peers (lap ${lap.id})`);
    const t1 = performance.now();
    const lgs = await E.headBatch(hs, K + 1);
    const out = []; let a = 0;
    for (let k = 0; k <= K; k++) {
      const t = this.sample(lgs[k]); out.push(t);
      if (k < K && t === lap.tokens[k + 1]) a++; else break;
    }
    this.stats.t.head += performance.now() - t1;
    this.stats.drafts += K; this.stats.accepted += a;
    for (let k = 0; k < K; k++) { this.calib.push([lap.p[k], k < a ? 1 : 0]); if (k >= a) break; }
    if (a === K) this.stats.fullAccept++;
    const bonusOk = a === K && child && out[K] === child.tokens[0];
    if (child) { this.calib.push([child.p0, bonusOk ? 1 : 0]); if (bonusOk) { this.stats.bonusMatch++; this.stats.gambleWon++; } }
    lap.settled = true;
    this.inflight.shift();

    const lost = child && !bonusOk;
    if (lost || (!child && a < K)) {
      // rollback: truth = state after column a of this lap; everything issued after it is cancelled
      if (child) { this.inflight.shift(); this.io.discard(child.id); }
      this.epoch++;
      const r = { afterLap: lap.id, k: a, epoch: this.epoch };
      E.restoreDN(E.slotOf(lap.id, a));                    // host's own layer share
      this.io.rollback(r);
      this.pendingRestore = r;                             // rides redundantly on the next lap
      if (lost) this.theta = this.gate.theta0;
    }
    // refill the MTP cache at accepted positions with exact trunk hiddens (clobbers this.x; xChain is safe)
    const t2 = performance.now();
    const nRefill = bonusOk ? K + 1 : a;
    for (let j = 1; j <= nRefill; j++) {
      E.setHidden(hs.subarray((j - 1) * dim, j * dim));
      await E.mtpRun(null, out[j - 1], lap.basePos + j, false);
    }
    this.stats.t.refill += performance.now() - t2;
    if (bonusOk) {
      this.theta = this.gate.theta0;                       // PipeInfer: reset cutoff on an accepted completed run
      E.pos = child.basePos;                               // == lap.basePos + K + 1
      return { tokens: out, next: null };                  // K+1 tokens; child is now the head of the ring
    }
    E.setHidden(hs.subarray(a * dim, (a + 1) * dim));      // chain restarts from the exact hidden
    E.pos = lap.basePos + a + 1;
    return { tokens: out.slice(0, a + 1), next: out[a] };
  }

  // ---- main loop --------------------------------------------------------------
  // emit(tok) -> false to stop (EOS / length). Precondition as specStep: this.x = trunk hidden of pos-1, `next` sampled for pos.
  async run(next, emit) {
    let stop = false;
    while (!stop) {
      if (this.inflight.length === 0) { await this.issueFresh(next); continue; }
      if (this.inflight.length < this.L) {
        const head = this.inflight[this.inflight.length - 1];
        if (!head.gambledFrom && !head.childTried) { head.childTried = true; if (await this.issueChild(head)) continue; }
      }
      const r = await this.settle();
      for (const tk of r.tokens) if (!emit(tk)) { stop = true; break; }
      next = r.next;
      if (!stop && next === null && this.inflight.length === 0) throw new Error("state machine: child promised but ring empty");
    }
    // stop: cancel whatever is in flight so peers stop burning GPU on it
    if (this.inflight.length) {
      const last = this.inflight[0];
      for (const l of this.inflight) this.io.discard(l.id);
      this.inflight = [];
      this.epoch++;
      this.io.rollback({ afterLap: last.parent >= 0 ? last.parent : last.id, k: 0, epoch: this.epoch });   // harmless: next generation resets
    }
  }
}
```

Notes on the loop:
- `issueChild` is attempted once per head lap (`childTried`). If the gate refuses, the loop settles the head; when the head settles with `a == K` and no child, the next lap is a fresh one from the exact hidden (`next = out[K]`).
- With L = 2, `settle()` always sees at most one child. `child.tokens[0]` is `d_{K+1}`, drafted from the parent's chain, so `out[K] === child.tokens[0]` is exactly "bonus matches the (K+1)-th chained draft".
- Under greedy sampling the emitted sequence must be bit-identical to today's sequential `specStep` loop: accepted tokens are always target tokens, DN restore is a byte copy, KV entries are position-indexed. That is the primary correctness test (Sec 9).
- The time from `ret(N)` arrival to `settle(N)` start can be delayed by up to one `issueChild` (≈ (K+1)·t_d + host trunk) if the return lands while the host is issuing. This is the host-side cancellation waste δ_host in Sec 8; it is zero whenever `(K+1)·t_d + t_ht < T_lap − t_ht`.

### 5.2 Host generation glue (`p2p.html`, replaces the `specStep` while-loop when continuous mode is on)

```js
if (ai.chain.length && ai.continuous && ai.engine.mtp) {
  const { SpecPipeline } = await import("./engine/specpipe.js");
  if (ai.lastHidden) ai.engine.setHidden(ai.lastHidden);
  ai.engine.pos = ai.pos;
  const pipe = new SpecPipeline(ai.engine, { K: 3, L: 2, sample: aiSample, io: lapIO });
  ai.pipe = pipe;
  let next = aiSample(logits);
  await pipe.run(next, (tk) => {
    if (tk === imEnd || tk === eot || count >= 400) return false;
    emit(tk); return true;
  });
  ai.pos = ai.engine.pos;
  aiStatus(`gambled ${pipe.stats.gambled}, won ${pipe.stats.gambleWon}, full-accept ${pipe.stats.fullAccept}/${pipe.stats.laps}`);
}
```

Solo mode and rooms without lap-capable peers keep the existing `specStep` + K ladder.

---

## 6. Cancellation — summary of guarantees

| Property | Mechanism |
|---|---|
| A stale lap never corrupts a device's state after the restore | restore is queued after the running lap (executor); slot copy is total for all DN layers; KV/MTP-KV beyond the frontier is never read |
| A stale return never resolves the wrong promise | waiters keyed by lap id; discarded before the rollback is sent |
| A rollback is applied exactly once per epoch, in order, regardless of arrival path | `restoredEpoch`, `pxApplyRestore` on both `ai-rollback` and `ai-lap.restore` |
| Wasted GPU work per cancelled lap per device | ≤ one layer group (default ¼ lap) on the device that was mid-lap; zero on downstream devices |
| The canonical column is never cancelled mid-lap | a lap is atomic: either dropped before start, aborted at a group boundary (then restored), or completed |
| No partial acceptance of a cancelled lap | tokens are emitted only at settle |

Not implemented (deliberately): partial-lap cancellation inside a layer group, the paper's "empty tensor" stubs, a separate cancel message type, KV copy-on-accept (slot restore replaces it).

---

## 7. Gamble condition and gate

Gamble event: `V(N) = [a(N) == K] ∧ [out_N[K] == d_{K+1}]`. Both are acceptance events of chained drafts; `d_{K+1}` is at chain depth K+1 from the last exact hidden.

Gate (PipeInfer IV.B.2 adapted): launch the child iff `Π_{k=1..K} p_k(N) · p(d_{K+1}) ≥ θ`, where `p_k` is the MTP head's probability of its own argmax (Sec 3.3). θ starts at θ₀, gains `recovery` per consecutive gamble, is reset to θ₀ when a completed lap is accepted, and loses `decay` when a gamble is skipped while `ret(N)` has not arrived. Defaults θ₀ = 0.35, recovery 0.10, decay 0.05, [0.15, 0.90] — **UNCERTAIN**, placeholders until calibrated.

Two-stage gating to avoid wasting a draft: stage A on `Π p_k(N)` alone (free, known before `d_{K+1}`); stage B after drafting `d_{K+1}`. A stage-A failure costs nothing; a stage-B failure wastes one draft (~30 ms) — but `d_{K+1}` is still a useful draft if the parent full-accepts (it is then `next` for the fresh lap only if it matches; today's code would redraft it from the exact hidden, which is better), so treat the 30 ms as pure gate cost.

Expected-gain criterion (preferred once instrumented): gamble iff `q̂(pp) · (T_seq − Δ_s) > (1 − q̂(pp)) · δ`, with `q̂` from the `calib` pairs, `T_seq`, `Δ_s`, `δ` from the `stats.t` timers. With layer-group probing δ is small, so this criterion accepts most gambles; the product gate mainly protects against burning host GPU time on hopeless chains.

Sampling: `aiSample` is temp 0.8 / top-k 40, so the bonus match is a sampled event; per-draft acceptance under sampling is below the greedy bench figures. Optional: use greedy at column K when `p(d_{K+1})` is high (raises the bonus-match rate at a small sampling-fidelity cost). Not in v1.

---

## 8. Expected gain with our numbers

Symbols (per lap, K = 3):

| symbol | meaning | value |
|---|---|---|
| t_d | one chained draft (MTP block + Q8 LM head + argmax readback) | **30 ms** (given) |
| t_r | one refill run (MTP block only, no head) | **UNCERTAIN**, 10 ms placeholder |
| t_h | `headBatch(4)` + 4 MB logits readback + JS top-k sampling | **UNCERTAIN**, 30 ms placeholder |
| t_ht | host's own trunk share inside the lap | **UNCERTAIN**, 65 ms placeholder |
| T_lap | measured `ai.lapMs` (host trunk + hops + peers) | **300–600 ms** (given) |
| s_max | slowest peer stage incl. its hop | **UNCERTAIN**; ≈ (T_lap − t_ht)/2 for a 2-peer chain |
| δ | extra delay before the corrected lap gets the pipeline on a lost gamble | ≈ ¼ s_max average with 4-group probing (**30 ms**); ≈ s_max without cancellation |
| p | per-draft acceptance | 0.85 (given; two readings below) |

Two readings of p = 0.85 (the bench-log figure is `accepted/drafts` averaged per lap, i.e. a mean prefix fraction, not a geometric per-token probability):

- **A (geometric, p = 0.85):** q = P(V) = p^(K+1) = 0.85⁴ = **0.52**; E_tok/lap = (1−p⁴)/(1−p) = 3.19.
- **B (0.85 = mean prefix fraction ⇒ fitted geometric p ≈ 0.92):** q ≈ 0.92⁴ = **0.72**; E_tok/lap = 1 + 0.85·3 = 3.55.

Both treat the bonus match as the (K+1)-th acceptance event (it is, at chain depth K+1) and ignore sampling temperature and the deeper chain of child laps (depths K+2..2K+1), which push q down. Measure `fullAccept/laps` and `bonusMatch/gambled` before trusting either.

Times:
- Sequential lap (today): `T_seq = K·t_d + T_lap + t_h + a·t_r ≈ 90 + T_lap + 30 + 25 = T_lap + 145`.
- Continuous, host stage per lap: `T_host = (K+1)·t_d + t_ht + t_h + (K+1)·t_r = 120 + 65 + 30 + 40 = 255 ms`.
- Success spacing: `Δ_s = max(T_host, s_max)`; failure spacing: `T_seq + δ`.

Renewal formula (tokens per completed lap are unchanged by the gamble, so the speedup is a pure time ratio):

```
S = T_seq / [ q·Δ_s + (1−q)·(T_seq + δ) ]
```

| T_lap | T_seq | s_max | Δ_s | S, q=0.52 (A) | S, q=0.72 (B) | S no-cancel (δ=s_max), A |
|---|---|---|---|---|---|---|
| 300 | 445 | ~118 | 255 | **1.23x** | **1.41x** | 1.10x |
| 450 | 595 | ~193 | 255 | **1.38x** | **1.66x** | 1.16x |
| 600 | 745 | ~268 | 268 | **1.46x** | **1.82x** | 1.19x |

Worked example, T_lap = 300, A: denominator = 0.52·255 + 0.48·(445+30) = 132.6 + 228 = 360.6 → S = 445/360.6 = 1.23. Absolute: sequential 3.19/0.445 = 7.2 tok/s → 8.8 tok/s.

Observations:
- At T_lap = 300 the ceiling is set by the host's own GPU work (T_host = 255 vs T_seq = 445 → ≤ 1.75x at q = 1); the gain grows with lap time because the host stage stays fixed while the sequential lap grows. This matches the paper's "most benefit on slow interconnects / marginal in shallow pipelines".
- Cancellation is worth ~0.15–0.3x of the gain; the δ_host term (return landing mid-issue) adds up to (K+1)·t_d + t_ht ≈ 185 ms on a lost gamble only when T_lap − t_ht < 185, i.e. not in the 300–600 ms regime.
- Comparison to today's K=7 ladder (chosen when lap > 260 ms): sequential K=7 yields 6.0 tok/lap but its 8-column lap costs two chunks on every peer plus 7 drafts; at T_lap(4-col) = 450 a rough estimate is 6.0 / (210 + ~600 + 60 + 50) ≈ 6.5 tok/s vs continuous K=3 at 7.4 (A) / 9.9 (B) tok/s. **UNCERTAIN** (8-column lap time not measured); keep the ladder selectable and let the existing tok/s-based picker decide between `{K=3 continuous, K=5, K=7 sequential}`.
- L = 3 would add a `q²`-weighted term (0.27 / 0.52) for 4 more slots; skip until L = 2 is measured.

Instrumentation to close the UNCERTAIN inputs: `stats.t.{draft,trunk,wait,head,refill}` per lap, `fullAccept/laps`, `bonusMatch/gambled`, per-hop timing from `ai-lapret` (add `tPeer` per hop to the header), and the `calib` pairs. Compute `q` online and feed the expected-gain gate.

---

## 9. Testing plan

1. **Greedy equivalence (must pass first).** Solo Deno test (`test_mtp_deno.js`-style): run the same prompt with (a) today's `specStep` loop, (b) `SpecPipeline` with a loopback `io` whose `sendLap` runs `runHiddenBatch` on a second engine instance holding the remaining layers (or the same engine with `layerRange` split) plus an artificial delay, and a `rollback` that calls `restoreDN(slotOf(...))` on it. Outputs must be bit-identical under argmax sampling. Vary the delay so returns land before/after `issueChild`.
2. **Slot map.** After a lost gamble, compare the peer's live `S` against a CPU-tracked copy of slot `(N, a)` (add a debug readback path). Also assert `snapSlots ≥ L·(K+1)` at pipeline construction.
3. **Ordering fuzz (LAN, 3 tabs).** Inject random delays into `sendTo` for `ai-rollback` vs `ai-lap` so peers ≥ 2 see every arrival order; assert `restoredEpoch` monotone and greedy output identical to sequential.
4. **Cancellation waste.** Log group index at abort; check ≤ 1 group of waste per cancelled lap per device.
5. **Perf.** Record `ai.lapMs`, `stats.t`, `q`, and tok/s for `{sequential K=3, sequential K=7, continuous K=3}` on the same room; compare against the table in Sec 8.
6. **Safari peer.** No new WGSL runs on peers; verify the executor and `onSubmittedWorkDone` probing on Safari 26.

---

## 10. UNCERTAIN / to verify before relying on it

- `this.x` buffer usage flags include `COPY_SRC` (needed for `saveChain`).
- Whether `_bg` uses `layout:"auto"` (drives whether `argmax_lse` needs its own bind group — spec assumes yes).
- Cost of `onSubmittedWorkDone()` probe points per layer group on Chrome/Safari peers (choose `groups` by measurement; 1 disables probing).
- Cost of the second `exp` sweep in `argmax_lse` on the 993 KB logits buffer.
- All placeholder timings: t_r, t_h, t_ht, s_max; the 8-column lap time for the K=7 comparison.
- q: neither reading of p = 0.85 is a measured `P(a == K)`; the bonus-match rate under temp 0.8 / top-k 40 is unmeasured; child laps draft at chain depths K+2..2K+1 and will accept less often than the bench numbers.
- Gate constants θ₀/recovery/decay and the `p_draft → P(accept)` calibration.
- What `ai-ready` carries today (needed for the `caps.lap` negotiation) and how `ai-reset` interacts with `pxReset` on mid-generation reconnects (spec: abort generation on any peer timeout, as today).
- PeerJS `send()` of a ~40 KB message chunks at ~16 KB; the per-hop serialization cost of the extra header fields is negligible but unmeasured.

Files touched: `engine/qwen35.js` (Sec 3), new `engine/specpipe.js` (Sec 5.1), `p2p.html` (Sec 4.2, 4.3, 5.2). Sequential `specStep`, solo prefill, and all kernels other than the additive `argmax_lse` are unchanged.

---

# Spec: 8-wide prefill (assessment + exact change) and 2-pass GPU argmax for the 248320-logit LM head

Scope: SwarmLLM engine (`engine/qwen35.js`, `engine.js`), Qwen3.8-27B Q4_0 on GB10 (Deno/wgpu/Vulkan) with Chrome + Safari 26 portability. Everything below is drawn from the verified findings and the engine code as it exists today; anything not directly measured is marked UNCERTAIN.

---

## Part A — 8-wide prefill: is it worth it?

### A.1 Verdict

**An 8-wide f32 batched pass is NOT worth it as a standalone change. Do not build it.** It already exists (env `BCOLS=8`), it was measured, and it gave no prefill gain. Widening only pays when it is done with DP4a (`dot4I8Packed` on Q8_1-quantized activations) AND the serial DeltaNet cost per column is cut first — otherwise the wider GEMV is masked by non-GEMV per-column work.

### A.2 Evidence

1. **Already implemented and measured in the real engine.** `coopWGSL(WG, ROWS, 64, COLS, ROWSB)` (engine.js:411) takes `COLS` from `batchCols`; `test_batch_q38_deno.js`, `test_mtp_deno.js` and `bench_deno.js` read `BCOLS`/`ROWSB` from env. `docs/bench-log.md` (Sep 1 row) records the 86-token prompt result: **27.3 tok/s (4-wide) vs 26.1 (8w x 2 rows) vs 27.9 (8w x 4 rows) — "no gain: 8-col pass ≈ 4-col pass ⇒ prefill is bound by the serial DeltaNet recurrence, not GEMV"**.

2. **Microbench on the f32 kernel family** (bench_ncols.js, reproduced by the verifier on the GB10, WGB=64, ROWS=4): relative to one single-column GEMV, 4 columns cost 1.09–1.17x (~3.5 cols per single-time), 8 columns cost 1.62–1.78x (~4.5–4.9 cols), 16 columns 2.6–2.9x in "stream" order (~5.5–6.2 cols). The f32 kernel flips from bandwidth-bound (N<=4) to L1-load + FMA-issue bound (N>=8): effective throughput saturates at ~1.8–2.5 TFLOP/s of dequant-FMA regardless of N. Conversion-free "magic number" nibble decode gave exactly 0%. So the f32 ceiling for going 4->8 is roughly –25–30% on the **matvec share only**, and the real engine did not even realise that.

3. **Batched-pass breakdown** (bench_breakdown_deno.js): 4-column batched pass 125–134 ms = batched matvecs ~95–97 ms + dn_delta_mc ~7–8.5 ms + rmsnorm/add_res ~3.4 + attention ~3.1 + dn gates/l2/gatenorm ~2.6 + ~20–37 ms head/readback/copies. Prefill (no head, no readback) is ~101.5 ms per 4 columns = ~95 matvec + ~6–8 other, dominated by dn_delta_mc. Every non-matvec family scales per column (the DN kernels loop columns sequentially inside one dispatch with nVH=48 workgroups).

4. **UNCERTAIN — unresolved discrepancy.** The microbench predicts an 8-wide f32 prefill pass at roughly 95/1.14 x 1.75 ≈ 146 ms matvec + ~13–16 ms other = ~160 ms per 8 columns (~20 ms/token, ~+25% vs 4-wide), but bench-log measured 0%. The bench-log attributes it to the serial DeltaNet recurrence; the subtractive bench cannot rank the other non-matvec families (skipping them makes the pass slower, likely NaN/denormal propagation), so the true per-column cost of non-matvec work at N=8 is not cleanly measured. Treat "8-wide f32 = no gain" as the measured fact and the +25% as an unrealised microbench ceiling.

5. **Verify is unaffected by this decision today.** The measured MTP configuration is K=3, one 4-column pass (16.0 vs 9.1 tok/s, 85% acceptance). K=7 verify (8 columns in 2 chunks of 4 via `verifyN`, qwen35.js:1223-1239) is a mode that exists but was not the measured one; the "8-wide verify at 1.25x single" figure is a projection from the DP4a microbench, not a measurement.

### A.3 Exact change — f32 path (if you want NC=8 anyway, e.g. to halve P2P prefill rounds)

No kernel change. The following is the complete list of what NC=8 touches, for the record:

- Construct with `batchCols: 8, coopRowsB: 4` (`_init`, qwen35.js:470-474). `coopWGSL(coopWG, coopRows, 64, 8, 4)` generates `matvec_{q4,q8}_coop_b` and `_gu_b` with COLS=8 (hold order: all 8 x-slices in registers). Keep WGB=64 (128/256 are 10–40% slower for hold n<=8; up to 1.5–1.8x slower for hold n16).
- `_initBatch` (qwen35.js:915-960): every `mkB` buffer doubles (small), `stageXB` = 8 x 20 KB, `stageLogitsN` = 8 x 993 KB = 7.9 MB per verify readback (this is the cost Part B removes). `S_shadow`/`conv_shadow` stay at 7 slots (K<=7, so slots 0..6 for the 7 non-final columns — unchanged).
- `prefillTokens` (1289-1316) consumes 8 ids per pass automatically; the tail path `prefillToken` is unchanged. `verifyN` with K=7 becomes one `embedRunBatch` + one `headBatch` (n<=NC).
- P2P (`p2p.html`): the hidden-batch messages carry `n` columns; a 16-token network round becomes 2 chunks of 8 instead of 4 of 4. Per bench-log this does not change tok/s on the GB10 host; on a dispatch-bound Mac peer it halves dispatch count per round (UNCERTAIN — not measured on a Mac).
- If you go to N=16 on f32, switch the batched body to "stream" order (cols outer, decoded rows held, scale folded into the decoded weights): 1.21 vs 1.70 ms at 17408x5120 (1.4x better than hold at N=16). Do NOT use stream order for N<=8 (tie at 8, hold wins at 4).

### A.4 Exact change — the path that IS worth it (DP4a 8/16-wide)

Do these in this order; (0) is the prerequisite that the bench-log result says gates everything else.

**(0) dn_delta_mc register-resident state** (qwen35.js:295-333). Today per column it reads `dlm_s[idx]`, writes the decayed value, reads/writes again for the rank-1 update, and copies to `dlm_shadow` on snapshot columns — ~4xS (6xS on snapshot columns) through the cache hierarchy per column per layer, 48 WGs x 128 threads. Rewrite: load the thread's 128-float S column into a constant-bound `var<private> state: array<f32,128>` once, stage k/q per column in `var<workgroup>`, loop columns, write S once at the end and snapshots from registers (llama.cpp `gated_delta_net.wgsl` structure; same 48-WG geometry). Expected 7–8.5 ms -> ~2–3 ms per 4-col pass is an **UNCERTAIN estimate** (no measurement; whether Tint/naga keeps a 128-float private array in registers is a compiler expectation, verify by A/B). Must be bit-exact vs the current kernel (same per-element op order is achievable). Re-run bench_breakdown_deno.js and `BCOLS=8` prefill after this; if 8-wide f32 still shows no gain, the bottleneck is elsewhere and DP4a widening will not help prefill either.

**(1) Activation quantization to Q8_1, fused into `rmsnorm_mc`'s epilogue** (the workgroup already holds the normalized 5120-float row). Per 32-block: `d = amax/127`, `qs` = 32 int8 packed 4/u32 (8 u32, same element order as the f32 x so lo nibbles pair with words `qt`, hi nibbles with `qt+4`), `s = d * sum(q)`. Output buffers per column: `xq_q: array<u32>` (dIn/4 words) and `xq_ds: array<f32>` (2 per block). Also needed for the SiLU output feeding the down projection -> add the same epilogue to `silu_mul`/`sigmoid_mul_mc` (or a tiny standalone quantize dispatch if the fusion is awkward; +1 dispatch per layer).

**(2) Kernels `matvec_{q4,q8}_coop_b8`, `_b16`, `_gu_b8`** generated from `coopWGSL` with the same lane layout as `_coop_b` (WGB=64, ROWSB=4, qt = t&3 quarter split, hold order for N=8, stream order for N=16). Inner loop (verified against llama.cpp `mul_mat_vec_q_acc.tmpl`; `alo/ahi` must be u32):

```wgsl
// per column m, per block b, quarter qt (loaded once per iteration)
let xq0a = xq_q[m*xqs + b*8u + qt];  let xq0b = xq_q[m*xqs + b*8u + qt + 4u];
let ds0  = vec2<f32>(xq_ds[(m*nb + b)*2u], xq_ds[(m*nb + b)*2u + 1u]);   // (d, d*sum q)
// per row r (word decoded once, reused by all columns)
let word = q4_qs[wIdx + r*rowWords];
let alo = word & 0x0F0F0F0Fu;  let ahi = (word >> 4u) & 0x0F0F0F0Fu;
let s = q4s(scBase + r*nb);
// sum((q-8)*x) = d*(sum q*xq) - 8*d*sum(xq); this thread owns 8 of 32 -> 8*s*ds.y/4 = 2*ds.y
a_r_m += s * (f32(dot4I8Packed(alo, xq0a) + dot4I8Packed(ahi, xq0b)) * ds0.x - 2.0 * ds0.y);
```
Q8_0 weights: `alo/ahi` are the raw signed words (no mask), no `-8` term: `a += s * f32(dot4I8Packed(w0, xq0a) + dot4I8Packed(w1, xq0b)) * ds0.x`. Accumulation: exact int32 inside a block, f32 across blocks — satisfies the f32-accumulation rule. Keep scalar named accumulators (`a0_0 … a3_7`), never a dynamically indexed local array. Reduction: the existing shared-memory tree (2 rounds per 2 columns for `_coop_b` style, i.e. 4 rounds at N=8), barriers outside conditionals.

**(3) Feature gate + fallback.** `navigator.gpu.wgslLanguageFeatures.has('packed_4x8_integer_dot_product')` (Deno 2.9.5 on the GB10 reports it; Chrome 123+; wgpu has native SPIR-V/HLSL/Metal lowering since v26.0.0). WebKit trunk now advertises it too, so do NOT hard-code "Safari lacks it" — but additionally gate on adapter vendor (llama.cpp excludes Apple because Metal emulates dp4a as scalar mul-adds and measured a slight regression on M2). Selection: DP4a available and vendor in {nvidia, amd, intel} -> NC=8 (or 16) with the `_b8/_b16` kernels; otherwise NC=4 with the existing f32 `_coop_b`. `requires packed_4x8_integer_dot_product;` directive: UNCERTAIN whether naga accepts/requires it — probe-compile with and without, as the plan already does for `enable subgroups;`.

**(4) Tolerance harness before shipping.** Bit-exactness vs the f32 path is gone by design: measured max|diff| 9.5e-3..1.6e-2 on logits of magnitude 2.6–4.2 (~0.4% rel). Extend `test_batch_q38_deno.js`/`test_mtp_deno.js` goldens with a logit tolerance (and compare argmax agreement rate) instead of bit-identity.

**Expected (measured ranges, GPU warm):** 8 DP4a columns cost 1.15–1.35x single (6–7 cols per single-time) vs 1.6–1.8x f32; 16 columns 2.0–2.8x at 17408x5120 but **3.3–3.5x at 5120x17408 (down-proj)** — so N=16 is shape-dependent; N=8 is the safe default. A cold-clock run showed DP4a n16 at 0.99x, so ratios swing with clock state. N=1 decode is unchanged by DP4a (1.02–1.06x) — this is a prefill/verify lever only. Prefill projection with (0)+(2) at N=16: (183 + 4x8)/16 ≈ 13.4 ms/token ≈ 70 tok/s upper bound vs 39 today — **UNCERTAIN**, derived from a single synthetic shape and contradicted in direction by the bench-log's DN-bound finding until (0) lands. Verify at K=7: one 8-column DP4a pass at ~1.25x single instead of 2 chunks at ~2.28–2.38x (projection).

---

## Part B — 2-pass GPU argmax for 248320 floats

### B.1 Why

Today (qwen35.js:335-359) `argmax` is one 256-thread workgroup striding all 248320 logits (970 serial elements per thread, ~1 MB streamed through one SM), used only in the draft chain (`mtpRun(..., "argmax")`, line 1205) with a 16-byte readback. `headBatch` (1163-1181) still copies and maps n x 993 KB of logits per verify (`stageLogitsN`), and the batched pass carries ~20–37 ms of head + copies + readback overhead. The 2-pass kernel (a) spreads pass 1 over 122 workgroups (all 48 SMs) with coalesced 16-byte loads and (b) extends to n columns so greedy verify reads back 8n bytes instead of n x 993 KB.

Constraints honoured: no subgroups, no f16, no pointer params (Safari), f32 only, barriers outside conditionals, scalar `bitcast<u32>(f32)` only (the naga `bitcast<vec2<f16>>` bug is not touched), same bind-group conventions as the existing `argmax` (`G1` table + `bgCommonFor`).

### B.2 Geometry

- vocab n = 248320, n % 4 == 0 -> nvec4 = 62080. The logits buffer (`this.logits`, size 993280 B; `B.logits` stride `al(vocab)` = 993280 B exactly = 62080 vec4) is bound as `array<vec4<f32>>` — same GPUBuffer, no repacking (buffer size is a multiple of 16).
- Pass 1: WG = 256 threads, 2 coalesced vec4 loads per thread (8 logits), 512 vec4 per WG -> nWG = ceil(62080/512) = **122** (last WG has 128 valid vec4, threads 128..255 idle on the second load only). Dispatch (122, nCols). Output: one (value, index) pair per (column, WG) -> `nCols x 122` pairs.
- Pass 2: one 256-thread WG per column reduces its 122 pairs -> `out[col*2] = index`, `out[col*2+1] = bitcast(value)` (same layout as today's `am_out`, so the single-column readback code in `mtpRun` is unchanged).
- Semantics: first occurrence wins on ties (`v > bv || (v == bv && i < bi)`) — identical to the existing kernel and to torch.argmax. Init value −3.402823e38 (not −1e9). NaN logits never win (`>` false); if every element is NaN or −inf the index stays 0xffffffff — host must treat that as an error.

### B.3 WGSL (complete; append to the `WGSL2` string next to the existing `argmax`)

```wgsl
// --- argmax, 2-pass, n columns of am1_p.x*4 floats each ---
// Pass 1: dispatch (nWG, nCols). Each WG scans 512 vec4 (2048 floats) of one column with
// two coalesced vec4 loads per thread and writes one (value,index) pair to am1_part[col*nWG + wg].
@group(1) @binding(0) var<storage, read> am1_x: array<vec4<f32>>;        // logits bound as vec4 (n % 4 == 0)
@group(1) @binding(1) var<storage, read_write> am1_part: array<vec2<u32>>; // .x = bitcast<u32>(value), .y = index
@group(1) @binding(2) var<uniform> am1_p: vec4<u32>;                      // x: nvec4 (= n/4), y: column stride in vec4, z: nWG, w: unused
var<workgroup> am1_v: array<f32, 256>;
var<workgroup> am1_i: array<u32, 256>;

// first-occurrence argmax order: larger value wins, equal value -> lower index wins
fn am_better(v: f32, i: u32, bv: f32, bi: u32) -> bool { return v > bv || (v == bv && i < bi); }

@compute @workgroup_size(256)
fn argmax_p1(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let nv = am1_p.x;
  let base = wg.y * am1_p.y;                     // column offset in vec4 units
  var bv: f32 = -3.402823e38;
  var bi: u32 = 0xffffffffu;
  // j0 = wg*512 + t, j1 = j0 + 256: consecutive threads -> consecutive 16-byte loads
  for (var k: u32 = 0u; k < 2u; k++) {
    let j = wg.x * 512u + k * 256u + t;
    if (j < nv) {
      let v = am1_x[base + j];
      let e0 = j * 4u;                           // vec4 components are ascending indices
      if (am_better(v.x, e0,      bv, bi)) { bv = v.x; bi = e0; }
      if (am_better(v.y, e0 + 1u, bv, bi)) { bv = v.y; bi = e0 + 1u; }
      if (am_better(v.z, e0 + 2u, bv, bi)) { bv = v.z; bi = e0 + 2u; }
      if (am_better(v.w, e0 + 3u, bv, bi)) { bv = v.w; bi = e0 + 3u; }
    }
  }
  am1_v[t] = bv; am1_i[t] = bi;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s >>= 1u) {    // halving tree; barrier outside the conditional
    if (t < s) {
      let ov = am1_v[t + s]; let oi = am1_i[t + s];
      if (am_better(ov, oi, am1_v[t], am1_i[t])) { am1_v[t] = ov; am1_i[t] = oi; }
    }
    workgroupBarrier();
  }
  if (t == 0u) { am1_part[wg.y * am1_p.z + wg.x] = vec2<u32>(bitcast<u32>(am1_v[0]), am1_i[0]); }
}

// Pass 2: dispatch (nCols). One WG per column reduces its nWG pairs.
// out[col*2] = index, out[col*2+1] = bitcast<u32>(value)  (same layout as the single-WG argmax)
@group(1) @binding(0) var<storage, read> am2_part: array<vec2<u32>>;
@group(1) @binding(1) var<storage, read_write> am2_out: array<u32>;
@group(1) @binding(2) var<uniform> am2_p: vec4<u32>;                      // x: nWG (pairs per column)
var<workgroup> am2_v: array<f32, 256>;
var<workgroup> am2_i: array<u32, 256>;

@compute @workgroup_size(256)
fn argmax_p2(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  let col = wg.x;
  let nw = am2_p.x;
  var bv: f32 = -3.402823e38;
  var bi: u32 = 0xffffffffu;
  for (var k: u32 = t; k < nw; k += 256u) {
    let p = am2_part[col * nw + k];
    let v = bitcast<f32>(p.x);
    if (am_better(v, p.y, bv, bi)) { bv = v; bi = p.y; }
  }
  am2_v[t] = bv; am2_i[t] = bi;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (t < s) {
      let ov = am2_v[t + s]; let oi = am2_i[t + s];
      if (am_better(ov, oi, am2_v[t], am2_i[t])) { am2_v[t] = ov; am2_i[t] = oi; }
    }
    workgroupBarrier();
  }
  if (t == 0u) { am2_out[col * 2u] = am2_i[0]; am2_out[col * 2u + 1u] = bitcast<u32>(am2_v[0]); }
}
```

Notes on the code: `am_better` has scalar params only (no storage-buffer pointers). The tie rule is applied explicitly at every merge, so the thread-local scan order (j0 before j1) and the tree pairing do not affect the result; output is deterministic and equals the single-WG kernel's output on every input without NaN. With nWG=122 < 256, pass 2's loop executes at most once per thread.

### B.4 Host integration (qwen35.js)

1. **Pipeline table** (`G1`, line ~535): add `argmax_p1: ["ro", "rw", "u"], argmax_p2: ["ro", "rw", "u"],`. `bgCommonFor` (710-712) is built for every pipe, so `_d()` works unchanged.

2. **Buffers** (in the `hasHead` block after line 705; `NCmax = this.NC`):
```js
const AM_WG = 256, AM_VEC_PER_WG = AM_WG * 2;                    // 2 vec4 per thread
this.amVec4 = vocab / 4;  this.amWG = Math.ceil(this.amVec4 / AM_VEC_PER_WG);   // 62080, 122
if (vocab % 4 !== 0) this.argmax2 = false;                        // fallback: keep single-WG argmax
this.argPart = device.createBuffer({ size: this.NC * this.amWG * 8, usage: GPUBufferUsage.STORAGE });
this.uAm1   = this._buf(new Uint32Array([this.amVec4, 0, this.amWG, 0]), GPUBufferUsage.UNIFORM);          // single column
this.uAm2   = this._buf(new Uint32Array([this.amWG, 0, 0, 0]), GPUBufferUsage.UNIFORM);
this.bgArgP1 = this._bg(this.pipes.argmax_p1, 1, [this.logits, this.argPart, this.uAm1]);
this.bgArgP2 = this._bg(this.pipes.argmax_p2, 1, [this.argPart, this.argBuf, this.uAm2]);               // argBuf: existing 16 B [index, bits]
```
   In `_initBatch` (after `B.logits = mkB(D.vocab)`):
```js
this.uAm1B   = this._buf(new Uint32Array([this.amVec4, B.logits.stride / 16, this.amWG, 0]), GPUBufferUsage.UNIFORM);
this.bgArgP1B = this._bg(this.pipes.argmax_p1, 1, [B.logits.buf, this.argPart, this.uAm1B]);
this.argOutN  = dev.createBuffer({ size: NC * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
this.stageArgN = dev.createBuffer({ size: NC * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
this.bgArgP2B = this._bg(this.pipes.argmax_p2, 1, [this.argPart, this.argOutN, this.uAm2]);
```
   `B.logits.stride` = `al(vocab)` = 993280 B -> stride/16 = 62080 vec4 (exact; assert `stride % 16 === 0`).

3. **2-D dispatch helper** (next to `_d`, since `_dMC` binds `bgCommonB[0][name]` which does not include the argmax pipes):
```js
_d2(pass, name, bg, nx, ny) {
  if (this.skip && this.skip.has(name)) return;
  pass.setPipeline(this.pipes[name]);
  pass.setBindGroup(0, this.bgCommonFor[name]);
  pass.setBindGroup(1, bg);
  pass.dispatchWorkgroups(nx, ny);
}
```

4. **Draft chain** (`mtpRun`, line 1205): replace
   `if (wantLogits === "argmax") this._d(p, "argmax", this.bgArgmax, 256, 256);`
   with
```js
if (wantLogits === "argmax") {
  if (this.argmax2 !== false) { this._d2(p, "argmax_p1", this.bgArgP1, this.amWG, 1); this._d(p, "argmax_p2", this.bgArgP2, 256, 256); }
  else this._d(p, "argmax", this.bgArgmax, 256, 256);
}
```
   Lines 1208-1215 (copy `argBuf` -> `stageArg`, map, read `[0]`) are unchanged. Keep the old kernel behind `this.argmax2 = Deno.env.get("ARGMAX2") !== "0"` for A/B.

5. **Greedy batched head** — new method beside `headBatch`, used by `verifyN` when the caller samples greedily (tests, bench, and any host whose `sample` is argmax):
```js
async headArgmaxBatch(hs, n = hs ? hs.length / this.dims.dim : this.NC) {
  if (!this.B) this._initBatch();
  const { dim } = this.dims;
  if (hs) for (let c = 0; c < n; c++) this.device.queue.writeBuffer(this.B.x.buf, c * this.B.x.stride, hs.subarray(c * dim, (c + 1) * dim));
  this.device.queue.writeBuffer(this.frameBufsB[0], 0, new Uint32Array([this.pos, this.pos + 1, n, 0]));
  const enc = this.device.createCommandEncoder();
  const p = enc.beginComputePass();
  this._dMC(p, "rmsnorm_mc", this.bgFinalNormMC, 256, 256, n);
  this._dop(p, this.headB);
  this._d2(p, "argmax_p1", this.bgArgP1B, this.amWG, n);
  this._d2(p, "argmax_p2", this.bgArgP2B, n, 1);
  p.end();
  enc.copyBufferToBuffer(this.argOutN, 0, this.stageArgN, 0, n * 8);
  this.device.queue.submit([enc.finish()]);
  await this.stageArgN.mapAsync(GPUMapMode.READ, 0, n * 8);
  const u = new Uint32Array(this.stageArgN.getMappedRange(0, n * 8)).slice();
  this.stageArgN.unmap();
  const ids = [], vals = [];
  for (let c = 0; c < n; c++) { ids.push(u[c * 2]); vals.push(new Float32Array(u.buffer, c * 8 + 4, 1)[0]); }
  if (ids.some((i) => i === 0xffffffff)) throw new Error("argmax: no finite logit");
  return { ids, vals };
}
```
   `verifyN(tokens, pos, runTrunk, { greedy })`: when `greedy`, loop `for (c0 = 0; c0 < n; c0 += NC)` calling `headArgmaxBatch` and return `{ ids, hs }` instead of `{ lgs, hs }`; `specStep` then compares `ids[k] === drafts[k]` directly and uses `ids[K]` as the bonus token. Readback per verify drops from n x 993 KB to 8n bytes. The `headBatch`/`stageLogitsN` path stays for the sampled (temp 0.8 / top-k 40) host in `p2p.html`.

6. **Uniform `am1_p.y` for the single-column path** is 0 (unused since `wg.y` = 0); for the batched path it is the column stride in vec4. `am1_p.z` (nWG) must equal the dispatch's x-extent — assert in JS.

### B.5 Validation

- Add to `test_mtp_deno.js`: for the first 8 draft calls, also `_readback` the logits and compare the CPU argmax (first-occurrence) with the GPU result; must be identical. The existing "greedy, bit-identical" MTP goldens exercise the draft chain end-to-end and must remain bit-identical (the kernel's output is a pure function of the logits; only the reduction shape changed).
- Edge test: a synthetic buffer with duplicates of the max at indices {5, 1000, 248319} -> expect 5; max at index 248319 only -> expect 248319 (last WG, second load, w component); all −inf -> expect 0xffffffff.
- Portability: compile on Chrome and Safari 26 (plain WGSL, no extensions); `bitcast<u32>(f32)`/`bitcast<f32>(u32)` scalar only.

### B.6 Expected cost (measure; do NOT enable timestamp queries)

- Pass 1 streams 993 KB per column across 122 WGs on 48 SMs — bandwidth-wise ~5–10 µs; each dispatch sits on the ~0.05 ms per-dispatch floor measured at DOUT=512, so the 2-pass path is ~0.1 ms per column. The single-WG kernel's actual time is **UNCERTAIN (never measured)**; it reads 1 MB from one SM with 970 dependent loop trips per thread, so it is plausibly 0.2–0.6 ms, but it could be close to the 2-pass cost. Benchmark both with 300 back-to-back dispatches, best-of-3 wall clock (bench_sg.js style), before keeping the extra dispatch in the 7-deep draft chain.
- The certain win is item 5: eliminating the n x 993 KB verify readback and its `copyBufferToBuffer`s. The batched pass carries ~20–37 ms of head + copies + readback overhead, but the readback's share of that is **UNCERTAIN** (not separately measured) — instrument `headBatch` vs `headArgmaxBatch` wall clock.

### B.7 Options deliberately left out of v1 (flagged)

- **Fuse pass 1 into the LM-head GEMV epilogue**: after the row reduction in `matvec_q8_coop_b`, thread 0 writes `(max of its 4 rows, row_base + argmax4)` to `partials[col][wg]` (62080 pairs per column); pass 2 then reduces 62080 pairs (243 per thread) and the logits buffer is never re-read. Saves one dispatch per column and the 993 KB re-read. Not specified here because the batched head kernel's reduction tail would need per-column max/argmax bookkeeping that has not been written or verified; do it only if B.6 shows pass 1 is measurable.
- **Gumbel-max temperature sampling on GPU** (`score = logit * invT + gumbel(i, seed)`, `gumbel = -log(-log(u))` with a hash RNG over (i, seed), then the same argmax) is an exact sample from softmax(logits/T) and would let `p2p.html`'s temp-0.8 path drop the readback too. **UNCERTAIN**: the top-k=40 truncation that `aiSample` applies has no exact GPU equivalent in this design (masking by `logit < max − threshold` is an approximation), so the sampled path stays on `headBatch` until that is decided.
- **Subgroup reduction** inside the tree: measured 0–4% on the big GEMVs and unavailable on Safari 26; not worth the feature gate for an argmax that sits on the dispatch floor.

---

## Summary of decisions

| Item | Decision | Basis |
|---|---|---|
| 8-wide f32 prefill | Do not build; already exists via `BCOLS=8`, measured 0% (bench-log Sep 1) | measured |
| 8/16-wide DP4a | Build, after dn_delta_mc register-resident rewrite; NC=8 default, NC=16 shape-dependent | measured ranges; prefill projection UNCERTAIN |
| Argmax | Ship 2-pass kernel (B.3) + batched greedy head (B.4.5); keep single-WG kernel behind `ARGMAX2=0` | design verified; speed delta vs single-WG UNCERTAIN until benchmarked |

Files touched: `engine/qwen35.js` (WGSL2 string, `G1` table, `_init` head block, `_initBatch`, `_d2`, `mtpRun`, new `headArgmaxBatch`, `verifyN`), `engine/engine.js` (`coopWGSL` DP4a variants, Part A only), `engine/test_mtp_deno.js` (argmax cross-check).