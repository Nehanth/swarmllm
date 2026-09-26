# kernels loop — branch `kernels` only (from opus). Ends 2026-09-26 08:15 UTC.
Goal: 27B kernels as good as possible + fix multi-device gibberish. No GPU/27B here: correctness via SwiftShader
(synthetic 27B-SHAPED model: scratchpad/synth27.gguf, `node tests/e2e/synth.mjs out --shape 27b`), speed only
claimed when measurable; else behind flags with a bench recipe for GB10/Mac.
Each item: branch kernels-work/<slug> off kernels, test, merge --no-ff into kernels, push kernels.

## In flight
- research agent (web): ranked techniques report -> docs/research/kernels-2026-09.md
- dense repro agent: synthetic Qwen3 dense, solo vs split
- big27 run: solo vs split2 on 27B-shaped synth (GEMM path)

## Known multi-device bugs already fixed on opus/kernels (NOT on main): workers never reset DeltaNet state
between questions; spec flag lost past the first worker (3+ devices); rollback/reset could be overtaken.

## Backlog
(fill from research)

## Log
Backlog (from docs/research/kernels-2026-09.md): 0 wire guard+telemetry | 2 batched MTP prefill fill | 1 Q8 prefill GEMM | 7 replay rollback | 4 one-submit spec/encode-ahead | 5 dispatch fusion | 3 flash-decoding (goldens) 
- 02:3x research saved (docs/research/kernels-2026-09.md)
- 02:25 DONE wire guard + |x| telemetry
- 02:59 DONE batched MTP fill (rows bit-identical). big27 solo vs split2 IDENTICAL (GEMM path, 2 turns).
- 03:18 DONE dense ROWS=8 gibberish fix (root cause of 'gibberish with >1 device' for Qwen3 0.6B-4B)
- 03:29 DONE replay rollback (45/45, room identical)
- 03:55 DONE Q8 GEMM (rel 2.7e-7; 27B-shaped Q8 room split==solo). Harness: https weights.
- 04:14 DONE workgroup softmax (bit-identical, default on)
- 04:33 DONE register-resident dn_delta_mc (45/45, bit-identical at 27B dims)
- 04:48 DONE single-token regs dn_delta (35/35). NEXT one-submit draft chain
- 04:53 DONE draft chain (opt-in, identical drafts)
- 05:27 VALIDATION: engine_synth 45/45; 3-device Q8 27B-shaped room solo==split3 both rounds. NEXT dispatch fusion (attn glue)
- 06:37 DONE attn_glue (−64 disp/token) + dn_delta_gn (−48) bit-identical; 45/45; 3-dev Q8 room identical. NEXT batched attention attn_*_mc
- 07:22 DONE batched attention attn_*_mc (−144 disp per K=3 verify) bit-identical; 45/45, dense PASS, 3-dev Q8 room identical. Room ?fuse=0.
