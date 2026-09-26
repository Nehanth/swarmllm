# Tabby loop — until 2026-09-26 16:00 UTC, branch tabby-kernel only
Merge via tabby-work/<slug> --no-ff; worktree for merges: scratchpad/wt-dense (switch it to tabby-kernel).
Validation: unit tests, engine_synth, relevant *_synth, room_synth --model synth27q8 --compare --devices 3.

## Backlog (ranked)
1. harness/sessions.js: session manager (GPU slots LRU, spill to OPFS, restore) + e2e test
2. q8 KV cache option (kvQ8) for 32K+ (flash kernel variant), tolerance test
3. n-gram/suffix drafting chained after MTP (room + engine specStepDrafts), spec==plain
4. Persist room checkpoints to OPFS on every device (opt-in), resume after reload
5. Tiled multi-column prefill attention (share K/V loads across columns) behind flag
6. Pipelined speculative windows across devices (design doc if too big)

## Log
- 09:55 loop start
- 10:20 DONE sessions manager (3 sessions, gpu+disk switch bit-exact). NEXT q8 KV
- 10:55 lookup-long committed on tabby-work/lookup-long (drafts_synth PASS); merge after q8 chain + room
- 11:20 attn-tile committed on tabby-work/attn-tile (fusion PASS + mutation)
- 10:45 kv-q8 committed (engine --q8 45/45, flash --q8, state --q8); agent harness committed on tabby-work/agent; waiting chain (f16 engine, flash27 q8, room)
- 11:15 agent2 (constraint in sampling, context budget) + design doc committed on tabby-work/agent2; combined validation of tabby-merge running (m_*.log)
