# Room protocol

Browsers in a room form a WebRTC mesh (PeerJS signaling for the introduction only). One browser is the **host**: it owns the conversation, the tokenizer, the embedding table, the LM head and the sampler. The others are **workers** holding contiguous layer ranges; together they form a **chain** in layer order, with the last worker sending back to the host.

## Lifecycle

| Message | Direction | Meaning |
|---|---|---|
| `ai-wait` | host → worker | join accepted; wait for assignment |
| `ai-load {model, range, next, host}` | host → worker | download and load layers `[range[0], range[1])`; forward to `next` |
| `ai-progress {pct}` / `ai-hostprog` | worker ↔ host | download progress for the room UI |
| `ai-ready` / `ai-ready-all` | worker → host / host → all | layers loaded; room online |
| `ai-reset` | host → all | new conversation: caches and states back to position 0 |
| `ai-genstart` / `ai-token` / `ai-gendone` | host → all | mirror the question and streamed answer to every screen |
| `ai-ask` / `ai-busy` | guest → host | anyone in the room can ask; one generation at a time |

## Compute frames

| Message | Payload | Use |
|---|---|---|
| `ai-hidden {pos}` → … → `ai-hiddenret` | one hidden state | single-token decode lap |
| `ai-hidden-b {basePos, n, spec?}` → … → `ai-hiddenret-b` | `n` hidden states (multiple of the batch width; up to 16) | batched prefill (`spec` absent) or speculative verify (`spec: 1`: the recurrent state is snapshotted after every non-final column) |
| `ai-rollback {k}` | — | host rejected drafts after column `k`; workers restore recurrent state to the snapshot after column `k` |

Hidden states travel as binary frames: an f16-packed `Uint16Array` (10 KB for `dim = 5120`) with the wire format flag `WIRE_F16`; decoders accept f32 for older peers. Frames are correlated by position (`pos` / `basePos`), and the host keeps a timeout per outstanding lap.

## Ordering guarantees

- Data channels are ordered and reliable; a worker processes frames in arrival order, so recurrent states advance deterministically.
- Inside a batched frame, columns are processed strictly in order; snapshot slots are indexed by global column (`frame.snap` packs base and total), so an 8-column verify split into two 4-column chunks on an older worker still rolls back correctly.

## Versioning

Protocol changes bump a message version and make older peers fail loudly at `ai-load`. See GOVERNANCE.md for what counts as a protocol change.
