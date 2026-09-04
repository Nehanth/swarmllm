#!/usr/bin/env bash
# GPU test runner. Usage: tests/run.sh [quick|q38|all]
# quick: small-model goldens (Qwen3 0.6B / SmolLM); q38: 27B suites; all: everything.
set -uo pipefail
cd "$(dirname "$0")"
D="deno run --unstable-webgpu --allow-read --allow-env"
quick=(test_qwen.js test_smollm.js test_stream.js test_batch.js test_reset.js test_qwen_split.js test_qwen_stream.js test_batch_split.js)
q38=(test_q38.js test_batch_q38.js test_mtp.js test_b4.js test_twins.js test_gemm.js test_q38_split.js test_mtp_split.js)
case "${1:-quick}" in quick) list=("${quick[@]}");; q38) list=("${q38[@]}");; all) list=("${quick[@]}" "${q38[@]}");; *) echo "unknown suite"; exit 2;; esac
fail=0
for t in "${list[@]}"; do
  echo "=== $t"
  if ! $D "$t" 2>&1 | grep -v "^TU:\|^MESA" | tail -4; then fail=1; fi
  [ "${PIPESTATUS[0]}" -eq 0 ] || fail=1
done
exit $fail
