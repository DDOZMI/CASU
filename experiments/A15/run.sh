#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
PY="$ROOT/.venv/bin/python"; A="$ROOT/experiments/A15"; RES="$A/results"; mkdir -p "$RES/logs"; STATUS="$RES/STATUS"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.5}"
stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
trap 'rc=$?; echo "$(stamp) FAILED at ${CUR:-?} (exit $rc)" >> "$STATUS"' ERR
echo "$(stamp) start (git $(git rev-parse --short HEAD))" >> "$STATUS"
CUR=wait_encode; t0=$(date +%s)
while pgrep -f "^[^ ]*python[^ ]* [^ ]*experiments/A15/02_encode.py" > /dev/null; do sleep 30; [ $(( $(date +%s) - t0 )) -lt 14400 ] || { echo "encode wait timeout"; false; }; done
CUR=A15_02_encode_check; "$PY" "$A/02_encode.py" >> "$RES/logs/02_encode.stdout" 2>&1; echo "$(stamp) done $CUR (all up to date or re-encoded)" >> "$STATUS"
CUR=A15_02b_encode_yahoo; "$PY" "$A/02b_encode_yahoo.py" >> "$RES/logs/02b_encode_yahoo.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
for rn in main_bge-m3 a5_bge-m3 a5_contriever-msmarco a5_qwen3-emb-0.6b; do
  CUR="A15_03_topk $rn"; "$PY" "$A/03_topk.py" --runs "$rn" >> "$RES/logs/03_topk_$rn.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
done
for rn in main_bge-m3 a5_bge-m3 a5_contriever-msmarco a5_qwen3-emb-0.6b; do
  CUR="A15_04_eval $rn"; "$PY" "$A/04_eval.py" --runs "$rn" >> "$RES/logs/04_eval_$rn.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
done
CUR=A15_05_report; "$PY" "$A/05_report.py" >> "$RES/logs/05_report.stdout" 2>&1
echo "$(stamp) complete: $RES/A15_summary.md" >> "$STATUS"
