#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
PY="$ROOT/.venv/bin/python"; A="$ROOT/experiments/A10"; RES="$A/results"; mkdir -p "$RES/logs"; STATUS="$RES/STATUS"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.5}"
stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
trap 'rc=$?; echo "$(stamp) FAILED at ${CUR:-?} (exit $rc)" >> "$STATUS"' ERR
echo "$(stamp) start (git $(git rev-parse --short HEAD))" >> "$STATUS"
for rn in main_bge-m3 a5_bge-m3 a5_contriever-msmarco a5_qwen3-emb-0.6b a8_bge-m3 a8_contriever-msmarco a8_qwen3-emb-0.6b; do
  CUR="01_collect $rn"; "$PY" "$A/01_collect.py" --runs "$rn" >> "$RES/logs/01_collect_$rn.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
done
CUR=02_analyze; "$PY" "$A/02_analyze.py" >> "$RES/logs/02_analyze.stdout" 2>&1
echo "$(stamp) complete: $RES/A10_summary.md" >> "$STATUS"
