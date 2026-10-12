#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
A6="$ROOT/experiments/A6"
RES="$A6/results"
mkdir -p "$RES/logs"
STATUS="$RES/STATUS"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.5}"
echo 1000 > /proc/self/oom_score_adj
stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "$(stamp) start (pid $$, git $(git rev-parse --short HEAD))" >> "$STATUS"
trap 'rc=$?; echo "$(stamp) FAILED at ${CUR:-?} (exit $rc)" >> "$STATUS"' ERR
for step in 01_prepare 02_encode 03_eval; do
  CUR=$step
  echo "$(stamp) start $step" >> "$STATUS"
  "$PY" "$A6/$step.py" --config "$A6/config.json" >> "$RES/logs/$step.stdout" 2>&1
  echo "$(stamp) done $step" >> "$STATUS"
done
echo "$(stamp) complete: experiments/A6/results/A6_summary.md" >> "$STATUS"
