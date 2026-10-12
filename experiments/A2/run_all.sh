#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
A2="$ROOT/experiments/A2"
RES="$A2/results"
mkdir -p "$RES/logs"
STATUS="$RES/STATUS"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.5}"
echo 1000 > /proc/self/oom_score_adj
stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "$(stamp) start (pid $$, git $(git rev-parse --short HEAD))" >> "$STATUS"
trap 'rc=$?; echo "$(stamp) FAILED at ${CUR:-?} (exit $rc)" >> "$STATUS"' ERR
for step in 01_tagvec 02_eval 03_analyze; do
  CUR=$step
  echo "$(stamp) start $step" >> "$STATUS"
  "$PY" "$A2/$step.py" --config "$A2/config.json" >> "$RES/logs/$step.stdout" 2>&1
  echo "$(stamp) done $step" >> "$STATUS"
done
echo "$(stamp) complete: experiments/A2/results/A2_summary.md" >> "$STATUS"
