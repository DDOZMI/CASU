#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
A1="$ROOT/experiments/A1"
A5="$ROOT/experiments/A5"
RES="$A5/results"
mkdir -p "$RES/logs"
STATUS="$RES/STATUS"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.5}"
echo 1000 > /proc/self/oom_score_adj
stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "$(stamp) start (pid $$, git $(git rev-parse --short HEAD))" >> "$STATUS"
trap 'rc=$?; echo "$(stamp) FAILED at ${CUR:-?} (exit $rc)" >> "$STATUS"' ERR
for r in bge-m3 contriever-msmarco qwen3-emb-0.6b; do
  for step in 01_topk 02_eval 03_analyze; do
    CUR="$r $step"
    echo "$(stamp) start $CUR" >> "$STATUS"
    "$PY" "$A1/$step.py" --config "$A5/config.json" --retriever "$r" >> "$RES/logs/${r}_$step.stdout" 2>&1
    echo "$(stamp) done $CUR" >> "$STATUS"
  done
done
CUR="A1 post hoc re-analysis"
echo "$(stamp) start $CUR (adds the N1-Zm by tier table with matching balance and the z-strict sensitivity to experiments/A1/results)" >> "$STATUS"
"$PY" "$A1/03_analyze.py" --config "$A1/config.json" >> "$RES/logs/A1_03_analyze_posthoc.stdout" 2>&1
echo "$(stamp) done $CUR" >> "$STATUS"
CUR=04_generality
"$PY" "$A5/04_generality.py" --config "$A5/config.json" >> "$RES/logs/04_generality.stdout" 2>&1
echo "$(stamp) complete: experiments/A5/results/A5_A1_generality.md" >> "$STATUS"
