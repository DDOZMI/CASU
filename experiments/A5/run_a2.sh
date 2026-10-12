#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
A2="$ROOT/experiments/A2"
A5="$ROOT/experiments/A5"
CFG="$A5/config_a2.json"
RES="$A5/results_a2"
mkdir -p "$RES/logs"
STATUS="$RES/STATUS"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.5}"
echo 1000 > /proc/self/oom_score_adj
stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "$(stamp) start (pid $$, git $(git rev-parse --short HEAD))" >> "$STATUS"
trap 'rc=$?; echo "$(stamp) FAILED at ${CUR:-?} (exit $rc)" >> "$STATUS"' ERR
CUR=01_tagvec; "$PY" "$A2/01_tagvec.py" --config "$CFG" >> "$RES/logs/01_tagvec.stdout" 2>&1; echo "$(stamp) done 01_tagvec" >> "$STATUS"
for r in bge-m3 contriever-msmarco qwen3-emb-0.6b; do
  for step in 02_eval 03_analyze; do
    CUR="$r $step"
    echo "$(stamp) start $CUR" >> "$STATUS"
    "$PY" "$A2/$step.py" --config "$CFG" --retriever "$r" >> "$RES/logs/${r}_$step.stdout" 2>&1
    echo "$(stamp) done $CUR" >> "$STATUS"
  done
done
CUR=06_generality_a2; "$PY" "$A5/06_generality_a2.py" --config "$CFG" >> "$RES/logs/06.stdout" 2>&1
echo "$(stamp) complete: $RES/A5_A2_generality.md" >> "$STATUS"
