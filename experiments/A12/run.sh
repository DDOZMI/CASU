#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
PY="$ROOT/.venv/bin/python"; A="$ROOT/experiments/A12"; B="$ROOT/experiments/A13"; RES="$A/results"; RESB="$B/results"
mkdir -p "$RES/logs" "$RESB/logs"; STATUS="$RES/STATUS"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.5}"
stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
trap 'rc=$?; echo "$(stamp) FAILED at ${CUR:-?} (exit $rc)" >> "$STATUS"' ERR
echo "$(stamp) start (git $(git rev-parse --short HEAD))" >> "$STATUS"
CUR=A12_00_encode; "$PY" "$A/00_encode.py" >> "$RES/logs/00_encode.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
CUR=A12_01_dose; "$PY" "$A/01_dose.py" >> "$RES/logs/01_dose.stdout" 2>&1; echo "$(stamp) done $CUR: $RES/A12_summary.md" >> "$STATUS"
for rn in main_bge-m3 a5_bge-m3 a5_contriever-msmarco a5_qwen3-emb-0.6b; do
  CUR="A13_01_baselines $rn"; "$PY" "$B/01_baselines.py" --runs "$rn" >> "$RESB/logs/01_baselines_$rn.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
done
CUR=A13_02_report; "$PY" "$B/02_report.py" >> "$RESB/logs/02_report.stdout" 2>&1
echo "$(stamp) complete: $RES/A12_summary.md, $RESB/A13_summary.md" >> "$STATUS"
