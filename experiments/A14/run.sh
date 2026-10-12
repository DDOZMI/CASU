#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
PY="$ROOT/.venv/bin/python"; A="$ROOT/experiments/A13"; B="$ROOT/experiments/A14"; RES="$B/results"; RA="$A/results"
mkdir -p "$RES/logs" "$RA/logs"; STATUS="$RES/STATUS"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.5}"
stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
trap 'rc=$?; echo "$(stamp) FAILED at ${CUR:-?} (exit $rc)" >> "$STATUS"' ERR
echo "$(stamp) start (git $(git rev-parse --short HEAD))" >> "$STATUS"
CUR=A13_03_extra; "$PY" "$A/03_extra.py" >> "$RA/logs/03_extra.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
CUR=A13_02_report; "$PY" "$A/02_report.py" >> "$RA/logs/02_report.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
CUR=A13_04_kbpredict; "$PY" "$A/04_kbpredict.py" >> "$RA/logs/04_kbpredict.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
CUR=A13_05_equiv; "$PY" "$A/05_equiv.py" >> "$RA/logs/05_equiv.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
for rn in main_bge-m3 a5_bge-m3 a5_contriever-msmarco a5_qwen3-emb-0.6b; do
  CUR="A14_01_wide $rn"; "$PY" "$B/01_wide.py" --runs "$rn" >> "$RES/logs/01_wide_$rn.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
done
CUR=A14_02_report; "$PY" "$B/02_report.py" >> "$RES/logs/02_report.stdout" 2>&1
echo "$(stamp) complete: $RA/A13_summary.md, $RA/A13_kbpredict.md, $RA/A13_equivalence.md, $RES/A14_summary.md" >> "$STATUS"
