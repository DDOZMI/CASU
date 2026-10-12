#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
PY="$ROOT/.venv/bin/python"; A="$ROOT/experiments/A16"; RES="$A/results"; mkdir -p "$RES/logs"; STATUS="$RES/STATUS"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.5}"
stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
trap 'rc=$?; echo "$(stamp) FAILED at ${CUR:-?} (exit $rc)" >> "$STATUS"' ERR
echo "$(stamp) start (git $(git rev-parse --short HEAD))" >> "$STATUS"
if ! "$PY" - <<'PYEOF'
import json, sys
sys.path.insert(0, "src")
from oodrag.common import load_config, sha256_file
m = json.load(open("data/processed/A16/meta_02.json"))
c = load_config("experiments/A16/config.json")
sys.exit(0 if m["config_sha256"] == c["_config_sha256"] and m["code_sha256"].get("experiments/A16/02_build.py") == sha256_file("experiments/A16/02_build.py") else 1)
PYEOF
then CUR=A16_02_build; "$PY" "$A/02_build.py" >> "$RES/logs/02_build.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"; else echo "$(stamp) 02_build up to date" >> "$STATUS"; fi
CUR=A16_encode; "$PY" experiments/A0/03_encode.py --config "$A/config_encode.json" --part main >> "$RES/logs/03_encode.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
CUR=A1_01_topk; "$PY" experiments/A1/01_topk.py --config "$A/config_a1.json" >> "$RES/logs/a1_01.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
CUR=A1_02_eval; "$PY" experiments/A1/02_eval.py --config "$A/config_a1.json" >> "$RES/logs/a1_02.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
CUR=A1_03_analyze; "$PY" experiments/A1/03_analyze.py --config "$A/config_a1.json" >> "$RES/logs/a1_03.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
CUR=A1_nbcat_02_eval; "$PY" experiments/A1/02_eval.py --config "$A/config_a1_nbcat.json" >> "$RES/logs/a1_nbcat_02.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
CUR=A1_nbcat_03_analyze; "$PY" experiments/A1/03_analyze.py --config "$A/config_a1_nbcat.json" >> "$RES/logs/a1_nbcat_03.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
CUR=A1_tree_03_analyze; mkdir -p "$A/results_tree/tables"; for f in pairs.csv draws.csv rows.parquet; do ln -sf "../../results/tables/$f" "$A/results_tree/tables/$f"; done
"$PY" experiments/A1/03_analyze.py --config "$A/config_a1_tree.json" >> "$RES/logs/a1_tree_03.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
CUR=A10_01_collect; "$PY" experiments/A10/01_collect.py --config "$A/config_a10.json" --runs a16_bge-m3 >> "$RES/logs/a10_01.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
CUR=A10_02_analyze; "$PY" experiments/A10/02_analyze.py --config "$A/config_a10.json" --runs a16_bge-m3 >> "$RES/logs/a10_02.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
CUR=A16_04_report; "$PY" "$A/04_report.py" --config "$A/config_a1.json" >> "$RES/logs/04_report.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
echo "$(stamp) complete: $RES/A1_summary.md, $RES/a10/A10_summary.md, $RES/A16_extra.md (+ results_nbcat, results_tree)" >> "$STATUS"
