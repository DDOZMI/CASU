#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
CFG="${1:-$ROOT/experiments/A3/config_diag.json}"
RES="$ROOT/$(python3 -c "import json,sys;print(json.load(open('$CFG'))['paths']['results'])")"
mkdir -p "$RES/logs"; STATUS="$RES/STATUS"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=8
stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
trap 'rc=$?; echo "$(stamp) FAILED (exit $rc)" >> "$STATUS"' ERR
echo "$(stamp) start (git $(git rev-parse --short HEAD), config $CFG)" >> "$STATUS"
"$ROOT/.venv/bin/python" "$ROOT/experiments/A3/02_diagnose.py" --config "$CFG" >> "$RES/logs/02_diagnose.stdout" 2>&1
"$ROOT/.venv/bin/python" "$ROOT/experiments/A3/03_report.py" >> "$RES/logs/03_report.stdout" 2>&1
echo "$(stamp) complete: $RES/A3_diag_summary.md, $RES/A3_report.md" >> "$STATUS"
