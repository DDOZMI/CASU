#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
A0="$ROOT/experiments/A0"
A1="$ROOT/experiments/A1"
RES="$A1/results"
mkdir -p "$RES/logs" "$A0/results/logs"
STATUS="$RES/STATUS"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export PATH="$ROOT/.venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.5}"
echo 1000 > /proc/self/oom_score_adj

stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "$(stamp) pipeline start (pid $$, git $(git rev-parse --short HEAD 2>/dev/null || echo none))" >> "$STATUS"
trap 'rc=$?; echo "$(stamp) FAILED at step ${CUR:-?} (exit $rc)" >> "$STATUS"' ERR
step_sha() {
  local script="$1" prev="$2" conf="$3"
  { cat "$conf" "$script" "$ROOT"/src/oodrag/*.py "$ROOT/experiments/P4/config.json"
    if [[ -n "$prev" ]]; then cat "$RES/.done_$prev" 2>/dev/null || echo "missing"; fi
  } | sha256sum | cut -d' ' -f1
}
PREV=""
run_step() {
  local name="$1" script="$2" conf="$3"; shift 3
  CUR=$name
  local SHA; SHA="$(step_sha "$script" "$PREV" "$conf")"
  if [[ -f "$RES/.done_$name" && "$(cat "$RES/.done_$name")" == "$SHA" && "${FORCE:-0}" != "1" ]]; then
    echo "$(stamp) skip $name (done)" >> "$STATUS"; PREV=$name; return
  fi
  echo "$(stamp) start $name" >> "$STATUS"
  "$PY" "$script" --config "$conf" "$@" >> "$RES/logs/$name.stdout" 2>&1
  echo "$SHA" > "$RES/.done_$name"
  echo "$(stamp) done $name" >> "$STATUS"
  PREV=$name
  if [[ "${STOP_AFTER:-}" == "$name" ]]; then echo "$(stamp) stop after $name (STOP_AFTER)" >> "$STATUS"; exit 0; fi
}
run_step a0_02_build "$A0/02_build.py" "$A0/config.json"
run_step a0_03_encode_main "$A0/03_encode.py" "$A0/config.json" --part main
run_step a1_01_topk "$A1/01_topk.py" "$A1/config.json"
run_step a1_02_eval "$A1/02_eval.py" "$A1/config.json"
run_step a1_03_analyze "$A1/03_analyze.py" "$A1/config.json"
echo "$(stamp) A1 summary ready: experiments/A1/results/A1_summary.md" >> "$STATUS"
run_step a0_03_encode_subset "$A0/03_encode.py" "$A0/config.json" --part subset
echo "$(stamp) pipeline complete" >> "$STATUS"
