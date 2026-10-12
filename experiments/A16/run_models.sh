#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
PY="$ROOT/.venv/bin/python"; A="$ROOT/experiments/A16"; RES="$A/results"; mkdir -p "$RES/logs"; STATUS="$RES/STATUS_models"
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.5}"
stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
trap 'rc=$?; echo "$(stamp) FAILED at ${CUR:-?} (exit $rc)" >> "$STATUS"' ERR
echo "$(stamp) start (git $(git rev-parse --short HEAD))" >> "$STATUS"
CUR=encode; "$PY" experiments/A0/03_encode.py --config "$A/config_encode.json" --part main >> "$RES/logs/03_encode_models.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
for r in contriever-msmarco qwen3-emb-0.6b; do
  CUR="A1_01_topk $r"; "$PY" experiments/A1/01_topk.py --config "$A/config_a1_models.json" --retriever "$r" >> "$RES/logs/a1m_01_$r.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
  CUR="A1_02_eval $r"; "$PY" experiments/A1/02_eval.py --config "$A/config_a1_models.json" --retriever "$r" >> "$RES/logs/a1m_02_$r.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
  CUR="A1_03_analyze $r"; "$PY" experiments/A1/03_analyze.py --config "$A/config_a1_models.json" --retriever "$r" >> "$RES/logs/a1m_03_$r.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
  CUR="A1_nbcat $r"; "$PY" experiments/A1/02_eval.py --config "$A/config_a1_models_nbcat.json" --retriever "$r" >> "$RES/logs/a1m_nbcat_02_$r.stdout" 2>&1
  "$PY" experiments/A1/03_analyze.py --config "$A/config_a1_models_nbcat.json" --retriever "$r" >> "$RES/logs/a1m_nbcat_03_$r.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
  CUR="A1_tree $r"; mkdir -p "$A/results_models_tree/$r/tables"; for f in pairs.csv draws.csv rows.parquet; do ln -sf "../../../results_models/$r/tables/$f" "$A/results_models_tree/$r/tables/$f"; done
  "$PY" experiments/A1/03_analyze.py --config "$A/config_a1_models_tree.json" --retriever "$r" >> "$RES/logs/a1m_tree_03_$r.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
done
CUR=A10_01_collect; "$PY" experiments/A10/01_collect.py --config "$A/config_a10.json" --runs a16_contriever-msmarco a16_qwen3-emb-0.6b >> "$RES/logs/a10_01_models.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
CUR=A10_02_analyze; "$PY" experiments/A10/02_analyze.py --config "$A/config_a10.json" --runs a16_bge-m3 a16_contriever-msmarco a16_qwen3-emb-0.6b >> "$RES/logs/a10_02_models.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
for r in contriever-msmarco qwen3-emb-0.6b; do
  CUR="A16_04_report $r"; "$PY" "$A/04_report.py" --config "$A/config_a1_models.json" --retriever "$r" --nbcat-res "$A/results_models_nbcat/$r" --tree-res "$A/results_models_tree/$r" >> "$RES/logs/04_report_$r.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
done
CUR=A16_05_models; "$PY" "$A/05_models.py" >> "$RES/logs/05_models.stdout" 2>&1; echo "$(stamp) done $CUR" >> "$STATUS"
echo "$(stamp) complete: $RES/A16_models.md, $RES/a10/A10_summary.md (3 runs), results_models/<r>/{A1_summary.md, A16_extra.md}" >> "$STATUS"
