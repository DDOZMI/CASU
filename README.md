# Cancellation or Suppression?

## Contents

| Path | Role |
|---|---|
| `src/oodrag/` | Shared utilities (paths, seeds, metadata, encoders, metrics) |
| `experiments/P0, P2, P4, P6, P9/` | Data preparation and encoding only (`01_prepare.py`, `02_encode.py`, `config.json`): NQ and MS MARCO queries, CQADupStack queries and corpora, the MS MARCO passage pool. `experiments/P4/config.json` also holds the encoder settings that later steps read. |
| `experiments/A0/` | StackExchange download, build (sites, splits, tag distances), encoding; download of the Li et al. textbook setting |
| `experiments/A1/` | Swap intervention: top-k, evaluation, analysis (main contrast, dose--response, tiers, recovery, intervals) |
| `experiments/A2/` | Same-platform and MS MARCO reference sets (input to A3, A5, A10, A13) |
| `experiments/A3/` | Raw-cosine contrast diagnostics (`results_diag`) |
| `experiments/A4/` | KB-overlap levels L0--L3 |
| `experiments/A5/` | 30-site subset with three encoders (A1 and A2 configurations) |
| `experiments/A6/` | Reproduction of the Li et al. (2024) textbook setting |
| `experiments/A7/`, `experiments/A9/` | Not reported in the paper. A10 reads their outputs as configured, so they are kept to run A10 unchanged. |
| `experiments/A8/` | CQADupStack mixed design |
| `experiments/A10/` | Moment decomposition (Eq. 4 terms, cancellation condition statistics) |
| `experiments/A12/` | Encoding of the Wikipedia passages and MS MARCO queries used by A13 (`00_encode.py`). Its `run.sh` also runs A13 steps 01--02. |
| `experiments/A13/` | Existing baselines (`01`--`05`) and the supervised KB-subspace gate (`06_trisup.py`) |
| `experiments/A14/` | Broad same-platform reference set. Its `run.sh` also runs A13 steps 03--05. |
| `experiments/A15/` | Reference and out-of-scope sources from other platforms (Yahoo! Answers, DBpedia/Wikipedia, MS MARCO) |
| `experiments/A16/` | Amazon product QA testbed (build, encoding, A1 and A10 runs, product-tree and category variants) |
| `requirements.lock.txt` | Pinned Python packages |

## Before running

- Create `.venv` at the project root from `requirements.lock.txt` (Python 3.12, CUDA GPU). The `run*.sh` scripts call `$ROOT/.venv/bin/python`.
- Edit the absolute paths in these configs:
  - `"models"` (local encoder directory) in `experiments/{A0,A6,A12,A15,A16}/config*.json` and `experiments/{P0,P2,P4,P6,P9}/config.json`
  - `"rag01_raw"`, `"rag01_msmarco_corpus"`, and `"msmarco_corpus"` in `experiments/{A6,A15,P0,P2,P9}/config.json`, which point to a local copy of BEIR NQ and MS MARCO.

## Order of execution

Each step writes `meta*.json` with the config hash, code hash, input fingerprints, and environment, and later steps check these where noted in the code.

1. **Data preparation**: `P0`, `P2`, `P4`, `P6`, `P9`, each `01_prepare.py` then `02_encode.py`.
2. **StackExchange**: `A0/00_download.py`, `A0/01_download_li_bridge.py`, then `A1/run_all.sh` (runs `A0/02_build.py`, `A0/03_encode.py`, and A1 steps 01--03).
3. **Reference sets and subset**: `A2/run_all.sh`, `A5/run_a1.sh`, `A5/run_a2.sh`.
4. **Diagnostics and variants**: `A3/run.sh`, `A3/run_diag.sh`, `A4/run_all.sh`, `A6/run_all.sh`, `A7/run.sh`, `A8/run_all.sh`, `A8/run_extras.sh`, `A9/run_main.sh`.
5. **Baselines**: `A12/run.sh` (A12 steps 00--01, A13 steps 01--02). Then rerun `A13/01_baselines.py --out experiments/A13/results_draws` for the run `main_bge-m3` and the three `a5_*` runs, which adds the per-draw rows. The cell tables are identical to those in `experiments/A13/results`. Then `A14/run.sh` (A14, A13 steps 03--05) and `A13/06_trisup.py`.
6. **Other sources**: `A15/00_download.py`, `A15/01_prepare.py`, `A15/01b_yahoo_bank.py`, then `A15/run.sh`.
7. **Decomposition**: `A10/run.sh`.
8. **Amazon**: `A16/00_download.py`, `A16/02_build.py`, then `A16/run.sh` and `A16/run_models.sh`. `A16/01_inventory.py` only reports the category inventory and is optional.
