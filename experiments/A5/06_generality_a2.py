from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import fingerprint, load_config, p, run_metadata, save_json

ROWS = [("val:site@0.01", "site S_τ@0.01 − abs"), ("val:site@inf", "site S_τ@∞ − abs"), ("val:random@0.01", "random S_τ@0.01 − abs"),
        ("val:random@inf", "random S_τ@∞ − abs"), ("val:se_mix@1", "SE pooled mix@1 − abs"), ("val:ms_mix@1", "MS pooled mix@1 − abs"),
        ("val:se_mix@0.5", "SE pooled mix@0.5 − abs"), ("val:ms_mix@0.5", "MS pooled mix@0.5 − abs"),
        ("site-random@0.01", "site − random @0.01"), ("site-random@inf", "site − random @∞"),
        ("tag-random@0.01", "tag − random @0.01"), ("tag-random@inf", "tag − random @∞"),
        ("site-mix1@0.01", "site − mix@1 @0.01"), ("tag-mix1@0.01", "tag − mix@1 @0.01"), ("random-mix1@0.01", "random − mix@1 @0.01"),
        ("site-mix1@inf", "site − mix@1 @∞"), ("random-mix1@inf", "random − mix@1 @∞"),
        ("msmix1-site@0.01", "MS pooled mix@1 − SE site @0.01"), ("msmix1-site@inf", "MS pooled mix@1 − SE site @∞"),
        ("spread@0.01", "spread across partitions @0.01"), ("spread@inf", "spread across partitions @∞"),
        ("se-ms_mix@0.5", "SE − MS pooled mix@0.5"), ("se-ms_mix@1", "SE − MS pooled mix@1"),
        ("ms_cross-ms_random@0.01", "MS cross − MS random @0.01"), ("ms_cross-ms_random@inf", "MS cross − MS random @∞")]
STRATA = [("Onb1_KBnb0", "val:site@0.01"), ("Onb0_KBnb0", "val:site@0.01"), ("Onb1_KBnb1", "val:site@0.01"), ("Onb0_KBnb1", "val:site@0.01"),
          ("Onb0_KBnb1", "site-random@inf"), ("Onb1_KBnb0", "site-random@inf")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config_a2.json")))
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = p(cfg, "results")
    rs = cfg["retrievers"]
    E = {r: pd.read_csv(res / r / "tables" / "estimands.csv") for r in rs}
    sets = [pd.read_csv(res / r / "tables" / "sets.csv") for r in rs]
    assert all(s_.equals(sets[0]) for s_ in sets[1:]), "comparison-site draws differ across models"
    C = {r: pd.read_parquet(res / r / "tables" / "contrasts.parquet") for r in rs}

    def get(r, est, stratum="all"):
        q = E[r][(E[r].estimand == est) & (E[r].stratum == stratum) & (E[r].m == 4)]
        return None if q.empty else q.iloc[0]

    def verdict(v, lo="lo", hi="hi"):
        if any(x is None or pd.isna(x[lo]) for x in v):
            return "—"
        if all(x[lo] > 0 for x in v):
            return "all >0"
        if all(x[hi] < 0 for x in v):
            return "all <0"
        sg = {np.sign(x.point) for x in v}
        return "same sign" if len(sg) == 1 and 0 not in sg else "mixed"

    fm = lambda x: "—" if x is None else f"{x.point:+.3f} [{x.lo:+.3f}, {x.hi:+.3f}]"
    L = ["# A5 — generality of the A2 (RQ2) key results across embedding models (m = 4, 30-site subset)", "",
         "- Δd′ vs abs; site-level two-way (KB x O) bootstrap 95%; agreement labels (descriptive, our operationalisation) for the site-level and the triple-level (cluster) interval. The 'same' partition is the detector's own embedding and 'cross' the next encoder in the rotation, so partition contrasts involving cross/same (and the spread) are model-specific; tag and random partitions (and their purity) are identical across models by construction. τ is in absolute cosine units; the per-model spread of bank similarity differs (~1.5x). The subset's neighbour-stratum shares differ from the 86-site A2, so values are not comparable with A2-main.", "",
         "| estimand | " + " | ".join(rs) + " | agreement (site) | agreement (triple) |", "|---" * (len(rs) + 3) + "|"]
    for est, lab in ROWS:
        v = [get(r, est) for r in rs]
        lbl = ("— (≥ 0 by construction)", "—") if est.startswith("spread") else (verdict(v), verdict(v, "lo_cluster", "hi_cluster"))
        L.append(f"| {lab} | " + " | ".join(fm(x) for x in v) + f" | {lbl[0]} | {lbl[1]} |")
    L += ["", "## Crossed strata (O-neighbour x KB-neighbour in the comparison set)", "", "| stratum | estimand | " + " | ".join(rs) + " | agreement |", "|---" * (len(rs) + 3) + "|"]
    for st, est in STRATA:
        v = [get(r, est, st) for r in rs]
        L.append(f"| {st} | {est} | " + " | ".join(fm(x) + (f" ({int(x.n_cells)})" if x is not None else "") for x in v) + f" | {verdict(v)} |")
    L += ["", "## Partition purity (mean majority-site share; m = 4)", "", "| model | tag | cross | same | random |", "|---|---|---|---|---|"]
    for r in rs:
        L.append(f"| {r} | " + " | ".join(f"{C[r][f'purity:{pt}@-'].mean():.3f}" for pt in ("tag", "cross", "same", "random")) + " |")
    (res / "A5_A2_generality.md").write_text("\n".join(L) + "\n")
    meta = run_metadata(cfg, "A5.06_generality_a2")
    meta["inputs"] = fingerprint([res / r / "tables" / f for r in rs for f in ("estimands.csv", "contrasts.parquet", "sets.csv")])
    save_json(meta, res / "meta_06.json")


if __name__ == "__main__":
    main()
