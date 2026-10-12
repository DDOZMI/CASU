from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import fingerprint, load_config, p, run_metadata, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = p(cfg, "results")
    rs = cfg["retrievers"]
    E = {r: pd.read_csv(res / r / "tables" / "estimands.csv") for r in rs}
    R = {r: pd.read_csv(res / r / "tables" / "regression.csv") for r in rs}
    full = Path(p(cfg, "results")).parents[1] / "A1" / "results" / "tables"
    Ef = pd.read_csv(full / "estimands.csv")
    Sf = pd.read_csv(full / "strata.csv")
    S = {r: pd.read_csv(res / r / "tables" / "strata.csv") for r in rs}
    scores = list(dict.fromkeys(E[rs[0]].score))
    L = ["# A5 — generality of the A1 (RQ3) key results across embedding models", "",
         f"- 30-site subset (10 tag-nearest triples), 870 (KB, O) pairs, identical planning for {', '.join(rs)}. Δd′, two-way (KB x O) bootstrap 95%.",
         "- Agreement labels are descriptive (our operationalisation): 'all >0' / 'all <0' = every model's interval excludes 0 with that sign; 'same sign' = point estimates agree only; 'mixed' = signs differ. Given for the site-level two-way interval and for the triple-level cluster interval (sensitivity; sites within a triple are not independent).",
         "- τ is in absolute cosine units; the spread of bank similarities differs across models (median per-query spread of top-10-mean cosine: bge-m3 ≈ 0.04, Qwen3 ≈ 0.07), so the same τ is a different effective temperature per model. Read agreement over the τ curve, not single cells.",
         "- The subset has a different tier mix (near 7%, mid 28%, far 65% of cells) and its N is usually the O's triple-mate, so pooled values are not comparable with full A1; see the tier table for that comparison.", ""]

    def cell(df, s, est, metric="ddp"):
        q = df[(df.score == s) & (df.estimand == est) & (df.metric == metric)]
        return None if q.empty else q.iloc[0]

    def verdict(vals, lo="lo", hi="hi"):
        if any(v is None for v in vals) or any(pd.isna(v[lo]) for v in vals):
            return "—"
        if all(v[lo] > 0 for v in vals):
            return "all >0"
        if all(v[hi] < 0 for v in vals):
            return "all <0"
        sg = {np.sign(v.point) for v in vals}
        return "same sign" if len(sg) == 1 and 0 not in sg else "mixed"

    for est, title in (("main", "(ii)−(iii) — RQ3 main estimand"), ("matched", "N1−Zm (KB-proximity-matched)"),
                       ("off", "(iii)−abs (does the gain switch off without a neighbour?)"), ("leak", "(i)−(iii) (leak upper bound)")):
        L += [f"## {title}", "", "| score | " + " | ".join(rs) + " | agreement (site) | agreement (triple) |", "|---" * (len(rs) + 3) + "|"]
        for s in scores:
            vals = [cell(E[r], s, est) for r in rs]
            L.append(f"| {s} | " + " | ".join(f"{v.point:+.3f} [{v.lo:+.3f}, {v.hi:+.3f}]" if v is not None else "—" for v in vals)
                     + f" | {verdict(vals)} | {verdict(vals, 'lo_cluster', 'hi_cluster')} |")
        L.append("")
        if est == "main":
            L += ["Tier-wise (ii)−(iii) (A5 models on the subset; full A1 bge-m3 for reference) and the equal-tier-weighted A5 mean", "",
                  "| score | " + " | ".join(f"{r} {t}" for r in rs for t in ("near", "mid", "far")) + " | A1 near | A1 mid | A1 far | "
                  + " | ".join(f"{r} eq-tier" for r in rs) + " |", "|---" * (3 * len(rs) + 4 + len(rs)) + "|"]
            for s in scores:
                row, eq = [], []
                for r in rs:
                    q = S[r][S[r].score == s].set_index("stratum")
                    pts = [q.loc[f"tier {t}"] for t in ("near", "mid", "far")]
                    row += [f"{x.point:+.2f} [{x.lo:+.2f}, {x.hi:+.2f}] ({int(x.n_cells)})" for x in pts]
                    eq.append(f"{np.mean([x.point for x in pts]):+.3f}")
                qf = Sf[Sf.score == s].set_index("stratum")
                row += [f"{qf.loc[f'tier {t}'].point:+.2f}" for t in ("near", "mid", "far")]
                L.append(f"| {s} | " + " | ".join(row + eq) + " |")
            L.append("")
    for coef, title in (("gamma_O", "γ_O (dose response to O proximity, tag rank)"), ("gamma_KB", "γ_KB (KB proximity, tag rank)")):
        L += [f"## {title}", "", "| score | " + " | ".join(rs) + " | agreement |", "|---" * (len(rs) + 2) + "|"]
        for s in scores:
            vals = [R[r][(R[r].score == s) & (R[r].distance == "rank") & (R[r].coef == coef)].iloc[0] for r in rs]
            L.append(f"| {s} | " + " | ".join(f"{v.point:+.3f} [{v.lo:+.3f}, {v.hi:+.3f}]" for v in vals) + f" | {verdict(vals)} |")
        L.append("")
    L += ["## Operating point: Δ OOD acceptance at 95% calibrated ID acceptance, (ii)−(iii)", "", "| score | " + " | ".join(rs) + " | agreement |", "|---" * (len(rs) + 2) + "|"]
    for s in scores:
        vals = [cell(E[r], s, "main", "dOP") for r in rs]
        L.append(f"| {s} | " + " | ".join(f"{v.point:+.3f} [{v.lo:+.3f}, {v.hi:+.3f}]" for v in vals) + f" | {verdict(vals)} |")
    (res / "A5_A1_generality.md").write_text("\n".join(L) + "\n")
    meta = run_metadata(cfg, "A5.04_generality")
    meta["inputs"] = fingerprint([res / r / "tables" / f for r in rs for f in ("estimands.csv", "regression.csv")] + [full / "estimands.csv"])
    save_json(meta, res / "meta_04.json")


if __name__ == "__main__":
    main()
