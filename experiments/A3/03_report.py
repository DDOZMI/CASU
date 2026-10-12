from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import PROJECT_ROOT, fingerprint, load_config, run_metadata, save_json

_spec = importlib.util.spec_from_file_location("a3_rq1", Path(__file__).with_name("01_rq1.py"))
a3m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a3m)
A2DIR = {"main_bge-m3": "experiments/A2/results", "a5_bge-m3": "experiments/A5/results_a2/bge-m3",
         "a5_contriever-msmarco": "experiments/A5/results_a2/contriever-msmarco", "a5_qwen3-emb-0.6b": "experiments/A5/results_a2/qwen3-emb-0.6b"}
CUTS = [0.95, 0.98, 0.99]
RUNS = {"main_bge-m3": ("results", "cells_main_bge-m3.csv"), "a5_bge-m3": ("results_a5", "cells_a5_bge-m3.csv"),
        "a5_contriever-msmarco": ("results_a5", "cells_a5_contriever-msmarco.csv"), "a5_qwen3-emb-0.6b": ("results_a5", "cells_a5_qwen3-emb-0.6b.csv")}
PRIMARY = ["a2:val:site@0.01", "a2:val:site@inf", "a2:val:se_mix@1", "a2:val:se_mix@0.5"]
TIERS = ["near", "mid", "far"]
C1, SURF, INK, MUTED, GRID = "#2a78d6", "#fcfcfb", "#1f1f1e", "#6b6a63", "#e6e5df"


def main():
    a3 = PROJECT_ROOT / "experiments" / "A3"
    cfg = load_config(a3 / "config.json")
    ceil = cfg["ceiling"]
    out = a3 / "results_diag"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    fm = lambda r: f"{r.slope:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]" + ("" if r.converged else " (nc)")
    L = ["# A3 RQ1 report (reading-rule order): G first, then the scale explanation", "",
         f"- G = (A_rel − A_abs)/(1 − A_abs) on cells with A_abs ≤ {ceil} (analysis rev3 §5, pre-specified secondary; the A3 reading rule is applied to it as to Δd′). Slopes = A3 interval of record (crossed random intercepts + slopes), per 1 SD of the KB–O tag-distance rank (> 0: far gains more).",
         "- π = σ̄_abs/σ̄_rel − 1 per cell (mean over draws; moment identity on test queries; A3 diagnosis). Δd′ ≈ d_abs·π + push: if π does not change with distance, the Δd′ slope comes from d_abs.", ""]
    grows = []
    L += ["## 1. G (pre-specified) and the cells the ceiling removes", "",
          "| run | outcome | cells kept / total: near | mid | far | G near | G mid | G far | G slope (rank) | Δd′ slope (rank, all cells) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for rn, (rdir, cf) in RUNS.items():
        C = pd.read_csv(a3 / rdir / "tables" / cf)
        S = pd.read_csv(a3 / rdir / "tables" / "slopes.csv")
        S = S[(S.run == rn) & (S.predictor == "rank") & (S.model == "mixed_slopes")].set_index("outcome")
        for o in PRIMARY:
            kept = C.a_abs <= ceil
            cnt = {t: (int((kept & (C.tier == t)).sum()), int((C.tier == t).sum())) for t in TIERS}
            gm = {t: C.loc[kept & (C.tier == t), "G:" + o].mean() for t in TIERS}
            grows.append({"run": rn, "outcome": o, **{f"kept_{t}": cnt[t][0] for t in TIERS}, **{f"n_{t}": cnt[t][1] for t in TIERS},
                          **{f"G_{t}": gm[t] for t in TIERS}, "G_slope": S.loc["G:" + o, "slope"], "G_lo": S.loc["G:" + o, "lo"], "G_hi": S.loc["G:" + o, "hi"]})
            L.append(f"| {rn} | {o[7:]} | " + " | ".join(f"{cnt[t][0]}/{cnt[t][1]}" for t in TIERS) + " | "
                     + " | ".join(f"{gm[t]:+.3f}" for t in TIERS) + f" | {fm(S.loc['G:' + o])} | {fm(S.loc[o])} |")
    pd.DataFrame(grows).to_csv(out / "tables" / "g_ceiling.csv", index=False)

    L += ["", "### G ceiling-cutoff sensitivity (A2 site, m = 4; recomputed from A2 rows as in A3: G per draw on cells with A_abs ≤ cutoff, cell mean)", "",
          "| run | τ | cutoff | cells kept near / mid / far | G slope (rank) |", "|---|---|---|---|---|"]
    srows_g = []
    for rn, (rdir, cf) in RUNS.items():
        C = pd.read_csv(a3 / rdir / "tables" / cf)[["kb", "ood", "x_rank", "tier", "a_abs"]]
        R = pd.read_parquet(PROJECT_ROOT / A2DIR[rn] / "tables" / "rows.parquet", columns=["kb", "ood", "m", "corpus", "partition", "family", "param", "auroc", "auroc_abs"])
        R = R[(R.m == 4) & (R.corpus == "se") & (R.partition == "site") & (R.family == "S_tau")]
        for t in ("0.01", "inf"):
            q = R[R.param.astype(str) == t]
            for c in CUTS:
                g = q[q.auroc_abs <= c].assign(G=lambda x: (x.auroc - x.auroc_abs) / (1 - x.auroc_abs)).groupby(["kb", "ood"]).G.mean().rename("G").reset_index()
                d = C.merge(g, on=["kb", "ood"], how="inner")
                r_ = a3m.mixed_slope(d, "G", "x_rank", slopes=True)
                kept = d.tier.value_counts().reindex(TIERS).fillna(0).astype(int)
                tot = C.tier.value_counts().reindex(TIERS)
                srows_g.append({"run": rn, "tau": t, "cutoff": c, **{f"kept_{k}": int(kept[k]) for k in TIERS}, **r_})
                L.append(f"| {rn} | {t} | {c} | " + " / ".join(f"{kept[k]}/{tot[k]}" for k in TIERS) + f" | {r_['slope']:+.3f} [{r_['lo']:+.3f}, {r_['hi']:+.3f}]" + ("" if r_["converged"] else " (nc)") + " |")
    pd.DataFrame(srows_g).to_csv(out / "tables" / "g_cutoff_sensitivity.csv", index=False)

    L += ["", "## 2. π per cell against the KB–O distance (A2 site, m = 4)", "",
          "| run | τ | π by rank sixth (1 = nearest) | Spearman(π, rank) | π slope (rank) | d_abs near / mid / far | Δd′ slope |", "|---|---|---|---|---|---|---|"]
    sl = pd.read_csv(out / "tables" / "slopes.csv")
    sl = sl[(sl.part == "D2")].set_index(["run", "outcome", "y"])
    fig, axes = plt.subplots(3, 4, figsize=(15, 9.5), sharex="col", facecolor=SURF)
    bins = []
    for j, rn in enumerate(RUNS):
        D = pd.read_csv(out / "tables" / f"decomp_cells_{rn}.csv")
        for i, t in enumerate(["0.01", "inf"]):
            g = D[D.tau.astype(str) == t].copy()
            g["bin"] = pd.qcut(g.x_rank.rank(method="first"), 6, labels=False) + 1
            b = g.groupby("bin").agg(rank_mid=("x_rank", "median"), pi=("pi", "mean"), q25=("pi", lambda v: v.quantile(.25)), q75=("pi", lambda v: v.quantile(.75)), n=("pi", "size")).reset_index()
            bins.append(b.assign(run=rn, tau=t))
            rho = spearmanr(g.pi, g.x_rank).statistic
            da = g.groupby("tier").d_abs_m.mean().reindex(TIERS)
            r_pi, r_dd = sl.loc[(rn, f"a2:site@{t}", "pi")], sl.loc[(rn, f"a2:site@{t}", "ddp")]
            L.append(f"| {rn} | {t} | " + " ".join(f"{v:+.3f}" for v in b.pi) + f" | {rho:+.3f} | {fm(r_pi)} | "
                     + " / ".join(f"{v:.2f}" for v in da) + f" | {fm(r_dd)} |")
            ax = axes[i, j]
            ax.scatter(g.x_rank, g.pi, s=8, color=C1, alpha=0.18, linewidths=0)
            ax.plot(b.rank_mid, b.pi, color=C1, lw=2, marker="o", ms=5)
            ax.axhline(0, color=MUTED, lw=1)
            ax.set_title(f"{rn} — π, τ = {t}", fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel("π = σ̄_abs/σ̄_rel − 1", color=INK)
        ax = axes[2, j]
        g = D[D.tau.astype(str) == "0.01"]
        ax.scatter(g.x_rank, g.d_abs_m, s=8, color=C1, alpha=0.18, linewidths=0)
        bb = g.assign(bin=pd.qcut(g.x_rank.rank(method="first"), 6, labels=False)).groupby("bin").agg(x=("x_rank", "median"), y=("d_abs_m", "mean"))
        ax.plot(bb.x, bb.y, color=C1, lw=2, marker="o", ms=5)
        ax.set_title(f"{rn} — d_abs (moment)", fontsize=10, color=INK)
        ax.set_xlabel("KB–O tag-distance rank (1 = nearest)", color=INK)
        if j == 0:
            ax.set_ylabel("d_abs", color=INK)
    for ax in axes.flat:
        ax.set_facecolor(SURF)
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp_ in ax.spines.values():
            sp_.set_color(GRID)
    fig.suptitle("Noise-cancellation ratio π per (KB, O) cell vs distance (dots: cells; line: means by rank sixth)", color=INK)
    fig.tight_layout()
    fig.savefig(out / "figures" / "pi_vs_rank.png", dpi=130, facecolor=SURF)
    pd.concat(bins).to_csv(out / "tables" / "pi_bins.csv", index=False)

    L += ["", "Figure: `figures/pi_vs_rank.png`.", "", "## 3. RQ3 contrast: the two effects of inserting the O-neighbour (raw cosine units)", "",
          "raw_ood_push = how much more the (ii) comparison term rises on OOD test queries than the (iii) one; raw_id_lower = the same on ID queries (it lowers ID scores). The gain needs raw_ood_push > raw_id_lower.", "",
          "| run | contrast | τ | ood push near / mid / far | id lowering near / mid / far | id / ood ratio near / mid / far | slope ood push | slope id lowering |", "|---|---|---|---|---|---|---|---|"]
    sc = pd.read_csv(out / "tables" / "slopes.csv")
    sc = sc[sc.part == "D2c"].set_index(["run", "outcome", "y"])
    for rn in RUNS:
        CC = pd.read_csv(out / "tables" / f"contrast_cells_{rn}.csv")
        for (cn, t), g in CC.groupby(["contrast", "tau"], sort=False):
            po, pi_ = g.groupby("tier").raw_ood_push.mean().reindex(TIERS), g.groupby("tier").raw_id_lower.mean().reindex(TIERS)
            L.append(f"| {rn} | {cn} | {t} | " + " / ".join(f"{v:.4f}" for v in po) + " | " + " / ".join(f"{v:.4f}" for v in pi_) + " | "
                     + " / ".join(f"{a / b:.2f}" for a, b in zip(pi_, po)) + f" | {fm(sc.loc[(rn, f'a1:{cn}@{t}', 'raw_ood_push')])} | {fm(sc.loc[(rn, f'a1:{cn}@{t}', 'raw_id_lower')])} |")
    (out / "A3_report.md").write_text("\n".join(L) + "\n")
    meta = run_metadata(cfg, "A3.03_report")
    meta["inputs"] = fingerprint([a3 / d / "tables" / f for d, f in RUNS.values()] + [a3 / "results" / "tables" / "slopes.csv", a3 / "results_a5" / "tables" / "slopes.csv", out / "tables" / "slopes.csv"]
                                 + [out / "tables" / f"{k}_cells_{rn}.csv" for rn in RUNS for k in ("decomp", "contrast")])
    save_json(meta, out / "meta_03.json")


if __name__ == "__main__":
    main()
