from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json, stable_seed

LEV = ["L0", "L1", "L2", "L3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--res", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = Path(args.res) if args.res else p(cfg, "results")
    log = get_logger("A4.03", res / "logs" / "03_analyze.log")
    B, clip = cfg["metrics"]["bootstrap_B"], cfg["metrics"]["dprime_clip"]
    sites = load_json(p(cfg, "a1_emb", cfg["detector"], "sites.json"))
    sidx = {s: i for i, s in enumerate(sites)}
    q_far = load_json(p(cfg, "a0_processed", "bands.json"))["q_far_value"]
    df = pd.read_parquet(res / "tables" / "rows.parquet")
    df["param"] = df["param"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["score"] = df.family + "@" + df.param
    dp = lambda a: np.sqrt(2) * norm.ppf(np.clip(a, clip, 1 - clip))
    df["ddp"] = dp(df.auroc.to_numpy()) - dp(df.auroc_abs.to_numpy())
    df["dA"] = df.auroc - df.auroc_abs
    df["dOP"] = df.op_ood_acc - df.op_ood_acc_abs
    df["dID"] = df.op_id_acc - df.op_id_acc_abs
    df["fr_excess"] = df.fr_overlap_share - df.base_overlap_share
    df["fr_did"] = df.fr_overlap_share - df.fr_overlap_share_abs
    key = ["score", "kb", "ood", "draw"]
    vals = ["ddp", "dA", "dOP", "dID", "mu_id", "mu_ood", "sd_id", "fr_excess", "fr_overlap_share", "base_overlap_share", "fr_did", "fr_overlap_share_abs"]
    W = df.pivot_table(index=key, columns="level", values=vals, aggfunc="first")
    cols = {}
    for v in ("ddp", "dA", "dOP", "dID"):
        for L in LEV:
            cols[f"{v}:{L}"] = W[(v, L)]
        for L in ("L1", "L2", "L3"):
            cols[f"{v}:{L}-L0"] = W[(v, L)] - W[(v, "L0")]
    for L in ("L1", "L2", "L3"):
        for side in ("id", "ood"):
            cols[f"shift_{side}:{L}-L0"] = (W[(f"mu_{side}", L)] - W[(f"mu_{side}", "L0")]) / W[("sd_id", "L0")]
    for L in ("L0", "L1", "L2"):
        cols[f"frx:{L}"] = W[("fr_excess", L)]
        cols[f"frs:{L}"] = W[("fr_overlap_share", L)]
        cols[f"base:{L}"] = W[("base_overlap_share", L)]
        cols[f"did:{L}"] = W[("fr_did", L)]
        cols[f"frsabs:{L}"] = W[("fr_overlap_share_abs", L)]
    C = pd.DataFrame(cols, index=W.index).reset_index()
    lv = pd.read_csv(res / "tables" / "levels.csv")
    for L in ("L0", "L1", "L2", "L3"):
        q = lv[lv.level == L].set_index(["kb", "ood", "draw"])
        C[f"farO_{L}"] = [bool(q.loc[(a, b, c), "dtag_slot_o"] > q_far) if (a, b, c) in q.index else False for a, b, c in zip(C.kb, C.ood, C.draw)]
        C[f"fb_{L}"] = [bool(q.loc[(a, b, c), "fallback"]) if (a, b, c) in q.index else True for a, b, c in zip(C.kb, C.ood, C.draw)]
    for L in ("L1", "L2", "L3"):
        C[f"clean_{L}"] = C[f"farO_{L}"] & C["farO_L0"]
    pairs = pd.read_csv(p(cfg, "a1_results", "tables", "pairs.csv")).set_index(["kb", "ood"])
    C["tier"] = [pairs.loc[(a, b), "tier"] for a, b in zip(C.kb, C.ood)]
    C.to_parquet(res / "tables" / "contrasts.parquet", index=False)

    rng = np.random.default_rng(stable_seed(cfg["seed"], "A4_boot2"))
    nS = len(sites)
    CK = rng.multinomial(nS, [1 / nS] * nS, size=B).astype(float)
    CO = rng.multinomial(nS, [1 / nS] * nS, size=B).astype(float)

    def boot(sub, col):
        s = sub.dropna(subset=[col])
        if s.empty:
            return np.nan, np.nan, np.nan, 0
        cell = s.groupby(["kb", "ood"])[col].mean()
        ki = np.array([sidx[a] for a, _ in cell.index])
        oi = np.array([sidx[b] for _, b in cell.index])
        v = cell.to_numpy()
        Wt = CK[:, ki] * CO[:, oi]
        ok = Wt.sum(1) > 0
        est = (Wt[ok] @ v) / Wt[ok].sum(1)
        return v.mean(), *np.quantile(est, [0.025, 0.975]), len(cell)

    def f(t, d=3):
        return "—" if t[3] == 0 else f"{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]"

    scores = list(dict.fromkeys(df.score))
    est = []
    D = pd.read_csv(p(cfg, "a0_processed", "tag_js.csv"), index_col=0).loc[sites, sites].to_numpy()
    rankm = np.argsort(np.argsort(D + np.eye(len(sites)) * 9, axis=1, kind="stable"), axis=1, kind="stable") + 1

    def regress(g):
        g = g.dropna(subset=["ddp"])
        x = rankm[g.ood.map(sidx).to_numpy(), g.slot.map(sidx).to_numpy()].astype(float)
        X = np.column_stack([(g.level == L).to_numpy(float) for L in ("L1", "L2", "L3")] + [x / x.std()])
        y = g.ddp.to_numpy()
        grp = g.groupby(["kb", "ood", "draw"]).ngroup().to_numpy()
        cnt = np.bincount(grp)

        def dm(A):
            A = A.reshape(len(A), -1)
            mu = np.zeros((grp.max() + 1, A.shape[1]))
            np.add.at(mu, grp, A)
            return A - (mu / cnt[:, None])[grp]
        Xd, yd = dm(X), dm(y)[:, 0]
        cid = g.groupby(["kb", "ood"]).ngroup().to_numpy()
        nc = cid.max() + 1
        XtX = np.zeros((nc, X.shape[1], X.shape[1]))
        Xty = np.zeros((nc, X.shape[1]))
        np.add.at(XtX, cid, Xd[:, :, None] * Xd[:, None, :])
        np.add.at(Xty, cid, Xd * yd[:, None])
        first = g.groupby(cid)[["kb", "ood"]].first()
        ki, oi = first.kb.map(sidx).to_numpy(), first.ood.map(sidx).to_numpy()
        coef = np.linalg.solve(XtX.sum(0), Xty.sum(0))
        Wt = CK[:, ki] * CO[:, oi]
        A = np.tensordot(Wt, XtX, axes=(1, 0))
        bv = Wt @ Xty
        bs = np.array([np.linalg.lstsq(A[b], bv[b], rcond=None)[0] for b in range(B)])
        return [(coef[j], *np.quantile(bs[:, j], [0.025, 0.975]), nc) for j in range(4)]
    L_ = ["# A4 — RQ4 overlap cost (StackExchange, detector " + cfg["detector"] + ")", "",
          f"- A1 pairs and fixed members (paired with A1); 4th slot = KB-overlap level. Cells {C.groupby(['kb', 'ood']).ngroups}. Δd′ vs the absolute score; [ ] two-way (KB x O) bootstrap 95%, B = {B}.",
          f"- level fallback share (no candidate in the band): " + ", ".join(f"{L} {lv[lv.level == L].fallback.mean():.3f}" for L in LEV)
          + "; level site in the far band to O: " + ", ".join(f"{L} {(lv[lv.level == L].dtag_slot_o > q_far).mean():.2f}" for L in LEV)
          + f"; draws without an L1 / L2 site: {C[C.score == scores[0]]['ddp:L1'].isna().mean():.3f} / {C[C.score == scores[0]]['ddp:L2'].isna().mean():.3f}",
          "- Levels also differ in proximity to O (L1–L3 are closer to O than L0 in most draws), so the unadjusted curve mixes the overlap cost with the RQ3 gain; the O-adjusted costs and the two-sided clean draws are co-reported (fixed before the run).", ""]
    L_ += ["## Level curve (Δd′ vs absolute) and costs relative to L0", "",
           "| score | L0 | L1 | L2 | L3 | L1 − L0 | L2 − L0 | L3 − L0 |", "|---|---|---|---|---|---|---|---|"]
    for s in scores:
        g = C[C.score == s]
        row = []
        for c in [f"ddp:{L}" for L in LEV] + [f"ddp:{L}-L0" for L in ("L1", "L2", "L3")]:
            t = boot(g, c)
            est.append({"score": s, "stratum": "all", "estimand": c, "point": t[0], "lo": t[1], "hi": t[2], "n_cells": t[3]})
            row.append(f(t) if "-L0" in c else f"{t[0]:+.3f}")
        L_.append(f"| {s} | " + " | ".join(row) + " |")
    L_ += ["", "## Costs in AUROC and at the operating point (95% calibrated ID acceptance)", "",
           "| score | ΔAUROC L2 − L0 | ΔAUROC L3 − L0 | Δ OOD acc. L2 − L0 | Δ OOD acc. L3 − L0 | Δ ID acc. L2 − L0 | Δ ID acc. L3 − L0 |", "|---|---|---|---|---|---|---|"]
    for s in scores:
        g = C[C.score == s]
        row = []
        for c in ("dA:L2-L0", "dA:L3-L0", "dOP:L2-L0", "dOP:L3-L0", "dID:L2-L0", "dID:L3-L0"):
            t = boot(g, c)
            est.append({"score": s, "stratum": "all", "estimand": c, "point": t[0], "lo": t[1], "hi": t[2], "n_cells": t[3]})
            row.append(f(t))
        L_.append(f"| {s} | " + " | ".join(row) + " |")
    key_scores = [s for s in ("S_tau@0.01", "S_tau@0.1", "S_tau@inf", "mix@0.5", "mix@1") if s in scores]
    L_ += ["", "## O-distance-adjusted level costs (Δd′ vs L0; regression with the slot's tag rank to O and (KB, O, draw) fixed effects)", "",
           "| score | L1 | L2 | L3 | rank-to-O (per SD) |", "|---|---|---|---|---|"]
    for s in scores:
        rr = regress(df[df.score == s])
        for nm, t in zip(("adj:L1-L0", "adj:L2-L0", "adj:L3-L0", "adj:rankO"), rr):
            est.append({"score": s, "stratum": "all", "estimand": nm, "point": t[0], "lo": t[1], "hi": t[2], "n_cells": t[3]})
        L_.append(f"| {s} | " + " | ".join(f(t) for t in rr) + " |")
    L_ += ["", "## Level curve and costs by KB–O tier (Δd′)", "", "| score | estimand | near | mid | far |", "|---|---|---|---|---|"]
    for s in key_scores:
        g = C[C.score == s]
        for c in ("ddp:L0", "ddp:L1", "ddp:L2", "ddp:L3", "ddp:L1-L0", "ddp:L2-L0", "ddp:L3-L0"):
            row = []
            for tn in ("near", "mid", "far"):
                t = boot(g[g.tier == tn], c)
                est.append({"score": s, "stratum": f"tier {tn}", "estimand": c, "point": t[0], "lo": t[1], "hi": t[2], "n_cells": t[3]})
                row.append(f"{f(t)} ({t[3]})")
            L_.append(f"| {s} | {c} | " + " | ".join(row) + " |")
    L_ += ["", "## Sensitivity: two-sided clean draws (L0 and the level site both in O's far band), by tier; L2 without fallback draws, by tier", "",
           "| score | estimand | all | near | mid | far |", "|---|---|---|---|---|---|"]
    for s in key_scores:
        g = C[C.score == s]
        for nm, c, msk in (("L1 − L0 clean", "ddp:L1-L0", g.clean_L1), ("L2 − L0 clean", "ddp:L2-L0", g.clean_L2),
                           ("L3 − L0 clean", "ddp:L3-L0", g.clean_L3), ("L2 − L0 no fallback", "ddp:L2-L0", ~g.fb_L2)):
            row = []
            for tn in ("all", "near", "mid", "far"):
                sub = g[msk] if tn == "all" else g[msk & (g.tier == tn)]
                t = boot(sub, c)
                est.append({"score": s, "stratum": f"{nm} {tn}", "estimand": c, "point": t[0], "lo": t[1], "hi": t[2], "n_cells": t[3]})
                row.append(f"{f(t)} ({t[3]})")
            L_.append(f"| {s} | {nm} | " + " | ".join(row) + " |")
    L_ += ["", "## Decomposition: mean shift of S vs L0 (units of the L0 ID sd); cost appears as ID shift below OOD shift", "",
           "| score | L2 ID | L2 OOD | L3 ID | L3 OOD |", "|---|---|---|---|---|"]
    for s in key_scores:
        g = C[C.score == s]
        L_.append(f"| {s} | " + " | ".join(f"{boot(g, c)[0]:+.2f}" for c in ("shift_id:L2-L0", "shift_ood:L2-L0", "shift_id:L3-L0", "shift_ood:L3-L0")) + " |")
    L_ += ["", "## Auxiliary: tag overlap among false rejections at the operating point", "",
           "Share of falsely rejected ID queries whose tags overlap the level site's bank tag set: with the level in the set / under the absolute score alone / among all ID queries. DiD = first − second (the level's effect on who is rejected).", "",
           "| score | L0 | L1 | L2 | DiD L1 | DiD L2 |", "|---|---|---|---|---|---|"]
    for s in key_scores:
        g = C[C.score == s]
        cells = [f"{boot(g, f'frs:{L}')[0]:.3f} / {boot(g, f'frsabs:{L}')[0]:.3f} / {boot(g, f'base:{L}')[0]:.3f}" for L in ("L0", "L1", "L2")]
        dd = []
        for L in ("L1", "L2"):
            t = boot(g, f"did:{L}")
            est.append({"score": s, "stratum": "all", "estimand": f"did:{L}", "point": t[0], "lo": t[1], "hi": t[2], "n_cells": t[3]})
            dd.append(f(t))
        L_.append(f"| {s} | " + " | ".join(cells + dd) + " |")
    pd.DataFrame(est).to_csv(res / "tables" / "estimands.csv", index=False)
    (res / "A4_summary.md").write_text("\n".join(L_) + "\n")
    meta = run_metadata(cfg, "A4.03_analyze")
    meta["inputs"] = fingerprint([res / "tables" / "rows.parquet", res / "tables" / "levels.csv", p(cfg, "a1_results", "tables", "pairs.csv"),
                                  p(cfg, "a0_processed", "bands.json"), p(cfg, "a0_processed", "tag_js.csv"), p(cfg, "a1_emb", cfg["detector"], "sites.json")])
    meta["outputs"] = fingerprint([res / "A4_summary.md", res / "tables" / "estimands.csv"])
    save_json(meta, res / "meta_03.json")
    log.info("summary written")


if __name__ == "__main__":
    main()
