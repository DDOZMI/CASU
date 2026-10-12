from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
from scipy.special import logsumexp
from scipy.stats import norm, rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import PROJECT_ROOT, fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json

_spec = importlib.util.spec_from_file_location("a3_rq1", Path(__file__).with_name("01_rq1.py"))
a3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a3)
TIERS = ["near", "mid", "far"]


def agg(A, t):
    return A.mean(1) if t == "inf" else float(t) * (logsumexp(A / float(t), axis=1) - np.log(A.shape[1]))


def auroc(neg, pos):
    r = rankdata(np.concatenate([neg, pos]))
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def moments(s_id, s_ood, c_id, c_ood):
    sb_abs = np.sqrt((s_id.var() + s_ood.var()) / 2)
    S_id, S_ood = s_id - c_id, s_ood - c_ood
    sb_rel = np.sqrt((S_id.var() + S_ood.var()) / 2)
    dmu_s, dmu_c = s_id.mean() - s_ood.mean(), c_id.mean() - c_ood.mean()
    d_abs, d_rel = dmu_s / sb_abs, (dmu_s - dmu_c) / sb_rel
    return {"d_abs_m": d_abs, "d_rel_m": d_rel, "dd_m": d_rel - d_abs, "pi": sb_abs / sb_rel - 1,
            "T_var": (dmu_s - dmu_c / 2) * (1 / sb_rel - 1 / sb_abs), "T_push": -dmu_c * (1 / sb_abs + 1 / sb_rel) / 2,
            "B": dmu_s - dmu_c, "sb": sb_rel, "mc_id": c_id.mean(), "mc_ood": c_ood.mean()}


def contrast_split(X, Y):
    h = (1 / X["sb"] + 1 / Y["sb"]) / 2
    return {"dd_m": X["d_rel_m"] - Y["d_rel_m"], "C_var": (X["B"] + Y["B"]) / 2 * (1 / X["sb"] - 1 / Y["sb"]),
            "C_idcost": -(X["mc_id"] - Y["mc_id"]) * h, "C_oodpush": (X["mc_ood"] - Y["mc_ood"]) * h,
            "raw_id_lower": X["mc_id"] - Y["mc_id"], "raw_ood_push": X["mc_ood"] - Y["mc_ood"]}


def rank_matrix(D):
    return np.argsort(np.argsort(D + np.eye(len(D)) * 9, axis=1, kind="stable"), axis=1, kind="stable") + 1


def run_one(rn, run, cfg, D_full, log):
    root = PROJECT_ROOT
    clip = cfg["dprime_clip"]
    dp = lambda a: np.sqrt(2) * norm.ppf(np.clip(a, clip, 1 - clip))
    edir = root / run["emb"]
    sites = load_json(edir / "sites.json")
    sidx = {s: i for i, s in enumerate(sites)}
    Lm = np.asarray(np.load(edir / "topk.npy", mmap_mode="r"), dtype=np.float64).mean(2)
    qi = pd.read_parquet(edir / "q_index.parquet")
    rows_of = {k: g.index.to_numpy() for k, g in qi.groupby(["site", "role"])}
    pairs = pd.read_csv(root / run["a1"] / "tables" / "pairs.csv")
    D = D_full.loc[sites, sites].to_numpy()
    rk = rank_matrix(D)
    m, taus = cfg["m"], cfg["taus"]

    R2 = pd.read_parquet(root / run["a2"] / "tables" / "rows.parquet")
    R2 = R2[(R2.m == m) & (R2.corpus == "se")].copy()
    R2["param"] = R2.param.astype(str).str.replace(r"\.0$", "", regex=True)
    sel = []
    for o in cfg["a2_outcomes"]:
        part, par = o.split("@")
        q = R2[(R2.partition == ("site" if part == "site" else "pooled")) & (R2.family == ("S_tau" if part == "site" else "mix")) & (R2.param == par)].copy()
        q["outcome"] = "a2:" + o
        sel.append(q)
    Q2 = pd.concat(sel)
    da, dr = dp(Q2.auroc_abs.to_numpy()), dp(Q2.auroc.to_numpy())
    Q2["ddp"], Q2["dA"] = dr - da, Q2.auroc - Q2.auroc_abs
    Q2["dOP"] = Q2.op_ood_acc_abs - Q2.op_ood_acc
    Q2["logratio"] = np.where((da > 0) & (dr > 0), np.log(np.maximum(dr, 1e-12) / np.maximum(da, 1e-12)), np.nan)
    drop_lr = Q2.groupby("outcome").logratio.apply(lambda v: float(v.isna().mean())).to_dict()
    clip_sh = Q2.assign(c=(Q2.auroc >= 1 - clip) | (Q2.auroc_abs >= 1 - clip)).groupby("outcome").c.mean().to_dict()
    sc2 = Q2.groupby(["outcome", "kb", "ood"])[["ddp", "dA", "dOP", "logratio"]].mean().reset_index()

    R1 = pd.read_parquet(root / run["a1"] / "tables" / "rows.parquet",
                         columns=["kb", "ood", "draw", "role", "family", "param", "auroc", "auroc_abs", "op_ood_acc", "op_ood_acc_abs"])
    R1 = R1[(R1.family == "S_tau") & R1.param.astype(str).isin(taus)]
    piv = {}
    for col in ("auroc", "op_ood_acc"):
        piv[col] = R1.pivot_table(index=["param", "kb", "ood", "draw"], columns="role", values=col)
    a_abs = R1.groupby(["param", "kb", "ood", "draw"]).auroc_abs.first()
    sc1 = []
    for t in taus:
        A = piv["auroc"].loc[t]
        OP = piv["op_ood_acc"].loc[t]
        dA0 = dp(a_abs.loc[t].reindex(A.index).to_numpy())
        dslot = {r: dp(A[r].to_numpy()) for r in ("N1", "N2", "Z1", "Z2", "Zm") if r in A}
        lr = {r: np.where((dslot[r] > 0) & (dA0 > 0), np.log(np.maximum(dslot[r], 1e-12) / np.maximum(dA0, 1e-12)), np.nan) for r in dslot}
        for cn, (xa, ya) in {"main": (("N1", "N2"), ("Z1", "Z2")), "matched": (("N1",), ("Zm",))}.items():
            if cn not in cfg["a1_contrasts"]:
                continue
            f = lambda vals, keys: np.mean([vals[k] for k in keys], axis=0)
            d = pd.DataFrame({"kb": A.index.get_level_values("kb"), "ood": A.index.get_level_values("ood"),
                              "ddp": f(dslot, xa) - f(dslot, ya),
                              "dA": f({r: A[r].to_numpy() for r in dslot}, xa) - f({r: A[r].to_numpy() for r in dslot}, ya),
                              "dOP": f({r: OP[r].to_numpy() for r in dslot}, ya) - f({r: OP[r].to_numpy() for r in dslot}, xa),
                              "logratio": f(lr, xa) - f(lr, ya)})
            d["outcome"] = f"a1:{cn}@{t}"
            drop_lr[f"a1:{cn}@{t}"] = float(np.isnan(d.logratio.to_numpy()).mean())
            sl_ = list(xa + ya)
            clip_sh[f"a1:{cn}@{t}"] = float(((A[sl_].to_numpy() >= 1 - clip).any(1) | (a_abs.loc[t].reindex(A.index).to_numpy() >= 1 - clip)).mean())
            sc1.append(d.groupby(["outcome", "kb", "ood"]).mean().reset_index())
    SC = pd.concat([sc2] + sc1, ignore_index=True)
    ki, oi = SC.kb.map(sidx).to_numpy(), SC.ood.map(sidx).to_numpy()
    SC["x_rank"] = rk[ki, oi].astype(float)
    SC = SC.merge(pairs[["kb", "ood", "tier"]], on=["kb", "ood"], how="left")

    sets = pd.read_csv(root / run["a2"] / "tables" / "sets.csv")
    rows2 = R2[(R2.partition == "site") & (R2.family == "S_tau")].set_index(["kb", "ood", "draw", "param"]).auroc
    draws = pd.read_csv(root / run["a1"] / "tables" / "draws.csv")
    a1auc = R1.set_index(["kb", "ood", "draw", "role", "param"]).auroc
    recs, crec, prof, maxerr, hits = [], [], [], 0.0, 0
    sets_of = {k: g for k, g in sets.groupby(["kb", "ood"])}
    draws_of = {k: g for k, g in draws.groupby(["kb", "ood"])}
    for kb, o in zip(pairs.kb, pairs.ood):
        i_id, i_ood = rows_of[(kb, "test")], rows_of[(o, "test")]
        s_id, s_ood = Lm[i_id, sidx[kb]], Lm[i_ood, sidx[kb]]
        oth = [sidx[s] for s in sites if s not in (kb, o)]
        prof.append({"kb": kb, "ood": o, "profile": float(np.corrcoef(Lm[np.ix_(i_id, oth)].mean(0), Lm[np.ix_(i_ood, oth)].mean(0))[0, 1])})
        for d, order in zip(sets_of[(kb, o)].draw, sets_of[(kb, o)].order):
            C = [sidx[s] for s in order.split("+")[:m]]
            for t in taus:
                c_id, c_ood = agg(Lm[np.ix_(i_id, C)], t), agg(Lm[np.ix_(i_ood, C)], t)
                if (kb, o, d, t) in rows2.index:
                    maxerr = max(maxerr, abs(auroc(-(s_id - c_id), -(s_ood - c_ood)) - rows2.loc[(kb, o, d, t)]))
                    hits += 1
                mo = moments(s_id, s_ood, c_id, c_ood)
                recs.append({"kb": kb, "ood": o, "draw": d, "tau": t, **{k: mo[k] for k in ("d_abs_m", "d_rel_m", "dd_m", "pi", "T_var", "T_push")}})
        g = draws_of.get((kb, o), draws.iloc[:0])
        for (d, fx), gg in g.groupby(["draw", "fixed"]):
            slot = dict(zip(gg.role, gg.slot))
            for t in taus:
                pieces = {}
                for r in ("N1", "N2", "Z1", "Z2", "Zm"):
                    if r not in slot:
                        continue
                    C = [sidx[x] for x in fx.split("+")] + [sidx[slot[r]]]
                    c_id, c_ood = agg(Lm[np.ix_(i_id, C)], t), agg(Lm[np.ix_(i_ood, C)], t)
                    pieces[r] = moments(s_id, s_ood, c_id, c_ood)
                    if r in ("N1", "Z1"):
                        maxerr = max(maxerr, abs(auroc(-(s_id - c_id), -(s_ood - c_ood)) - a1auc.loc[(kb, o, d, r, t)]))
                        hits += 1
                for cn, (xa, ya) in {"main": (("N1", "N2"), ("Z1", "Z2")), "matched": (("N1",), ("Zm",))}.items():
                    if not all(k in pieces for k in xa + ya):
                        continue
                    sp_ = [contrast_split(pieces[x], pieces[y]) for x in xa for y in ya]
                    crec.append({"kb": kb, "ood": o, "draw": d, "tau": t, "contrast": cn,
                                 **{k: float(np.mean([q[k] for q in sp_])) for k in ("dd_m", "C_var", "C_idcost", "C_oodpush", "raw_id_lower", "raw_ood_push")}})
    log.info(f"{rn}: max |AUROC(recomputed) - stored| = {maxerr:.2e} over {hits} checks")
    assert hits > 0, f"{rn}: reproduction check found no keys"
    assert maxerr < 1e-5, f"{rn}: recomputed scores do not reproduce the stored AUROCs ({maxerr})"
    DC = pd.DataFrame(recs).groupby(["tau", "kb", "ood"])[["d_abs_m", "d_rel_m", "dd_m", "pi", "T_var", "T_push"]].mean().reset_index()
    CC = pd.DataFrame(crec).groupby(["contrast", "tau", "kb", "ood"])[["dd_m", "C_var", "C_idcost", "C_oodpush", "raw_id_lower", "raw_ood_push"]].mean().reset_index()
    PR = pd.DataFrame(prof)
    for X in (DC, CC):
        X["x_rank"] = rk[X.kb.map(sidx).to_numpy(), X.ood.map(sidx).to_numpy()].astype(float)
    DC = DC.merge(PR, on=["kb", "ood"]).merge(pairs[["kb", "ood", "tier"]], on=["kb", "ood"], how="left")
    CC = CC.merge(pairs[["kb", "ood", "tier"]], on=["kb", "ood"], how="left")
    obs2 = SC[SC.outcome.str.startswith("a2:site@")].assign(tau=lambda x: x.outcome.str.split("@").str[1])[["tau", "kb", "ood", "ddp"]]
    DC = DC.merge(obs2, on=["tau", "kb", "ood"], how="left")
    DC["resid"] = DC.ddp - DC.dd_m
    a_abs2 = R2[(R2.partition == "site") & (R2.family == "S_tau")].groupby(["kb", "ood"]).auroc_abs.first()
    DC["d_abs_auc"] = dp(a_abs2.reindex(pd.MultiIndex.from_frame(DC[["kb", "ood"]])).to_numpy())
    a1cells = set(map(tuple, CC[["kb", "ood"]].drop_duplicates().to_numpy()))
    SC["in_a1"] = [(a, b) in a1cells for a, b in zip(SC.kb, SC.ood)]
    obs1 = SC[SC.outcome.str.startswith("a1:")].assign(contrast=lambda x: x.outcome.str[3:].str.split("@").str[0],
                                                        tau=lambda x: x.outcome.str.split("@").str[1])[["contrast", "tau", "kb", "ood", "ddp"]]
    CC = CC.merge(obs1, on=["contrast", "tau", "kb", "ood"], how="left")
    CC["resid"] = CC.ddp - CC.dd_m
    return SC, DC, CC, drop_lr, clip_sh, maxerr, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config_diag.json")))
    ap.add_argument("--runs", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = p(cfg, "results")
    (res / "tables").mkdir(parents=True, exist_ok=True)
    log = get_logger("A3.02", res / "logs" / "02_diagnose.log")
    D_full = pd.read_csv(p(cfg, "a0_processed", "tag_js.csv"), index_col=0)

    def fm(t):
        flag = "" if t["converged"] else " (nc)"
        return f"{t['slope']:+.3f} [{t['lo']:+.3f}, {t['hi']:+.3f}]{flag}"

    srows = []
    L = ["# A3 post hoc — RQ1 diagnosis: scale dependence and decomposition of the distance slope", "",
         "- Post hoc (after the A3/A5 results). Slopes: A3 interval of record (crossed random intercepts + slopes, REML, Wald 95%; (nc) = not converged, warnings in slopes.csv), predictor = KB–O tag-distance rank per 1 SD (> 0: far gains more, i.e. near gains less). G slopes: see A3_summary.md (not repeated here).",
         "- Scales: ddp = Δd′ (A3 metric; AUROC clipped at 1 − 1e-4); dA = ΔAUROC; dOP = reduction of OOD acceptance at the calibrated threshold (abs − rel, > 0 = gain; the opposite sign of A1's stored *_dOP); logratio = log(d′_rel / d′_abs) (multiplicative; draws with d′ ≤ 0 dropped).",
         "- Decomposition (binormal moment identity, test queries; midpoint split, invariant to the cosine origin): A2 moment Δd = T_var (noise cancellation) + T_push (mean push of the comparison term); pi = σ̄_abs/σ̄_rel − 1. RQ3 contrasts: variance part + ID-side cost + OOD-side push (common midpoint denominator). resid = ddp − moment Δd (non-binormality).",
         "- raw_id_lower / raw_ood_push: how much the (ii) comparison term exceeds the (iii) one on ID / OOD test queries, in raw cosine units (no scaling): the two effects of inserting a neighbour (ID scores lowered too; OOD scores pushed). Gain needs raw_ood_push > raw_id_lower.",
         "- profile: correlation of the KB and O test queries' mean similarity profiles over the banks other than KB and O. Diagnostic (uses OOD test queries); read as a competing embedding-space KB–O distance, not as a mechanism.", ""]
    runs = {k: v for k, v in cfg["runs"].items() if not args.runs or k in args.runs}
    sfx = "" if not args.runs else "_" + "_".join(runs)
    for rn, run in runs.items():
        SC, DC, CC, drop_lr, clip_sh, maxerr, hits = run_one(rn, run, cfg, D_full, log)
        SC.to_csv(res / "tables" / f"scale_cells_{rn}.csv", index=False)
        DC.to_csv(res / "tables" / f"decomp_cells_{rn}.csv", index=False)
        CC.to_csv(res / "tables" / f"contrast_cells_{rn}.csv", index=False)
        L += [f"## {rn}", "", f"Reproduction check: max |AUROC(recomputed) − stored| = {maxerr:.1e} over {hits} checks.", "",
              "### D1 scale dependence (tier means near / mid / far; slope on rank)", "",
              "| outcome | cells | scale | near | mid | far | slope |", "|---|---|---|---|---|---|---|"]
        for o, g in SC.groupby("outcome", sort=False):
            subsets = [("all", g)] + ([("A1 cells", g[g.in_a1])] if o.startswith("a2:site") and (~g.in_a1).any() else [])
            for lab, gg in subsets:
                for s_ in cfg["scales"]["list"] if lab == "all" else ["ddp", "dA"]:
                    tm = gg.groupby("tier")[s_].mean().reindex(TIERS)
                    sl = a3.mixed_slope(gg, s_, "x_rank", slopes=True)
                    srows.append({"run": rn, "part": "D1", "outcome": o, "cells": lab, "y": s_, **sl})
                    L.append(f"| {o} | {lab} ({len(gg)}) | {s_} | " + " | ".join(f"{v:+.4f}" for v in tm) + f" | {fm(sl)} |")
        L += ["", "Share of draws dropped from logratio (d′ ≤ 0): " + ", ".join(f"{k} {v:.3f}" for k, v in drop_lr.items()),
              "", "Share of draws with an AUROC at the clip (1 − 1e-4): " + ", ".join(f"{k} {v:.3f}" for k, v in clip_sh.items()), ""]
        L += ["### D2 decomposition, A2 site S_τ (m = 4)", "",
              "| τ | Spearman(moment Δd, ddp) | mean moment Δd / ddp | component | near | mid | far | slope |", "|---|---|---|---|---|---|---|---|"]
        for t, g in DC.groupby("tau", sort=False):
            rho = spearmanr(g.dd_m, g.ddp).statistic
            for comp in ("ddp", "dd_m", "resid", "T_var", "T_push", "pi", "d_abs_m", "d_abs_auc", "profile"):
                tm = g.groupby("tier")[comp].mean().reindex(TIERS)
                sl = a3.mixed_slope(g, comp, "x_rank", slopes=True)
                srows.append({"run": rn, "part": "D2", "outcome": f"a2:site@{t}", "cells": "all", "y": comp, **sl})
                L.append(f"| {t} | {rho:.3f} | {g.dd_m.mean():+.3f} / {g.ddp.mean():+.3f} | {comp} | " + " | ".join(f"{v:+.3f}" for v in tm) + f" | {fm(sl)} |")
            slc = a3.mixed_slope(g, "T_push", "x_rank", slopes=True, covar="profile")
            srows.append({"run": rn, "part": "D3", "outcome": f"a2:site@{t}", "cells": "all", "y": "T_push|profile", **slc})
            L.append(f"| {t} | | | T_push, profile as covariate (competing distance) | | | | {fm(slc)} |")
            L.append(f"| {t} | | | Spearman(profile, rank) = {spearmanr(g.profile, g.x_rank).statistic:+.3f}; Spearman(d_abs_m, rank) = {spearmanr(g.d_abs_m, g.x_rank).statistic:+.3f}; Spearman(pi, rank) = {spearmanr(g.pi, g.x_rank).statistic:+.3f} | | | | |")
        L += ["", "### D2 decomposition, A1 RQ3 contrasts", "",
              "| contrast | τ | Spearman(moment, ddp) | component | near | mid | far | slope |", "|---|---|---|---|---|---|---|---|"]
        for (cn, t), g in CC.groupby(["contrast", "tau"], sort=False):
            rho = spearmanr(g.dd_m, g.ddp, nan_policy="omit").statistic
            for comp in ("ddp", "dd_m", "resid", "C_var", "C_idcost", "C_oodpush", "raw_id_lower", "raw_ood_push"):
                tm = g.groupby("tier")[comp].mean().reindex(TIERS)
                sl = a3.mixed_slope(g, comp, "x_rank", slopes=True)
                srows.append({"run": rn, "part": "D2c", "outcome": f"a1:{cn}@{t}", "cells": "A1 cells", "y": comp, **sl})
                L.append(f"| {cn} | {t} | {rho:.3f} | {comp} | " + " | ".join(f"{v:+.3f}" for v in tm) + f" | {fm(sl)} |")
        L.append("")
        log.info(f"{rn} done")
    pd.DataFrame(srows).to_csv(res / "tables" / f"slopes{sfx}.csv", index=False)
    (res / f"A3_diag_summary{sfx}.md").write_text("\n".join(L) + "\n")
    meta = run_metadata(cfg, "A3.02_diagnose (post hoc)")
    meta["statsmodels"] = statsmodels.__version__
    meta["inputs"] = fingerprint([PROJECT_ROOT / r["emb"] / f for r in runs.values() for f in ("topk.npy", "q_index.parquet", "sites.json")]
                                 + [PROJECT_ROOT / r[a] / "tables" / f for r in runs.values() for a, f in
                                    (("a1", "rows.parquet"), ("a1", "draws.csv"), ("a1", "pairs.csv"), ("a2", "rows.parquet"), ("a2", "sets.csv"))]
                                 + [p(cfg, "a0_processed", "tag_js.csv")])
    save_json(meta, res / f"meta{sfx}.json")


if __name__ == "__main__":
    main()
