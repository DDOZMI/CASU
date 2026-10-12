from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.formula.api as smf
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import PROJECT_ROOT, fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json, stable_seed


def a2_key(o):
    part, par = o.split(":", 2)[2].split("@")
    return ("pooled", part[3:], par) if part.startswith("se_") else (part, "S_tau", par)


def cells_for_run(run, cfg, D_full, cat):
    root = PROJECT_ROOT
    sites = load_json(root / run["sites"])
    a1 = pd.read_parquet(root / run["a1"] / "tables" / "contrasts.parquet")
    a2 = pd.read_parquet(root / run["a2"] / "tables" / "contrasts.parquet")
    a2 = a2[a2.m == cfg["m"]]
    nei = a2[~a2.has_O_neighbour & ~a2.has_KB_neighbour]
    out = {}
    for o in cfg["outcomes"]["primary"]:
        col = o.split(":", 1)[1]
        out[o] = a2.groupby(["kb", "ood"])[col].mean()
        out["neither:" + o] = nei.groupby(["kb", "ood"])[col].mean()
    for o in cfg["outcomes"]["secondary"]:
        col, score = o.split(":", 1)[1].split("@", 1)
        out[o] = a1[a1.score == score].groupby(["kb", "ood"])[col].mean()
    rows = pd.read_parquet(root / run["a2"] / "tables" / "rows.parquet",
                           columns=["kb", "ood", "draw", "m", "corpus", "partition", "family", "param", "auroc", "auroc_abs"])
    rows = rows[(rows.m == cfg["m"]) & (rows.corpus == "se")]
    rows["param"] = rows["param"].astype(str).str.replace(r"\.0$", "", regex=True)
    for o in cfg["outcomes"]["primary"]:
        part, fam, par = a2_key(o)
        q = rows[(rows.partition == part) & (rows.family == fam) & (rows.param == par)]
        g = np.where(q.auroc_abs <= cfg["ceiling"], (q.auroc - q.auroc_abs) / (1 - q.auroc_abs), np.nan)
        out["G:" + o] = q.assign(G=g).groupby(["kb", "ood"]).G.mean()
    a_abs = rows[rows.family == "S_tau"].groupby(["kb", "ood"]).auroc_abs.first()
    C = pd.DataFrame(out).reset_index()
    C["a_abs"] = [a_abs[(a, b)] for a, b in zip(C.kb, C.ood)]
    C["probit_abs"] = norm.ppf(np.clip(C.a_abs, 1e-4, 1 - 1e-4))
    sidx = {s: i for i, s in enumerate(sites)}
    D = D_full.loc[sites, sites].to_numpy()
    rankm = np.argsort(np.argsort(D + np.eye(len(sites)) * 9, axis=1, kind="stable"), axis=1, kind="stable") + 1
    ki, oi = C.kb.map(sidx).to_numpy(), C.ood.map(sidx).to_numpy()
    C["x_rank"], C["x_tag"] = rankm[ki, oi].astype(float), D[ki, oi]
    C["x_samecat"] = [float(cat[a] == cat[b]) for a, b in zip(C.kb, C.ood)]
    tiers = pd.read_csv(root / run["a1"] / "tables" / "pairs.csv").set_index(["kb", "ood"]).tier
    C["tier"] = [tiers.get((a, b), np.nan) for a, b in zip(C.kb, C.ood)]
    return C, sites


def mixed_slope(cells, y, x, slopes=True, covar=None):
    cols = ["kb", "ood", y, x] + ([covar] if covar else [])
    d = cells[cols].dropna().rename(columns={y: "y", x: "x"})
    if x != "x_samecat":
        d["x"] = (d.x - d.x.mean()) / d.x.std(ddof=0)
    d["g"] = 1
    vc = {"kb": "0 + C(kb)", "ood": "0 + C(ood)"}
    if slopes:
        vc.update({"kbx": "0 + C(kb):x", "oodx": "0 + C(ood):x"})
    form = "y ~ x" + (f" + {covar}" if covar else "")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = smf.mixedlm(form, d, groups="g", re_formula="0", vc_formula=vc).fit(reml=True)
    ci = m.conf_int().loc["x"].to_numpy()
    vcomp = dict(zip(m.model.exog_vc.names, np.round(m.vcomp, 6)))
    return {"slope": float(m.params["x"]), "lo": float(ci[0]), "hi": float(ci[1]), "n": int(len(d)), "converged": bool(m.converged),
            "n_warnings": len(w), "warnings": "; ".join(sorted({type(x_.message).__name__ + ": " + str(x_.message)[:80] for x_ in w})),
            "vcomp": str(vcomp), "resid_var": float(m.scale)}


def ols_boot(C, y, x, sites, B, seed):
    d = C[["kb", "ood", y, x]].dropna()
    xv = d[x].to_numpy(float)
    if x != "x_samecat":
        xv = (xv - xv.mean()) / xv.std()
    yv = d[y].to_numpy(float)
    sidx = {s: i for i, s in enumerate(sites)}
    ki, oi = d.kb.map(sidx).to_numpy(), d.ood.map(sidx).to_numpy()
    rng = np.random.default_rng(seed)
    n = len(sites)
    CK = rng.multinomial(n, [1 / n] * n, size=B).astype(float)
    CO = rng.multinomial(n, [1 / n] * n, size=B).astype(float)
    W = CK[:, ki] * CO[:, oi]
    sw = W.sum(1)
    mx, my = (W @ xv) / sw, (W @ yv) / sw
    bs = ((W @ (xv * yv)) / sw - mx * my) / ((W @ (xv * xv)) / sw - mx * mx)
    b = np.cov(xv, yv, bias=True)[0, 1] / np.var(xv)
    lo, hi = np.quantile(bs[np.isfinite(bs)], [0.025, 0.975])
    return {"slope": float(b), "lo": float(lo), "hi": float(hi), "n": int(len(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = p(cfg, "results")
    (res / "tables").mkdir(parents=True, exist_ok=True)
    log = get_logger("A3.01", res / "logs" / "01_rq1.log")
    D_full = pd.read_csv(p(cfg, "a0_processed", "tag_js.csv"), index_col=0)
    cat = {s: v["category"] for s, v in load_json(p(cfg, "a0_processed", "sites.json")).items()}
    prim = cfg["outcomes"]["primary"]
    outcomes = prim + cfg["outcomes"]["secondary"] + ["G:" + o for o in prim] + ["neither:" + o for o in prim]
    srows = []
    fm = lambda t: f"{t['slope']:+.3f} [{t['lo']:+.3f}, {t['hi']:+.3f}]"
    L = ["# A3 — RQ1: gain of relative judgment vs KB–O distance", "",
         "- Δd′ vs the absolute score per (KB, O) cell (mean over draws). Interval of record: y ~ x + (1|KB) + (1|O) + (0+x|KB) + (0+x|O), REML, Wald 95%; reference: random intercepts only; robustness: OLS two-way (KB x O) bootstrap. Slope per 1 SD (rank, tag) or per same-category switch.",
         f"- Reading rule (fixed before the run): {cfg['reading_rule']}",
         f"- Outcomes (config): A2 m = {cfg['m']} site S_τ and pooled mix (primary; blind); A1 (ii), (iii) − abs, (ii) − (iii), N1 − Zm (secondary; A1 tier tables were already seen); G on A_abs ≤ {cfg['ceiling']}; primary outcomes on 'neither' draws (no O- or KB-neighbour in the comparison set).",
         f"- {cfg.get('runs_note', '')}", ""]
    for rn, run in cfg["runs"].items():
        C, sites = cells_for_run(run, cfg, D_full, cat)
        C.to_csv(res / "tables" / f"cells_{rn}.csv", index=False)
        L += [f"## {rn} ({len(C)} cells)", "", "### Slopes (interval of record: random intercepts + slopes)", "",
              "| outcome | mean | near | mid | far | rank | tag | samecat | rank (intercepts only) | rank (OLS, 2-way boot) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for o in outcomes:
            if o not in C or C[o].notna().sum() < 30:
                continue
            r_ = {}
            for x in cfg["predictors"]["all"]:
                r_[x] = mixed_slope(C, o, f"x_{x}", slopes=True)
                srows.append({"run": rn, "outcome": o, "predictor": x, "model": "mixed_slopes", **r_[x]})
            ri = mixed_slope(C, o, "x_rank", slopes=False)
            srows.append({"run": rn, "outcome": o, "predictor": "rank", "model": "mixed_intercepts", **ri})
            bo = ols_boot(C, o, "x_rank", sites, cfg["bootstrap_B"], stable_seed(cfg["seed"], "A3_boot", rn, o))
            srows.append({"run": rn, "outcome": o, "predictor": "rank", "model": "ols_2way_boot", **bo})
            tm = C.groupby("tier")[o].mean()
            L.append(f"| {o} | {C[o].mean():+.3f} | {tm.get('near', np.nan):+.3f} | {tm.get('mid', np.nan):+.3f} | {tm.get('far', np.nan):+.3f} | "
                     f"{fm(r_['rank'])} | {fm(r_['tag'])} | {fm(r_['samecat'])} | {fm(ri)} | {fm(bo)} |")
        L += ["", "### Ceiling sensitivities (primary outcomes, rank, random intercepts + slopes)", "",
              "| outcome | all cells | A_abs ≤ ceiling | + probit(A_abs) covariate |", "|---|---|---|---|"]
        for o in prim:
            a = mixed_slope(C, o, "x_rank")
            b = mixed_slope(C[C.a_abs <= cfg["ceiling"]], o, "x_rank")
            c = mixed_slope(C, o, "x_rank", covar="probit_abs")
            for nm, t in (("ceiling_subset", b), ("probit_covariate", c)):
                srows.append({"run": rn, "outcome": o, "predictor": "rank", "model": f"mixed_slopes_{nm}", **t})
            L.append(f"| {o} | {fm(a)} ({a['n']}) | {fm(b)} ({b['n']}) | {fm(c)} |")
        L += ["", f"- A_abs median by tier: " + ", ".join(f"{t} {C[C.tier == t].a_abs.median():.3f}" for t in ("near", "mid", "far")), ""]
        log.info(f"{rn}: {len(C)} cells")
    S = pd.DataFrame(srows)
    S.to_csv(res / "tables" / "slopes.csv", index=False)
    mm = S[S.model.str.startswith("mixed")]
    L.append(f"- mixed fits: {len(mm)}; not converged: {int((~mm.converged.astype(bool)).sum())}; with warnings: {int((mm.n_warnings > 0).sum())} (tables/slopes.csv: warnings, vcomp)")
    (res / "A3_summary.md").write_text("\n".join(L) + "\n")
    meta = run_metadata(cfg, "A3.01_rq1")
    files = []
    for r_ in cfg["runs"].values():
        files += [PROJECT_ROOT / r_["a1"] / "tables" / f for f in ("contrasts.parquet", "pairs.csv")]
        files += [PROJECT_ROOT / r_["a2"] / "tables" / f for f in ("contrasts.parquet", "rows.parquet")]
        files.append(PROJECT_ROOT / r_["sites"])
    meta["inputs"] = fingerprint(files + [p(cfg, "a0_processed", "tag_js.csv"), p(cfg, "a0_processed", "sites.json")])
    meta["statsmodels"] = statsmodels.__version__
    save_json(meta, res / "meta.json")


if __name__ == "__main__":
    main()
