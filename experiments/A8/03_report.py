from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import PROJECT_ROOT, fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json, stable_seed

_spec = importlib.util.spec_from_file_location("a3_rq1", PROJECT_ROOT / "experiments" / "A3" / "01_rq1.py")
A3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A3)
TIERS = ["near", "mid", "far"]


def twoway(cells, col, B, seed):
    d = cells[["kb", "ood", col]].dropna()
    if len(d) == 0:
        return np.nan, np.nan, np.nan, 0
    sites = sorted(set(d.kb) | set(d.ood))
    si = {s: i for i, s in enumerate(sites)}
    ki, oi, v = d.kb.map(si).to_numpy(), d.ood.map(si).to_numpy(), d[col].to_numpy()
    rng = np.random.default_rng(seed)
    WK = rng.multinomial(len(sites), [1 / len(sites)] * len(sites), size=B).astype(float)
    WO = rng.multinomial(len(sites), [1 / len(sites)] * len(sites), size=B).astype(float)
    W = WK[:, ki] * WO[:, oi]
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (W @ v) / W.sum(1)
    lo, hi = np.nanquantile(bs, [0.025, 0.975])
    return float(v.mean()), float(lo), float(hi), int(len(v))


def fmt(t, nd=3):
    return f"{t[0]:+.{nd}f} [{t[1]:+.{nd}f}, {t[2]:+.{nd}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--base", default="experiments/A8", help="directory holding results_a1, results_a1_full, results_a2")
    ap.add_argument("--retrievers", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    a2c = cfg["a2"]
    B, clip, ceil = a2c["bootstrap_B"], a2c["dprime_clip"], a2c["ceiling"]
    res = p(cfg, "results")
    (res / "tables").mkdir(parents=True, exist_ok=True)
    log = get_logger("A8.03", res / "logs" / "03_report.log")
    dp = lambda a: np.sqrt(2) * norm.ppf(np.clip(a, clip, 1 - clip))
    R_ = args.retrievers or cfg["retrievers"]
    sinfo = load_json(p(cfg, "a0_processed", "sites.json"))
    cat = {s: v["category"] for s, v in sinfo.items()}
    D = pd.read_csv(p(cfg, "a0_processed", "tag_js.csv"), index_col=0)
    q_nb = load_json(p(cfg, "a0_processed", "bands.json"))["q_neighbor_value"]
    L = ["# A8 — CQADupStack replication (mixed design B; nopos), three models", "",
         f"- Design: {cfg['design']['_source']}", f"- A2 part: {cfg['a2']['_source']}",
         "- d′ = √2(Φ⁻¹(A) − Φ⁻¹(A_abs)), AUROC clipped at 1 − 1e-4; intervals: KB × O two-way bootstrap 95% (12 KB × 12 O sites).", ""]
    signs = []

    L += ["## 1. RQ3 (A1 core): d′ vs abs [two-way 95%]; nopos (main) | no mask (sensitivity)", "",
          "| model | score | (ii)−(iii) | (iii)−abs | (i)−(iii) | recovery | N1−Zm | γ_O (rank) | γ_KB (rank) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in R_:
        E = {v: pd.read_csv(PROJECT_ROOT / args.base / f"{d}/{r}/tables/estimands.csv") for v, d in (("nopos", "results_a1"), ("full", "results_a1_full"))}
        G = pd.read_csv(PROJECT_ROOT / args.base / f"results_a1/{r}/tables/regression.csv")
        for sc in ("S_tau@0.01", "S_tau@inf", "mix@0.5"):
            cells = []
            for est in ("main", "off", "leak", "recovery", "matched"):
                vals = []
                for v in ("nopos", "full"):
                    e = E[v]
                    q = e[(e.score == sc) & (e.estimand == est) & (e.metric == ("ddp" if est != "recovery" else e[(e.score == sc) & (e.estimand == est)].metric.iloc[0]))]
                    vals.append(f"{q.point.iloc[0]:+.3f}" + (f" [{q.lo.iloc[0]:+.3f}, {q.hi.iloc[0]:+.3f}]" if v == "nopos" and est != "recovery" else ""))
                    if v == "nopos" and est in ("main", "off"):
                        signs.append({"model": r, "result": f"RQ3 {est} {sc}", "point": q.point.iloc[0], "lo": q.lo.iloc[0], "hi": q.hi.iloc[0]})
                cells.append(" | ".join(vals[:1]) + f" ‖ {vals[1]}")
            gg = G[(G.score == sc) & (G.distance == "rank")].set_index("coef")
            for c_ in ("gamma_O", "gamma_KB"):
                cells.append(f"{gg.loc[c_, 'point']:+.3f} [{gg.loc[c_, 'lo']:+.3f}, {gg.loc[c_, 'hi']:+.3f}]")
                signs.append({"model": r, "result": f"RQ3 {c_} {sc}", "point": gg.loc[c_, "point"], "lo": gg.loc[c_, "lo"], "hi": gg.loc[c_, "hi"]})
            L.append(f"| {r} | {sc} | " + " | ".join(cells) + " |")
    set12 = all("Set mode: the bootstrap resamples" in (PROJECT_ROOT / args.base / f"{d}/{r}/A1_summary.md").read_text() for r in R_ for d in ("results_a1", "results_a1_full"))
    boot_note = ("in set mode its two-way bootstrap resamples the 12 kb_set sites (changed 2026-10-07; the earlier 98-column multinomial tables are kept as *_98col.csv, comparison in A8_extras.md §3)"
                 if set12 else "its two-way bootstrap draws multinomial weights over all 98 columns (Poisson-like for the 12 KB / O sites; 4–7% wider than the 12-site resampling of §2–3)")
    L += ["", "(each estimand cell: nopos value [interval] ‖ no-mask value). Intervals in §1 come from A1/03_analyze; " + boot_note + ". Its rank predictors are computed over the 98 columns (twin and cqa: duplicates shift slot ranks by 1–12; within-draw correlation with the 85-site rank ≈ 0.9995).", ""]
    mc = pd.read_csv(p(cfg, "a0_processed", "mask_counts.csv"))
    L += [f"Mask reach: {int(mc.queries_affected.sum())} of {int(mc.queries.sum())} queries ({mc.queries_affected.sum() / mc.queries.sum():.1%}; per forum " + ", ".join(f"{a}/{b}" for a, b in zip(mc.queries_affected, mc.queries)) + ") have a masked document, so the nopos ‖ no-mask contrast is diluted by construction.", "",
          "| model | KB test queries whose s_KB changes | mean drop of s_KB on them (ID-sd of s_KB) | mean A_abs nopos / no mask |", "|---|---|---|---|"]
    for r in R_:
        T1, T0 = np.load(p(cfg, "emb", r, "topk.npy"), mmap_mode="r"), np.load(p(cfg, "emb_full", r, "topk.npy"), mmap_mode="r")
        qi = pd.read_parquet(p(cfg, "emb", r, "q_index.parquet"))
        sites = load_json(p(cfg, "emb", r, "sites.json"))
        ch, drops, aa = [], [], {}
        for kb in sorted(qi.site.unique()):
            ii = qi.index[(qi.site == kb) & (qi.role == "test")].to_numpy()
            j = sites.index(kb)
            s1, s0 = np.asarray(T1[ii, j]).mean(1), np.asarray(T0[ii, j]).mean(1)
            d = s0 - s1
            ch.append(int((d > 0).sum()))
            drops += list(d[d > 0] / s1.std())
        for v, dd in (("nopos", "results_a1"), ("full", "results_a1_full")):
            aa[v] = pd.read_parquet(PROJECT_ROOT / args.base / f"{dd}/{r}/tables/contrasts.parquet", columns=["kb", "ood", "a_abs", "score"]).query("score == 'S_tau@0.01'").groupby(["kb", "ood"]).a_abs.mean().mean()
        L.append(f"| {r} | {sum(ch)} | {np.mean(drops):.3f} | {aa['nopos']:.4f} / {aa['full']:.4f} |")
    L.append("")

    L += ["## 2. RQ2 (A2 core, m = 4): mean [two-way 95%]", ""]
    rows_all = {}
    for r in R_:
        df = pd.read_parquet(PROJECT_ROOT / args.base / f"results_a2/{r}/tables/rows.parquet")
        df["param"] = df.param.astype(str)
        df["ddp"] = dp(df.auroc.to_numpy()) - dp(df.auroc_abs.to_numpy())
        df["dA"] = df.auroc - df.auroc_abs
        df["key"] = df.corpus + ":" + df.partition + ":" + df.family + "@" + df.param
        W = df.pivot_table(index=["kb", "ood", "draw"], columns="key", values="ddp", aggfunc="first")
        C = pd.DataFrame(index=W.index)
        for t in ("0.01", "inf"):
            C[f"site@{t}"] = W[f"se:site:S_tau@{t}"]
            C[f"random@{t}"] = W[f"se:random:S_tau@{t}"]
            C[f"site-random@{t}"] = C[f"site@{t}"] - C[f"random@{t}"]
            C[f"msmix1-site@{t}"] = W["ms:pooled:mix@1"] - C[f"site@{t}"]
        C["site-mix1@0.01"] = C["site@0.01"] - W["se:pooled:mix@1"]
        C["random-mix1@0.01"] = C["random@0.01"] - W["se:pooled:mix@1"]
        for a in ("0.5", "1"):
            C[f"se_mix@{a}"] = W[f"se:pooled:mix@{a}"]
            C[f"ms_mix@{a}"] = W[f"ms:pooled:mix@{a}"]
            C[f"se-ms_mix@{a}"] = C[f"se_mix@{a}"] - C[f"ms_mix@{a}"]
        C = C.reset_index()
        sets = pd.read_csv(PROJECT_ROOT / args.base / f"results_a2/{r}/tables/sets.csv").set_index(["kb", "ood", "draw"])
        tw = lambda s: sinfo[s].get("twin", s)
        nb = lambda s, x: cat[s] == cat[tw(x)] and D.loc[s, tw(x)] <= q_nb
        C["has_O_nb"] = [any(nb(s, o) for s in sets.loc[(kb, o, d), "order"].split("+")) for kb, o, d in zip(C.kb, C.ood, C.draw)]
        C["has_KB_nb"] = [any(nb(s, kb) for s in sets.loc[(kb, o, d), "order"].split("+")) for kb, o, d in zip(C.kb, C.ood, C.draw)]
        rows_all[r] = df
        cells = C.groupby(["kb", "ood"]).mean(numeric_only=True).reset_index()
        cells.to_csv(res / "tables" / f"rq2_cells_{r}.csv", index=False)
        L += [f"### {r}", "", "| quantity | mean [95%] |", "|---|---|"]
        for col in [c for c in cells.columns if c not in ("kb", "ood", "draw", "has_O_nb", "has_KB_nb")]:
            t = twoway(cells, col, B, stable_seed(cfg["seed"], "A8_boot", r, col))
            L.append(f"| {col} | {fmt(t)} |")
            if col in ("site-random@0.01", "site-random@inf", "site-mix1@0.01", "ms_mix@1", "se_mix@1"):
                signs.append({"model": r, "result": f"RQ2 {col}", "point": t[0], "lo": t[1], "hi": t[2]})
        L += ["", "| stratum (comparison set has) | share of draws | site@0.01 | site − random @inf |", "|---|---|---|---|"]
        for lab, msk in (("O-nb only", C.has_O_nb & ~C.has_KB_nb), ("neither", ~C.has_O_nb & ~C.has_KB_nb), ("both", C.has_O_nb & C.has_KB_nb), ("KB-nb only", ~C.has_O_nb & C.has_KB_nb)):
            sub = C[msk].groupby(["kb", "ood"]).mean(numeric_only=True).reset_index()
            t1 = twoway(sub, "site@0.01", B, stable_seed(cfg["seed"], "A8_boot_s", r, lab, 1))
            t2 = twoway(sub, "site-random@inf", B, stable_seed(cfg["seed"], "A8_boot_s", r, lab, 2))
            L.append(f"| {lab} | {msk.mean():.2f} | {fmt(t1)} | {fmt(t2)} |")
        L.append("")

    L += ["## 3. RQ1 (agreed order: G first, then d′ and AUROC); slope per 1 SD of the KB–O rank (> 0: far gains more), A3 interval of record", ""]
    srows = []
    for r in R_:
        df = rows_all[r]
        pairs = pd.read_csv(PROJECT_ROOT / args.base / f"results_a1/{r}/tables/pairs.csv").set_index(["kb", "ood"])
        a1c = pd.read_parquet(PROJECT_ROOT / args.base / f"results_a1/{r}/tables/contrasts.parquet")
        tot = pairs.tier.value_counts().reindex(TIERS).fillna(0).astype(int)

        def slopes(cell, y, r_, on_, lab_):
            parts = []
            for nm, fn in (("RE", lambda: A3.mixed_slope(cell, y, "x_rank", slopes=True)), ("RI", lambda: A3.mixed_slope(cell, y, "x_rank", slopes=False)),
                           ("OLS", lambda: {**A3.ols_boot(cell, y, "x_rank", sorted(set(cell.kb) | set(cell.ood)), B, stable_seed(cfg["seed"], "A8_ols", r_, on_, lab_)), "converged": True, "warnings": ""})):
                try:
                    sl = fn()
                except Exception as e:
                    parts.append(f"{nm} n/a ({type(e).__name__})")
                    continue
                flag = ("" if sl.get("converged", True) else " nc") + (" bd" if "boundary" in str(sl.get("warnings", "")) else "")
                parts.append(f"{nm} {sl['slope']:+.3f} [{sl['lo']:+.3f}, {sl['hi']:+.3f}]{flag}")
                srows.append({"model": r_, "outcome": on_, "measure": lab_, "model_type": nm, **{k_: v_ for k_, v_ in sl.items() if k_ in ("slope", "lo", "hi", "n", "converged", "warnings")}})
            return "; ".join(parts)
        L += [f"### {r}", "", "Slopes: RE = A3 interval of record (crossed random intercepts + slopes; nc = not converged, bd = boundary warning; 12 x 12 sites, Wald intervals likely anti-conservative), RI = random intercepts only, OLS = OLS slope with the 12-site two-way bootstrap.", "",
              "| outcome | measure | cells kept / total near · mid · far | near | mid | far | slopes (rank) |", "|---|---|---|---|---|---|---|"]
        outs = {"site@0.01": ("se", "site", "S_tau", "0.01"), "site@inf": ("se", "site", "S_tau", "inf"),
                "mix@1": ("se", "pooled", "mix", "1"), "mix@0.5": ("se", "pooled", "mix", "0.5")}
        for on, (src, part, fam, par) in outs.items():
            q = df[(df.corpus == src) & (df.partition == part) & (df.family == fam) & (df.param == par)]
            for meas, cut in (("G", 0.98), ("G", 0.95), ("G", 0.99), ("ddp", None), ("dA", None)):
                if meas == "G":
                    qq = q[q.auroc_abs <= cut].assign(G=lambda x: (x.auroc - x.auroc_abs) / (1 - x.auroc_abs))
                    cell = qq.groupby(["kb", "ood"]).G.mean().rename("y").reset_index()
                    lab = f"G (A_abs ≤ {cut})"
                else:
                    cell = q.groupby(["kb", "ood"])[meas].mean().rename("y").reset_index()
                    lab = meas
                cell["x_rank"] = [float(pairs.loc[(a, b), "rank_kb_o"]) for a, b in zip(cell.kb, cell.ood)]
                cell["tier"] = [pairs.loc[(a, b), "tier"] for a, b in zip(cell.kb, cell.ood)]
                kept = cell.tier.value_counts().reindex(TIERS).fillna(0).astype(int)
                tm = cell.groupby("tier").y.mean().reindex(TIERS)
                L.append(f"| {on} | {lab} | " + " / ".join(f"{kept[t]}/{tot[t]}" for t in TIERS) + " | " + " | ".join(f"{v:+.3f}" for v in tm)
                         + " | " + slopes(cell, "y", r, on, lab) + " |")
        for est in ("main", "matched"):
            for t in ("0.01", "inf"):
                g = a1c[a1c.score == f"S_tau@{t}"].groupby(["kb", "ood"])[[f"{est}_ddp", f"{est}_dA"]].mean().reset_index()
                g["x_rank"] = [float(pairs.loc[(a, b), "rank_kb_o"]) for a, b in zip(g.kb, g.ood)]
                g["tier"] = [pairs.loc[(a, b), "tier"] for a, b in zip(g.kb, g.ood)]
                for meas in ("ddp", "dA"):
                    tm = g.groupby("tier")[f"{est}_{meas}"].mean().reindex(TIERS)
                    L.append(f"| A1 {est}@{t} | {meas} | {len(g)} cells | " + " | ".join(f"{v:+.3f}" for v in tm)
                             + " | " + slopes(g, f"{est}_{meas}", r, f"A1 {est}@{t}", meas) + " |")
        L.append("")
    pd.DataFrame(srows).to_csv(res / "tables" / "rq1_slopes.csv", index=False)

    S = pd.DataFrame(signs)
    L += ["## 4. Cross-model signs of core results (interval excludes 0: + / −; else 0)", "", "| result | " + " | ".join(R_) + " |", "|---|" + "---|" * len(R_)]
    for res_, g in S.groupby("result", sort=False):
        g = g.set_index("model")
        L.append(f"| {res_} | " + " | ".join(f"{g.loc[r, 'point']:+.3f} ({'+' if g.loc[r, 'lo'] > 0 else ('−' if g.loc[r, 'hi'] < 0 else '0')})" for r in R_) + " |")
    (res / "A8_summary.md").write_text("\n".join(L) + "\n")
    meta = run_metadata(cfg, "A8.03_report")
    meta["inputs"] = fingerprint([PROJECT_ROOT / args.base / f"{d}/{r}/tables/{f}" for r in R_ for d, f in
                                  (("results_a1", "estimands.csv"), ("results_a1", "regression.csv"), ("results_a1", "contrasts.parquet"), ("results_a1", "pairs.csv"),
                                   ("results_a1_full", "estimands.csv"), ("results_a2", "rows.parquet"), ("results_a2", "sets.csv"))])
    save_json(meta, res / "meta_03.json")
    log.info("summary written")


if __name__ == "__main__":
    main()
