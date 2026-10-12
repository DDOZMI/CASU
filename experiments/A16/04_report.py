from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm, rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import PROJECT_ROOT as PROJECT, fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json, stable_seed

SCORES = [("S_tau", "0.01"), ("S_tau", "inf"), ("mix", "0.5")]


def auroc(neg, pos):
    r = rankdata(np.concatenate([neg, pos]))
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config_a1.json")))
    ap.add_argument("--res", default=None)
    ap.add_argument("--retriever", default=None, help="default: config detector (model generality runs)")
    ap.add_argument("--nbcat-res", default=None, help="default: <res parent>/results_nbcat")
    ap.add_argument("--tree-res", default=None, help="default: <res parent>/results_tree")
    args = ap.parse_args()
    cfg = load_config(args.config)
    r = args.retriever or cfg["detector"]
    res = Path(args.res) if args.res else (p(cfg, "results", r) if cfg.get("results_per_retriever") else p(cfg, "results"))
    (res / "tables").mkdir(parents=True, exist_ok=True)
    log = get_logger("A16.04", res / "logs" / "04_report.log")
    clip = cfg["metrics"]["dprime_clip"]
    dp = lambda a: np.sqrt(2) * norm.ppf(np.clip(a, clip, 1 - clip))
    B, k, m = cfg["metrics"]["bootstrap_B"], cfg["k"], cfg["comparison"]["m"]
    proc = p(cfg, "a0_processed")
    edir = p(cfg, "emb", r)
    sites = load_json(edir / "sites.json")
    sidx = {s: i for i, s in enumerate(sites)}
    nS = len(sites)
    D = pd.read_csv(proc / cfg["distance_file"], index_col=0).loc[sites, sites].to_numpy()
    Tr = pd.read_csv(proc / "tree_dist.csv", index_col=0).loc[sites, sites].to_numpy()
    C2 = np.load(edir / f"centroids_{cfg['second_distance_encoder'][r]}.npy")
    D2 = 1 - C2 @ C2.T
    rep, bands = load_json(proc / "build_report.json"), load_json(proc / "bands.json")
    sinfo = load_json(proc / "sites.json")

    rng = np.random.default_rng(stable_seed(cfg["seed"], "A16_extra_boot"))
    CK = rng.multinomial(nS, [1 / nS] * nS, size=B).astype(float)
    CO = rng.multinomial(nS, [1 / nS] * nS, size=B).astype(float)

    def boot(cells):
        if len(cells) == 0:
            return np.nan, np.nan, np.nan, 0
        ki, oi, v = cells.kb.map(sidx).to_numpy(), cells.ood.map(sidx).to_numpy(), cells.v.to_numpy()
        W = CK[:, ki] * CO[:, oi]
        ok = W.sum(1) > 0
        est = (W[ok] @ v) / W[ok].sum(1)
        return float(v.mean()), *np.quantile(est, [0.025, 0.975]), len(cells)

    f = lambda t: "—" if t[3] == 0 else f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}] ({t[3]})"
    L = [f"# A16 — Amazon product QA: additions to the A1 / A10 outputs (detector {r})", "",
         "- Plan, decisions and pre-run forecasts: experiments/A16/PLAN.md, JOURNAL 2026-10-08 14:40. A1-format results: results/A1_summary.md; moments: results/A10_summary.md.",
         "- Site category = Amazon top-level node; the A1 summary's SE category strata do not apply (config ood_cat_keep = []). 'tag' in the A1 regression table = raw co-purchase JS.", ""]
    L += ["## 1. Build", "", f"- sites {rep['n_sites']} of {rep['n_candidate_sites']} candidates, {rep['n_categories']} top-level categories",
          "- dropped: " + "; ".join(f"{k_} ({v['reason']}; questions after dedup {v['n_within_dedup'] - v['n_cross_removed']}, query questions available {v['n_query_available']}, bank pool {v['bank_pool']})" for k_, v in rep["dropped"].items()),
          f"- cross-site duplicate normalised questions {rep['cross_site_duplicate_norms']}, questions removed over candidate sites {rep['cross_site_removed_total_candidates']} (final sites {rep['cross_site_removed_final_sites']}); unmapped also_bought targets: {rep['unmapped_target_share']:.3f}",
          f"- co-purchase JS bands: neighbour ≤ {bands['q_neighbor_value']:.4f} (q10, + same category), far > {bands['q_far_value']:.4f} (q50); share of pairs > 0.95: {bands['share_d_gt_0.95']:.3f}; quantiles 5/25/50/75/95: {', '.join(f'{x:.3f}' for x in bands['d_quantiles_5_25_50_75_95'])}",
          f"- tree-distance pair counts (1 sibling / 2 same top / 3 other): {rep['tree_dist_counts']}; test query types: {rep['test_qtype_share']}",
          f"- query products per site (median): {np.median([v['n_query_products'] for v in sinfo.values()]):.0f}", ""]
    iu = np.triu_indices(nS, 1)
    agree = {"JS vs tree": spearmanr(D[iu], Tr[iu])[0], "JS vs 2nd-encoder": spearmanr(D[iu], D2[iu])[0], "tree vs 2nd-encoder": spearmanr(Tr[iu], D2[iu])[0]}
    L += ["## 2. Site-distance agreement (Spearman over site pairs)", "", "; ".join(f"{k_}: {v:+.3f}" for k_, v in agree.items()), ""]
    pd.DataFrame([agree]).to_csv(res / "tables" / "extra_distance_agreement.csv", index=False)

    Cn = pd.read_parquet(res / "tables" / "contrasts.parquet")
    Cn["tree_kb_o"] = [Tr[sidx[a], sidx[b]] for a, b in zip(Cn.kb, Cn.ood)]
    Cn["tree_n1_o"] = [Tr[sidx[a], sidx[b]] for a, b in zip(Cn.N1, Cn.ood)]
    srows = []
    L += ["## 3. (ii)−(iii) Δd′ by tree relation (n = cells)", "", "| score | KB–O sibling | KB–O same top | KB–O other top | N1–O sibling | N1–O same top | N1–O other top |", "|---|---|---|---|---|---|---|"]
    for fam, par in SCORES:
        g = Cn[Cn.score == f"{fam}@{par}"]
        cells = []
        for col in ("tree_kb_o", "tree_n1_o"):
            for tv in (1, 2, 3):
                sub = g[g[col] == tv].groupby(["kb", "ood"]).main_ddp.mean().rename("v").reset_index()
                t = boot(sub)
                srows.append({"score": f"{fam}@{par}", "stratum": f"{col}={tv}", "point": t[0], "lo": t[1], "hi": t[2], "n_cells": t[3]})
                cells.append(f(t))
        L.append(f"| {fam}@{par} | " + " | ".join(cells) + " |")
    pd.DataFrame(srows).to_csv(res / "tables" / "extra_tree_strata.csv", index=False)

    qi = pd.read_parquet(edir / "q_index.parquet")
    it = pd.read_parquet(proc / "items.parquet", columns=["site", "role", "pos", "qtype"]).set_index(["site", "role", "pos"]).qtype
    qtype = it.reindex(pd.MultiIndex.from_frame(qi[["site", "role", "pos"]])).to_numpy()
    T = np.asarray(np.load(edir / "topk.npy", mmap_mode="r"))
    Lm = T.mean(2).astype(np.float64)
    rows_of = {kk: g.index.to_numpy() for kk, g in qi.groupby(["site", "role"])}
    dr = pd.read_csv(res / "tables" / "draws.csv")
    qrows = []
    for (kb, o), g in dr.groupby(["kb", "ood"]):
        i_id, i_ood = rows_of[(kb, "test")], rows_of[(o, "test")]
        for qt in ("yes/no", "open-ended"):
            a_, b_ = i_id[qtype[i_id] == qt], i_ood[qtype[i_ood] == qt]
            if len(a_) < 50 or len(b_) < 50:
                continue
            sel = np.r_[a_, b_]
            s = Lm[sel, sidx[kb]]
            A0 = auroc(-s[: len(a_)], -s[len(a_):])
            for d, gd in g.groupby("draw"):
                F = gd.fixed.iloc[0].split("+")
                rl = dict(zip(gd.role, gd.slot))
                for fam, par in SCORES:
                    v = {}
                    for ro in ("N1", "N2", "Z1", "Z2"):
                        Ci = [sidx[x] for x in (*F, rl[ro])]
                        if fam == "S_tau":
                            sA = Lm[sel][:, Ci]
                            S = s - (sA.mean(1) if par == "inf" else float(par) * (logsumexp(sA / float(par), axis=1) - np.log(m)))
                        else:
                            U = T[sel][:, Ci, :].reshape(len(sel), -1).astype(np.float64)
                            S = s - float(par) * (-np.partition(-U, k - 1, axis=1)[:, :k]).mean(1)
                        v[ro] = dp(auroc(-S[: len(a_)], -S[len(a_):])) - dp(A0)
                    qrows.append({"kb": kb, "ood": o, "draw": d, "qtype": qt, "score": f"{fam}@{par}", "n_id": len(a_), "n_ood": len(b_),
                                  "auroc_abs": A0, "main_ddp": (v["N1"] + v["N2"]) / 2 - (v["Z1"] + v["Z2"]) / 2})
    Q = pd.DataFrame(qrows)
    Q.to_parquet(res / "tables" / "extra_qtype_draws.parquet", index=False)
    L += ["", "## 4. (ii)−(iii) Δd′ by question type (ID and OOD test queries of one type; cells with ≥ 50 of each; n = cells)", "",
          "| score | yes/no | open-ended | paired difference yes/no − open-ended | mean abs AUROC yes/no / open-ended |", "|---|---|---|---|---|"]
    qsum = []
    for sc_ in [f"{a}@{b}" for a, b in SCORES]:
        g = Q[Q.score == sc_].groupby(["kb", "ood", "qtype"]).agg(v=("main_ddp", "mean"), a=("auroc_abs", "first")).reset_index()
        t1, t2 = boot(g[g.qtype == "yes/no"]), boot(g[g.qtype == "open-ended"])
        w = g.pivot_table(index=["kb", "ood"], columns="qtype", values="v").dropna()
        t3 = boot(pd.DataFrame({"kb": w.index.get_level_values(0), "ood": w.index.get_level_values(1), "v": (w["yes/no"] - w["open-ended"]).to_numpy()}))
        aa = g.groupby("qtype").a.mean()
        L.append(f"| {sc_} | {f(t1)} | {f(t2)} | {f(t3)} | {aa.get('yes/no', np.nan):.3f} / {aa.get('open-ended', np.nan):.3f} |")
        qsum.append({"score": sc_, "yesno": t1, "open": t2, "diff": t3})
    pd.DataFrame(qsum).to_csv(res / "tables" / "extra_qtype.csv", index=False)
    qnb = bands["q_neighbor_value"]
    L += ["", "## 5. Review diagnostics (2026-10-08)", ""]
    for nm_, rd in (("primary (band without category)", res), ("A1 rule (same top-level node), results_nbcat", Path(args.nbcat_res) if args.nbcat_res else res.parent / "results_nbcat")):
        if not (rd / "tables" / "draws.csv").exists():
            L.append(f"- {nm_}: not available")
            continue
        dd = pd.read_csv(rd / "tables" / "draws.csv")
        sh = {ro: float(np.mean([D[sidx[a], sidx[b]] <= qnb for a, b in zip(dd[dd.role == ro].kb, dd[dd.role == ro].slot)])) for ro in ("N1", "N2", "Zm", "Z1")}
        es = pd.read_csv(rd / "tables" / "estimands.csv")
        es = es[(es.metric == "ddp") & es.estimand.isin(["main", "matched", "off"])]
        txt = "; ".join(f"{r_.score} {r_.estimand} {r_.point:+.3f} [{r_.lo:+.3f}, {r_.hi:+.3f}]" for r_ in es.itertuples() if r_.score in ("S_tau@0.01", "S_tau@inf", "mix@0.5"))
        L.append(f"- {nm_}: share of draws with the slot in the KB's band d ≤ q10 (any category): " + ", ".join(f"{k_} {v:.3f}" for k_, v in sh.items()) + f". Δd′: {txt}")
    rt = (Path(args.tree_res) if args.tree_res else res.parent / "results_tree") / "tables" / "regression.csv"
    if rt.exists():
        g = pd.read_csv(rt)
        g = g[(g.distance == "tag") & g.score.isin(["S_tau@0.01", "S_tau@inf", "mix@0.5"])]
        L.append("- tree-distance dose response (Δd′ per 1 SD of tree distance; slot-site and (KB, O, draw) fixed effects): "
                 + "; ".join(f"{r_.score} {r_.coef} {r_.point:+.3f} [{r_.lo:+.3f}, {r_.hi:+.3f}]" for r_ in g.itertuples()))
    Du = pd.read_csv(proc / "copurchase_js_unmapped.csv", index_col=0).loc[sites, sites].to_numpy()
    rk = lambda M: np.argsort(np.argsort(M + np.eye(nS) * 9, axis=1, kind="stable"), axis=1, kind="stable")
    same_nn = float(np.mean(rk(D).argmin(1) == rk(Du).argmin(1)))
    L.append(f"- co-purchase JS vs the unmapped-pseudo-node variant: Spearman over pairs {spearmanr(D[iu], Du[iu])[0]:+.3f}; same nearest site for {same_nn:.2f} of sites")
    br = np.array([sinfo[s_]["profile_nodes"] for s_ in sites])
    mD = (D.sum(1)) / (nS - 1)
    L.append(f"- hub tendency A16: Spearman(profile breadth, mean co-purchase JS to the others) {spearmanr(br, mD)[0]:+.3f}")
    try:
        a0 = PROJECT / "data" / "processed" / "A0"
        it0 = pd.read_parquet(a0 / "items.parquet", columns=["site", "role", "tags"])
        tj = pd.read_csv(a0 / "tag_js.csv", index_col=0)
        b0 = it0[it0.role == "bank"].groupby("site").tags.apply(lambda x: len({t for v in x for t in __import__("json").loads(v)}))
        m0 = tj.sum(1) / (len(tj) - 1)
        L.append(f"- hub tendency SE reference: Spearman(distinct tags in the bank, mean tag-JS) {spearmanr(b0.loc[tj.index], m0)[0]:+.3f} ({len(tj)} sites)")
    except Exception as e:
        L.append(f"- hub tendency SE reference: not computed ({type(e).__name__})")
    (res / "A16_extra.md").write_text("\n".join(L) + "\n")
    meta = run_metadata(cfg, "A16.04_report")
    meta["inputs"] = fingerprint([res / "tables" / "contrasts.parquet", res / "tables" / "draws.csv", edir / "topk.npy", edir / "q_index.parquet",
                                  proc / "items.parquet", proc / cfg["distance_file"], proc / "tree_dist.csv", proc / "copurchase_js_unmapped.csv", proc / "build_report.json"])
    save_json(meta, res / "meta_04.json")
    log.info("done")


if __name__ == "__main__":
    main()
