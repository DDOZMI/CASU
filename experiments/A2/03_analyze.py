from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json, stable_seed

PARTS = ["site", "tag", "cross", "same", "random"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--res", default=None)
    ap.add_argument("--retriever", default=None, help="default: config detector (A5: per model)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    r = args.retriever or cfg["detector"]
    res = Path(args.res) if args.res else (p(cfg, "results", r) if cfg.get("results_per_retriever") else p(cfg, "results"))
    log = get_logger("A2.03", res / "logs" / "03_analyze.log")
    B, clip = cfg["metrics"]["bootstrap_B"], cfg["metrics"]["dprime_clip"]
    taus = [str(t) for t in cfg["scores"]["tau_grid"]]
    alphas = [str(a) for a in cfg["scores"]["alpha_grid"]]
    sites = load_json(p(cfg, "a1_emb", r, "sites.json"))
    sidx = {s: i for i, s in enumerate(sites)}
    D = pd.read_csv(p(cfg, "a0_processed", "tag_js.csv"), index_col=0).loc[sites, sites].to_numpy()
    q_nb = load_json(p(cfg, "a0_processed", "bands.json"))["q_neighbor_value"]
    cat = {s: v["category"] for s, v in load_json(p(cfg, "a0_processed", "sites.json")).items()}
    df = pd.read_parquet(res / "tables" / "rows.parquet")
    df["param"] = df["param"].astype(str).str.replace(r"\.0$", "", regex=True)
    dp = lambda a: np.sqrt(2) * norm.ppf(np.clip(a, clip, 1 - clip))
    df["ddp"] = dp(df.auroc.to_numpy()) - dp(df.auroc_abs.to_numpy())
    df["dOP"] = df.op_ood_acc - df.op_ood_acc_abs
    df["dA"] = df.auroc - df.auroc_abs
    edir = p(cfg, "a1_emb", r)
    T1 = np.load(edir / "topk.npy", mmap_mode="r")
    qi = pd.read_parquet(edir / "q_index.parquet")
    rows_of = {kk: g.index.to_numpy() for kk, g in qi.groupby(["site", "role"])}
    sets0 = pd.read_csv(res / "tables" / "sets.csv")
    chk = []
    for kb, o, order in sets0[sets0.draw == 0][["kb", "ood", "order"]].head(30).itertuples(index=False):
        i_id, i_ood = rows_of[(kb, "test")], rows_of[(o, "test")]
        sel = np.r_[i_id, i_ood]
        Lm = np.asarray(T1[sel]).astype(np.float64).mean(2)
        sA = Lm[:, [sidx[x] for x in order.split("+")[:4]]]
        S = Lm[:, sidx[kb]] - 0.01 * (logsumexp(sA / 0.01, axis=1) - np.log(4))
        rk = rankdata(-S)
        a_re = (rk[len(i_id):].sum() - len(i_ood) * (len(i_ood) + 1) / 2) / (len(i_ood) * len(i_id))
        a_a2 = df[(df.kb == kb) & (df.ood == o) & (df.draw == 0) & (df.m == 4) & (df.corpus == "se") & (df.partition == "site")
                  & (df.family == "S_tau") & (df.param == "0.01")].auroc.iloc[0]
        chk.append(abs(a_re - a_a2))
    sanity = float(np.max(chk))
    df["key"] = df.corpus + ":" + df.partition + ":" + df.family + "@" + df.param
    idx = ["kb", "ood", "draw", "m"]
    W = df.pivot_table(index=idx, columns="key", values="ddp", aggfunc="first")
    Wo = df.pivot_table(index=idx, columns="key", values="dOP", aggfunc="first")
    Wa = df.pivot_table(index=idx, columns="key", values="dA", aggfunc="first")
    Wi = df.pivot_table(index=idx, columns="key", values="op_id_acc", aggfunc="first")
    Wp = df[df.family == "S_tau"].pivot_table(index=idx, columns="partition", values="purity", aggfunc="first")
    cols = {}
    for t in taus:
        g = {pt: W[f"se:{pt}:S_tau@{t}"] for pt in PARTS}
        for pt in PARTS:
            cols[f"val:{pt}@{t}"] = g[pt]
            cols[f"op:{pt}@{t}"] = Wo[f"se:{pt}:S_tau@{t}"]
        for a_, b_ in (("site", "random"), ("tag", "random"), ("cross", "random"), ("same", "random"), ("same", "cross"), ("site", "tag")):
            cols[f"{a_}-{b_}@{t}"] = g[a_] - g[b_]
        for pt in PARTS:
            cols[f"{pt}-mix1@{t}"] = g[pt] - W["se:pooled:mix@1"]
        stack = np.stack([g[pt].to_numpy() for pt in PARTS])
        cols[f"spread@{t}"] = pd.Series(stack.max(0) - stack.min(0), index=W.index)
        cols[f"ms_cross-site@{t}"] = W[f"ms:cross:S_tau@{t}"] - g["site"]
        cols[f"msmix1-site@{t}"] = W["ms:pooled:mix@1"] - g["site"]
        cols[f"dA:site-random@{t}"] = Wa[f"se:site:S_tau@{t}"] - Wa[f"se:random:S_tau@{t}"]
        for pt in ("site", "random"):
            cols[f"idacc:{pt}@{t}"] = Wi[f"se:{pt}:S_tau@{t}"]
        cols[f"val:ms_random@{t}"] = W[f"ms:random:S_tau@{t}"]
        cols[f"val:ms_cross@{t}"] = W[f"ms:cross:S_tau@{t}"]
        cols[f"ms_cross-ms_random@{t}"] = W[f"ms:cross:S_tau@{t}"] - W[f"ms:random:S_tau@{t}"]
    for a in alphas:
        for fam in ("mix", "mixk"):
            cols[f"val:se_{fam}@{a}"] = W[f"se:pooled:{fam}@{a}"]
            cols[f"val:ms_{fam}@{a}"] = W[f"ms:pooled:{fam}@{a}"]
            cols[f"se-ms_{fam}@{a}"] = W[f"se:pooled:{fam}@{a}"] - W[f"ms:pooled:{fam}@{a}"]
            cols[f"op:se_{fam}@{a}"] = Wo[f"se:pooled:{fam}@{a}"]
            cols[f"op:ms_{fam}@{a}"] = Wo[f"ms:pooled:{fam}@{a}"]
    for pt in ("tag", "cross", "same", "random"):
        cols[f"purity:{pt}@-"] = Wp[pt]
    cols["clip@-"] = ((df[df.family == "S_tau"].assign(c=lambda x: (x.auroc >= 1 - clip) | (x.auroc_abs >= 1 - clip))
                       .groupby(idx).c.any()).reindex(W.index).astype(float))
    C = pd.DataFrame(cols, index=W.index)
    C = C.reset_index()
    sets = pd.read_csv(res / "tables" / "sets.csv").set_index(["kb", "ood", "draw"])
    nbO, nbK = [], []
    for kb, o, d, m in zip(C.kb, C.ood, C.draw, C.m):
        first = sets.loc[(kb, o, d), "order"].split("+")[:m]
        nbO.append(any(cat[s] == cat[o] and D[sidx[s], sidx[o]] <= q_nb for s in first))
        nbK.append(any(cat[s] == cat[kb] and D[sidx[s], sidx[kb]] <= q_nb for s in first))
    C["has_O_neighbour"], C["has_KB_neighbour"] = nbO, nbK
    C.to_parquet(res / "tables" / "contrasts.parquet", index=False)

    rng = np.random.default_rng(stable_seed(cfg["seed"], "A2_boot2"))
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

    clus = None
    if cfg.get("cluster_bootstrap") == "subset_triples":
        groups = load_json(p(cfg, "a0_processed", "subset.json"))["groups"]
        gi = {s_: i for i, g_ in enumerate(groups) for s_ in g_["sites"]}
        clus = np.array([gi[s_] for s_ in sites])
        nG = len(groups)
        rngc = np.random.default_rng(stable_seed(cfg["seed"], "A2_boot_cluster"))
        CKc = rngc.multinomial(nG, [1 / nG] * nG, size=B).astype(float)
        COc = rngc.multinomial(nG, [1 / nG] * nG, size=B).astype(float)

    def boot_cluster(sub, col):
        if clus is None:
            return np.nan, np.nan
        s_ = sub.dropna(subset=[col])
        if s_.empty:
            return np.nan, np.nan
        cell = s_.groupby(["kb", "ood"])[col].mean()
        ki = clus[[sidx[a] for a, _ in cell.index]]
        oi = clus[[sidx[b] for _, b in cell.index]]
        Wc = CKc[:, ki] * COc[:, oi]
        ok = Wc.sum(1) > 0
        if not ok.any():
            return np.nan, np.nan
        return tuple(np.quantile((Wc[ok] @ cell.to_numpy()) / Wc[ok].sum(1), [0.025, 0.975]))

    def f(t, d=3):
        return "—" if t[3] == 0 else f"{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]"

    est = []
    for m in sorted(C.m.unique()):
        g = C[C.m == m]
        for col in [c for c in C.columns if "@" in c]:
            t = boot(g, col)
            lc, hc = boot_cluster(g, col)
            est.append({"m": m, "stratum": "all", "estimand": col, "point": t[0], "lo": t[1], "hi": t[2], "n_cells": t[3],
                        "lo_cluster": lc, "hi_cluster": hc})
    E = pd.DataFrame(est)

    def gE(m, col, stratum="all"):
        q = E[(E.m == m) & (E.estimand == col) & (E.stratum == stratum)]
        return tuple(q.iloc[0][["point", "lo", "hi", "n_cells"]]) if len(q) else (np.nan,) * 3 + (0,)

    L = ["# A2 — RQ2 three layers (StackExchange, detector " + r + (", A5 30-site subset" if cfg.get("site_set") == "subset" or "A5" in str(res) else "") + ")", "",
         f"- pairs: A1 pairs ({C.groupby(['kb', 'ood']).ngroups}); draws R = {cfg['sets']['R']}; m ∈ {cfg['sets']['m_grid']} (nested). Δd′ vs the absolute score; [ ] two-way (KB x O) bootstrap 95%, B = {B}.",
         f"- sanity: site S_τ@0.01 (m = 4, draw 0, 30 pairs) recomputed from the A1 top-k: max |ΔAUROC| = {sanity:.2e}.",
         "- Purity (mean majority-site share of a group; site = 1, random ≈ 1/m) shows how far each clustering reproduces the sites; on this test bed embedding clusters nearly do, so partition contrasts among site/tag/cross/same are read together with purity, and the clustering effect proper is read from MS cross − MS random (no site structure).", ""]
    L += ["## Purity of the partitions (mean over draws)", "", "| m | tag | cross | same | random | share of draws at the d′ clip |", "|---|---|---|---|---|---|"]
    for m in sorted(C.m.unique()):
        g_ = C[C.m == m]
        L.append(f"| {m} | " + " | ".join(f"{g_[f'purity:{pt}@-'].mean():.3f}" for pt in ("tag", "cross", "same", "random")) + f" | {g_['clip@-'].mean():.3f} |")
    L.append("")
    m0 = 4
    g4 = C[C.m == m0]
    L += [f"## Layer 2 and 3 at m = {m0}: Δd′ of each partition (point) and partition contrasts", "",
          "| τ | site | tag | cross | same | random | site − random | tag − random | cross − random | same − random | same − cross | ΔAUROC site − random |", "|---" * 12 + "|"]
    for t in taus:
        L.append(f"| {t} | " + " | ".join(f"{gE(m0, f'val:{pt}@{t}')[0]:+.3f}" for pt in PARTS) + " | "
                 + " | ".join(f(gE(m0, f"{c}@{t}")) for c in ("site-random", "tag-random", "cross-random", "same-random", "same-cross", "dA:site-random")) + " |")
    L += ["", f"## Layer 2 vs layer 1 at m = {m0}: partition S_τ − pooled kNN (mix@1), and the spread across the five partitions", "",
          "| τ | site − mix@1 | tag − mix@1 | cross − mix@1 | same − mix@1 | random − mix@1 | spread (max − min) |", "|---" * 7 + "|"]
    for t in taus:
        L.append(f"| {t} | " + " | ".join(f(gE(m0, f"{pt}-mix1@{t}")) for pt in PARTS) + f" | {f(gE(m0, f'spread@{t}'))} |")
    L += ["", f"Realised ID acceptance at the calibrated threshold (target 0.95), m = {m0}: " + "; ".join(
        f"{pt}@{t} {gE(m0, f'idacc:{pt}@{t}')[0]:.3f}" for t in ("0.01", "inf") for pt in ("site", "random"))]
    L += ["", f"## Layer 1 (pooled background) at m = {m0}: same platform vs MS MARCO (same size)", "",
          "| α | SE mix | SE mixk | MS mix | MS mixk | SE − MS (mix) | SE − MS (mixk) |", "|---" * 7 + "|"]
    for a in alphas:
        L.append(f"| {a} | {gE(m0, f'val:se_mix@{a}')[0]:+.3f} | {gE(m0, f'val:se_mixk@{a}')[0]:+.3f} | {gE(m0, f'val:ms_mix@{a}')[0]:+.3f} | "
                 f"{gE(m0, f'val:ms_mixk@{a}')[0]:+.3f} | {f(gE(m0, f'se-ms_mix@{a}'))} | {f(gE(m0, f'se-ms_mixk@{a}'))} |")
    L += ["", f"## General corpus partitions at m = {m0} (MS MARCO)", "",
          "| τ | MS random | MS cross | MS cross − MS random | MS cross − SE site (mixes corpus and partition) |", "|---|---|---|---|---|"]
    for t in taus:
        L.append(f"| {t} | {gE(m0, f'val:ms_random@{t}')[0]:+.3f} | {gE(m0, f'val:ms_cross@{t}')[0]:+.3f} | {f(gE(m0, f'ms_cross-ms_random@{t}'))} | {f(gE(m0, f'ms_cross-site@{t}'))} |")
    L += ["", "## Dependence on m (Δd′ points; contrast with interval)", "",
          "| m | site@0.01 | random@0.01 | site@inf | random@inf | SE mix@1 | MS mix@1 | site − random @0.01 | site − random @inf |", "|---" * 9 + "|"]
    for m in sorted(C.m.unique()):
        L.append(f"| {m} | {gE(m, 'val:site@0.01')[0]:+.3f} | {gE(m, 'val:random@0.01')[0]:+.3f} | {gE(m, 'val:site@inf')[0]:+.3f} | "
                 f"{gE(m, 'val:random@inf')[0]:+.3f} | {gE(m, 'val:se_mix@1')[0]:+.3f} | {gE(m, 'val:ms_mix@1')[0]:+.3f} | "
                 f"{f(gE(m, 'site-random@0.01'))} | {f(gE(m, 'site-random@inf'))} |")
    L += ["", f"## Strata at m = {m0}: O-neighbour x KB-neighbour in the comparison set (crossed; RQ3 link and KB-proximity cost)", "",
          "| estimand | O-nb yes, KB-nb no | O-nb no, KB-nb no | O-nb yes, KB-nb yes | O-nb no, KB-nb yes |", "|---|---|---|---|---|"]
    for col in ("val:site@0.01", "val:random@0.01", "val:site@inf", "val:random@inf", "val:se_mix@1", "site-random@0.01", "site-random@inf",
                "tag-random@0.01", "tag-random@inf"):
        cells = []
        for lab, msk in (("Onb1_KBnb0", g4.has_O_neighbour & ~g4.has_KB_neighbour), ("Onb0_KBnb0", ~g4.has_O_neighbour & ~g4.has_KB_neighbour),
                         ("Onb1_KBnb1", g4.has_O_neighbour & g4.has_KB_neighbour), ("Onb0_KBnb1", ~g4.has_O_neighbour & g4.has_KB_neighbour)):
            t = boot(g4[msk], col)
            lc, hc = boot_cluster(g4[msk], col)
            est.append({"m": m0, "stratum": lab, "estimand": col, "point": t[0], "lo": t[1], "hi": t[2], "n_cells": t[3],
                        "lo_cluster": lc, "hi_cluster": hc})
            cells.append(f"{f(t)} ({t[3]})")
        L.append(f"| {col} | " + " | ".join(cells) + " |")
    L += ["", f"- m = {m0} draws: O-nb & no KB-nb {(g4.has_O_neighbour & ~g4.has_KB_neighbour).mean():.2f}, neither {(~g4.has_O_neighbour & ~g4.has_KB_neighbour).mean():.2f}, "
          f"both {(g4.has_O_neighbour & g4.has_KB_neighbour).mean():.2f}, KB-nb only {(~g4.has_O_neighbour & g4.has_KB_neighbour).mean():.2f}"]
    L += ["", f"## Operating point at m = {m0}: Δ OOD acceptance at 95% calibrated ID acceptance (vs absolute)", "",
          "| score | Δ OOD acceptance |", "|---|---|"]
    for col in ("op:site@0.01", "op:random@0.01", "op:site@inf", "op:random@inf", "op:se_mix@1", "op:ms_mix@1", "op:se_mix@0.5", "op:ms_mix@0.5"):
        L.append(f"| {col[3:]} | {f(gE(m0, col))} |")
    pd.DataFrame(est).to_csv(res / "tables" / "estimands.csv", index=False)
    (res / "A2_summary.md").write_text("\n".join(L) + "\n")
    meta = run_metadata(cfg, "A2.03_analyze")
    meta["inputs"] = fingerprint([res / "tables" / "rows.parquet", res / "tables" / "sets.csv", p(cfg, "a0_processed", "tag_js.csv"), edir / "topk.npy",
                                  p(cfg, "a0_processed", "bands.json"), p(cfg, "a0_processed", "sites.json")])
    meta["outputs"] = fingerprint([res / "A2_summary.md", res / "tables" / "estimands.csv"])
    meta["sanity_site_recompute_max_abs_dauroc"] = float(sanity)
    save_json(meta, res / "meta_03.json")
    log.info("summary written")


if __name__ == "__main__":
    main()
