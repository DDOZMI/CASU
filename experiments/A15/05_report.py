from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import PROJECT_ROOT, fingerprint, get_logger, load_config, p, run_metadata, save_json, stable_seed

TIERS = ["near", "mid", "far"]
FAMS = ["se", "yahoo", "nq", "msq"]
BANKS = ["se_n", "wiki_n", "yahoo_n", "ms_n", "se_full", "wiki_full", "yahoo_full", "ms_full"]


def boot(d, col, B, seed, kb_only=False):
    d = d[["kb", "ood", col]].dropna()
    if len(d) == 0:
        return (np.nan, np.nan, np.nan, 0)
    kbs, oos = sorted(d.kb.unique()), sorted(d.ood.unique())
    ki = d.kb.map({s: i for i, s in enumerate(kbs)}).to_numpy()
    oi = d.ood.map({s: i for i, s in enumerate(oos)}).to_numpy()
    v = d[col].to_numpy(float)
    rng = np.random.default_rng(seed)
    WK = rng.multinomial(len(kbs), [1 / len(kbs)] * len(kbs), size=B).astype(float)
    WO = rng.multinomial(len(oos), [1 / len(oos)] * len(oos), size=B).astype(float)
    W = WK[:, ki] * (1.0 if kb_only else WO[:, oi])
    bs = (W @ v) / W.sum(1)
    lo, hi = np.nanquantile(bs, [0.025, 0.975])
    return (float(v.mean()), float(lo), float(hi), int(len(v)))


def f3(x, nd=3):
    return f"{x[0]:+.{nd}f} [{x[1]:+.{nd}f}, {x[2]:+.{nd}f}]"


def interaction(C, sc, B, seed, col="dG_ref", balanced=False, other="wiki_n"):
    x = C[(C.score == sc) & C.bank.isin(["se_n", other]) & C.family.isin(["se", "yahoo"])]
    w = x.pivot_table(index=["kb", "ood", "family"], columns="bank", values=col).dropna().reset_index()
    w["c"] = w.se_n - w[other]
    kbs = sorted(w.kb.unique())
    ki = w.kb.map({s: i for i, s in enumerate(kbs)}).to_numpy()
    rng = np.random.default_rng(seed)
    WK = rng.multinomial(len(kbs), [1 / len(kbs)] * len(kbs), size=B).astype(float)
    parts = {}
    for fam in ("se", "yahoo"):
        m = (w.family == fam).to_numpy()
        oos = sorted(w.ood[m].unique())
        oi = w.ood[m].map({s: i for i, s in enumerate(oos)}).to_numpy()
        WO = rng.multinomial(len(oos), [1 / len(oos)] * len(oos), size=B).astype(float)
        v = w.c[m].to_numpy()
        if balanced:
            Wk = WK[:, ki[m]]
            per_o = np.stack([(Wk[:, oi == j] @ v[oi == j]) / Wk[:, oi == j].sum(1) for j in range(len(oos))], 1)
            bs = np.nansum(per_o * WO, 1) / np.where(np.isnan(per_o), 0, WO).sum(1)
            pt = float(np.mean([v[oi == j].mean() for j in range(len(oos))]))
        else:
            Wf = WK[:, ki[m]] * WO[:, oi]
            bs = (Wf @ v) / Wf.sum(1)
            pt = float(v.mean())
        parts[fam] = (bs, pt, int(m.sum()))
    bs = parts["se"][0] - parts["yahoo"][0]
    lo, hi = np.nanquantile(bs, [0.025, 0.975])
    return (parts["se"][1] - parts["yahoo"][1], float(lo), float(hi)), parts["se"][1:], parts["yahoo"][1:]


def cov_adjusted(C, sc, B, seed, col="dG_ref", other="wiki_n"):
    x = C[(C.score == sc) & C.bank.isin(["se_n", other]) & C.family.isin(["se", "yahoo"])]
    w = x.pivot_table(index=["kb", "ood", "family"], columns="bank", values=[col, "coverage"]).dropna().reset_index()
    y = (w[(col, "se_n")] - w[(col, other)]).to_numpy()
    X = np.column_stack([np.ones(len(w)), (w[("coverage", "se_n")] - w[("coverage", other)]).to_numpy(), (w.family == "se").to_numpy(float)])
    kbs = sorted(w.kb.unique())
    ki = w.kb.map({s: i for i, s in enumerate(kbs)}).to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        wt = rng.multinomial(len(kbs), [1 / len(kbs)] * len(kbs))[ki].astype(float)
        sw = np.sqrt(wt)
        bs.append(np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0])
    bs = np.array(bs)
    q = np.nanquantile(bs, [0.025, 0.975], axis=0)
    return (beta[2], q[0, 2], q[1, 2]), (beta[1], q[0, 1], q[1, 1]), len(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--res", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = Path(args.res) if args.res else p(cfg, "results")
    log = get_logger("A15.05", res / "logs" / "05_report.log")
    ec = cfg["eval"]
    B, ceil = ec["bootstrap_B"], ec["ceiling"]
    runs = [r for r in cfg["runs"] if not args.runs or r in args.runs]
    L = ["# A15 — cross-source test: bank source × out-of-scope query source", "",
         "- Plan, forecasts and reading rule: JOURNAL 2026-10-07 21:22 (fixed before the run). Settings: config eval._source.",
         "- Banks: se_n / wiki_n / ms_n = the same number n of 3,000-document groups per KB (matched across sources; wiki and MS banks shared by all O of a KB); *_full = all eligible groups (ms_full = MS chunks matched to the SE full bank's count, capped by the pool). SE and wiki groups are filtered by the KB-adjacency rule (common 0.9-quantile crowd threshold per source); MS chunks are random and unfiltered.",
         "- dG_ref = G(score) − G(flats_ms@0.5, FLatS eq. 5 on the 12k P9 MS MARCO sample), per cell, G on cells with A_abs ≤ 0.98. Intervals: KB × O two-way and KB bootstrap 95%.",
         "- Reading rule: if the gain survives with out-of-scope queries from another platform and tracks topic distance → topic-neighbour coverage; if it is large only for SE bank → SE queries → source family. Primary: the 2×2 interaction I (se_n − wiki_n contrast on SE O minus on Yahoo O).",
         "- Design notes (recorded before the run, JOURNAL; our choices): the 2×2 is at a matched group count n = min(|SE eligible| − 1, |wiki eligible|, MS chunks) per KB, smaller than the A14 bank (wiki is binding in most main KBs); se_full / wiki_full / ms_full give the full-size lines (not size-matched across sources). Primary score for I: S_tau@0.01 (A14's primary); the other scores are secondary. The DBpedia wiki bank covers SE how-to topics poorly, so topic coverage alone predicts I > 0: I does not separate (a) from (b) by its sign; read it with the coverage-adjusted family effect (post-hoc), the ΔAUROC and O-balanced versions, and the Yahoo per-topic table. Tiers are not comparable across families; NQ / MS MARCO q have few G cells (read ΔAUROC). Yahoo topics are labelled OOD for every KB, including plausibly in-scope pairs (e.g. Computers & Internet vs superuser).",
         "- Limitations (recorded before the run): the wiki bank is DBpedia abstracts (entity descriptions; 31 l2 classes, skewed to athletes and persons); non-SE out-of-scope queries differ in format from SE ID queries, so A_abs is near the ceiling and G cells are few (counts shown); length AUROC is shown per family.", ""]
    for rn in runs:
        R = pd.read_parquet(res / "tables" / f"rows_{rn}.parquet")
        M = pd.read_parquet(res / "tables" / f"cells_meta_{rn}.parquet").set_index(["kb", "ood"])
        C = R.groupby(["kb", "ood", "family", "tier", "bank", "score"])[["auroc", "op_ood_acc", "op_id_acc", "fpr95", "T_push", "T_var", "pi", "rho_id", "nn_sim", "coverage", "auroc_abs"]].mean().reset_index()
        mm = M.reindex(pd.MultiIndex.from_frame(C[["kb", "ood"]]))
        C["auroc_ref"] = mm.auroc_ref.to_numpy()
        ok = C.auroc_abs <= ceil
        C["G"] = np.where(ok, (C.auroc - C.auroc_abs) / (1 - C.auroc_abs), np.nan)
        C["G_ref"] = np.where(ok, (C.auroc_ref - C.auroc_abs) / (1 - C.auroc_abs), np.nan)
        C["dG_ref"] = C.G - C.G_ref
        C["dA_ref"] = C.auroc - C.auroc_ref
        C.to_parquet(res / "tables" / f"cells_{rn}.parquet", index=False)
        L += [f"## {rn}", "", "| family | cells | G cells | mean A_abs | mean A_ref | length AUROC (shorter = OOD) | n matched (KB 5/50/95%) |", "|---|---|---|---|---|---|---|"]
        Mr = M.reset_index()
        for fam in FAMS:
            x = Mr[Mr.family == fam]
            if len(x) == 0:
                continue
            L.append(f"| {fam} | {len(x)} | {int((x.auroc_abs <= ceil).sum())} | {x.auroc_abs.mean():.4f} | {x.auroc_ref.mean():.4f} | {x.len_auroc.mean():.3f} | "
                     + " / ".join(f"{v:.0f}" for v in x.groupby("kb").n_matched.first().quantile([.05, .5, .95])) + " |")
        L += ["", "### dG and dAUROC vs FLatS-MS (12k) by family × bank × score", "",
              "| family | bank | score | dG two-way 95% (G cells) | dG KB 95% | dG near / mid / far | dAUROC two-way 95% | T_push / T_var | ρ_ID |", "|---|---|---|---|---|---|---|---|---|"]
        seed = lambda *q: stable_seed(cfg["seed"], "A15_boot", rn, *q)
        for fam in FAMS:
            for bank in BANKS:
                for sc in sorted(C.score.unique()):
                    d = C[(C.family == fam) & (C.bank == bank) & (C.score == sc)]
                    if len(d) == 0:
                        continue
                    g2, gk, a2 = boot(d, "dG_ref", B, seed(fam, "dG")), boot(d, "dG_ref", B, seed(fam, "dG", "kb"), kb_only=True), boot(d, "dA_ref", B, seed(fam, "dA"))
                    mo = f"{d.T_push.mean():+.3f} / {d.T_var.mean():+.3f}" if d.T_push.notna().any() else "—"
                    rh = f"{d.rho_id.mean():+.3f}" if d.rho_id.notna().any() else "—"
                    L.append(f"| {fam} | {bank} | {sc} | {f3(g2)} ({g2[3]}) | {f3(gk)} | " + " / ".join(f"{d[d.tier == t].dG_ref.mean():+.3f}" for t in TIERS)
                             + f" | {f3(a2, 4)} | {mo} | {rh} |")
        L += ["", "### 2×2 interaction (primary): I = [dG(se_n) − dG(wiki_n)] on SE O − the same on Yahoo O (KBs, SE O sites and Yahoo topics resampled)", "",
              "| score | I [95%] | se_n − wiki_n on SE O (cells) | on Yahoo O (cells) |", "|---|---|---|---|"]
        for sc in sorted(C.score.unique()):
            I, se_, ya = interaction(C, sc, B, seed("I", sc))
            L.append(f"| {sc} | {f3(I)} | {se_[0]:+.3f} ({se_[1]}) | {ya[0]:+.3f} ({ya[1]}) |")
        L += ["", "Robustness of I (post-hoc rows, review 2026-10-07): ΔAUROC version (no ceiling, all cells); each SE O site / Yahoo topic weighted equally; coverage-adjusted family effect", "",
              "| score | I in ΔAUROC [95%] | I (G, O-balanced) [95%] | I (ΔAUROC, O-balanced) [95%] | coverage-adjusted family effect, G [95%] (coverage slope) | same, ΔAUROC | cells |",
              "|---|---|---|---|---|---|---|"]
        for sc in sorted(C.score.unique()):
            Ia = interaction(C, sc, B, seed("Ia", sc), col="dA_ref")[0]
            Ib = interaction(C, sc, B, seed("Ib", sc), balanced=True)[0]
            Iab = interaction(C, sc, B, seed("Iab", sc), col="dA_ref", balanced=True)[0]
            fg, sg, ng = cov_adjusted(C, sc, min(B, 1000), seed("cov", sc))
            fa, sa, _ = cov_adjusted(C, sc, min(B, 1000), seed("cova", sc), col="dA_ref")
            L.append(f"| {sc} | {f3(Ia, 4)} | {f3(Ib)} | {f3(Iab, 4)} | {f3(fg)} ({sg[0]:+.2f}) | {f3(fa, 4)} ({sa[0]:+.3f}) | {ng} |")
        L += ["", "### Symmetric 2×2 (user request 'Yahoo도 해보도록 해'): I_Y = [dG(se_n) − dG(yahoo_n)] on SE O − the same on Yahoo O (each family's own source in the bank, the out-of-scope O's own site / topic removed)", "",
              "| score | I_Y (G) [95%] | se_n − yahoo_n on SE O (cells) | on Yahoo O (cells) | I_Y in ΔAUROC [95%] | I_Y (G, O-balanced) [95%] | coverage-adjusted family effect, G [95%] (slope) |",
              "|---|---|---|---|---|---|---|"]
        for sc in sorted(C.score.unique()):
            I, se_, ya = interaction(C, sc, B, seed("IY", sc), other="yahoo_n")
            Ia = interaction(C, sc, B, seed("IYa", sc), col="dA_ref", other="yahoo_n")[0]
            Ib = interaction(C, sc, B, seed("IYb", sc), balanced=True, other="yahoo_n")[0]
            fg, sg, _ = cov_adjusted(C, sc, min(B, 1000), seed("IYcov", sc), other="yahoo_n")
            L.append(f"| {sc} | {f3(I)} | {se_[0]:+.3f} ({se_[1]}) | {ya[0]:+.3f} ({ya[1]}) | {f3(Ia, 4)} | {f3(Ib)} | {f3(fg)} ({sg[0]:+.2f}) |")
        L += ["", "Four cells of the 2×2 (dG vs FLatS-MS, matched size): bank (rows) × out-of-scope source (columns)", "", "| score | bank | SE O | Yahoo O | NQ | MS MARCO q |", "|---|---|---|---|---|---|"]
        for sc in sorted(C.score.unique()):
            for bank in ("se_n", "wiki_n", "yahoo_n", "ms_n"):
                L.append(f"| {sc} | {bank} | " + " | ".join(f"{C[(C.family == f) & (C.bank == bank) & (C.score == sc)].dG_ref.mean():+.3f}" for f in FAMS) + " |")
        Rd = R[R.bank.isin(["se_n", "wiki_n", "yahoo_n"])].copy()
        mr = M.reindex(pd.MultiIndex.from_frame(Rd[["kb", "ood"]]))
        okr = Rd.auroc_abs <= ceil
        Rd["dG_ref"] = np.where(okr, (Rd.auroc - mr.auroc_ref.to_numpy()) / (1 - Rd.auroc_abs), np.nan)
        L += ["", "### Moderator: dG vs FLatS-MS by nn_sim quartile (second-encoder cosine of the O query centroid to the bank's nearest group; quartiles within family × bank; per draw; S_tau@0.01 and mix@0.5)", "",
              "| family | bank | score | Q1 (far) | Q2 | Q3 | Q4 (near) | Spearman(nn_sim, dG) |", "|---|---|---|---|---|---|---|---|"]
        for fam in ("se", "yahoo"):
            for bank in ("se_n", "wiki_n", "yahoo_n"):
                for sc in ("S_tau@0.01", "mix@0.5"):
                    d = Rd[(Rd.family == fam) & (Rd.bank == bank) & (Rd.score == sc)].dropna(subset=["dG_ref"])
                    if len(d) < 8:
                        continue
                    qs = pd.qcut(d.nn_sim.rank(method="first"), 4, labels=False)
                    L.append(f"| {fam} | {bank} | {sc} | " + " | ".join(f"{d.dG_ref[qs == j].mean():+.3f}" for j in range(4))
                             + f" | {d[['nn_sim', 'dG_ref']].corr(method='spearman').iloc[0, 1]:+.2f} |")
        L += ["", "### Coverage (mean over O queries of the max second-encoder cosine to the bank's group centroids; cell means; not comparable across families: same-platform offset)", "",
              "| family | se_n | wiki_n | yahoo_n | se_full | wiki_full | yahoo_full |", "|---|---|---|---|---|---|---|"]
        for fam in FAMS:
            L.append(f"| {fam} | " + " | ".join(f"{C[(C.family == fam) & (C.bank == b) & (C.score == 'mix@0.5')].coverage.mean():.3f}" for b in ("se_n", "wiki_n", "yahoo_n", "se_full", "wiki_full", "yahoo_full")) + " |")
        Rt = R[(R.family == "se") & (R.bank == "se_n")].copy()
        mt_ = M.reindex(pd.MultiIndex.from_frame(Rt[["kb", "ood"]]))
        Rt["dG_ref"] = np.where(Rt.auroc_abs <= ceil, (Rt.auroc - mt_.auroc_ref.to_numpy()) / (1 - Rt.auroc_abs), np.nan)
        L += ["", "### SE O, se_n: dG vs FLatS-MS by tag rank of the bank's nearest site to O (A14 moderator; per draw)", "", "| score | nn 1 | nn 2-3 | nn 4-10 | nn 11+ |", "|---|---|---|---|---|"]
        for sc in ("S_tau@0.01", "mix@0.5"):
            d = Rt[Rt.score == sc]
            L.append(f"| {sc} | " + " | ".join(f"{d[(d.nn_rank_tag >= lo) & (d.nn_rank_tag <= hi)].dG_ref.mean():+.3f} (n={int(((d.nn_rank_tag >= lo) & (d.nn_rank_tag <= hi)).sum())})" for lo, hi in ((1, 1), (2, 3), (4, 10), (11, 999))) + " |")
        L += ["", "### Yahoo per topic (dG vs FLatS-MS; se_n, wiki_n, yahoo_n; S_tau@0.01 | mix@0.5; G cells)", "", "| topic | G cells | se_n | wiki_n | yahoo_n | nn_sim se_n / wiki_n / yahoo_n |", "|---|---|---|---|---|---|"]
        for o, d in C[C.family == "yahoo"].groupby("ood"):
            g = lambda b, s_: d[(d.bank == b) & (d.score == s_)].dG_ref.mean()
            L.append(f"| {o} | {int(d[(d.bank == 'se_n') & (d.score == 'mix@0.5')].G.notna().sum())} | {g('se_n', 'S_tau@0.01'):+.3f} \\| {g('se_n', 'mix@0.5'):+.3f} | "
                     f"{g('wiki_n', 'S_tau@0.01'):+.3f} \\| {g('wiki_n', 'mix@0.5'):+.3f} | {g('yahoo_n', 'S_tau@0.01'):+.3f} \\| {g('yahoo_n', 'mix@0.5'):+.3f} | "
                     f"{d[d.bank == 'se_n'].nn_sim.mean():.3f} / {d[d.bank == 'wiki_n'].nn_sim.mean():.3f} / {d[d.bank == 'yahoo_n'].nn_sim.mean():.3f} |")
        L.append("")
        log.info(f"{rn} done")
    (res / "A15_summary.md").write_text("\n".join(L) + "\n")
    meta = run_metadata(cfg, "A15.05_report")
    meta["inputs"] = fingerprint([res / "tables" / f"{x}_{rn}.parquet" for rn in runs for x in ("rows", "cells_meta")])
    save_json(meta, res / "meta_05.json")


if __name__ == "__main__":
    main()
