from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import PROJECT_ROOT, get_logger, load_config, p, run_metadata, save_json, stable_seed, fingerprint

TIERS = ["near", "mid", "far"]
GROUPS = [
    ("absolute (KB only): Li et al. 2024 family; KB Mahalanobis; ID-only KB-PCA variants (ours, after Triantafyllopoulos et al. 2026) and their diagnostics",
     ["abs", "li_mss", "li_knn10", "li_knn32", "li_avg32", "li_energy32", "li_entropy32", "maha_kb", "kbpca_q", "kbpca_d", "knn_cal_full", "kbpca_res",
      "kbpca_rec15", "kbpca_rec90"]),
    ("query-only normalisation (no comparison documents)", ["qz_kb", "qz_ms"]),
    ("qz_kb components (diagnostic, 03_extra): kb_mean = mean cosine to all 3,000 KB documents, kb_sd = their sd, kb_peak = s_KB − kb_mean, kb_scaled = s_KB / kb_sd (scored as ID-ness: AUROC < 0.5 means the component points the other way, not 'no signal')", ["kb_mean", "kb_sd", "kb_peak", "kb_scaled"]),
    ("general background: FLatS (eq. 5 distance form), FLatS-type mix / mixk, RMD-type (external background); pre-specified a = 0.5, other a = sweep", None),
    ("deployable comparison sets (label-free; exclude only KB and O): A2 random 4-site set and the supervised competitor", ["a2_site@0.01", "a2_site@inf", "clf_se", "clf_ms"]),
    ("O-informed analysis levels (need O to build; NOT deployable; not to be ranked against the baselines)", ["a1_iii_Z@0.01", "a1_iii_Z@inf", "a1_ii_N@0.01", "a1_ii_N@inf"]),
    ("upper bounds (use O): level (i) O's own bank; supervised on out-of-scope labels (max over the C grid of the out-of-fold AUROC, optimistic)",
     ["a1_i_O@0.01", "a1_i_O@inf", "tri_sup", "oracle"]),
]
REF = "flats_ms@0.5"


def twoway(d, col, B, seed, kb_only=False, q=(0.025, 0.975)):
    d = d[["kb", "ood", col]].dropna()
    if len(d) == 0:
        return np.nan, np.nan, np.nan, 0
    kbs, oos = sorted(d.kb.unique()), sorted(set(d.ood.unique()) | set(d.kb.unique()))
    ki = d.kb.map({s: i for i, s in enumerate(kbs)}).to_numpy()
    oi = d.ood.map({s: i for i, s in enumerate(oos)}).to_numpy()
    v = d[col].to_numpy(float)
    rng = np.random.default_rng(seed)
    WK = rng.multinomial(len(kbs), [1 / len(kbs)] * len(kbs), size=B).astype(float)
    WO = rng.multinomial(len(oos), [1 / len(oos)] * len(oos), size=B).astype(float)
    W = WK[:, ki] * (1.0 if kb_only else WO[:, oi])
    bs = (W @ v) / W.sum(1)
    lo, hi = np.nanquantile(bs, list(q))
    return float(v.mean()), float(lo), float(hi), int(len(v))


def fmt(x, nd=4):
    return f"{x[0]:+.{nd}f} [{x[1]:+.{nd}f}, {x[2]:+.{nd}f}]"


def ours_rows(run):
    root = PROJECT_ROOT
    A1 = pd.read_parquet(root / run["a1"] / "tables" / "rows.parquet",
                         columns=["kb", "ood", "draw", "role", "family", "param", "auroc", "op_id_acc", "op_ood_acc", "fpr95"])
    A1 = A1[(A1.family == "S_tau") & A1.param.isin(["0.01", "inf"])]
    lab = {"O": "a1_i_O", "N1": "a1_ii_N", "N2": "a1_ii_N", "Z1": "a1_iii_Z", "Z2": "a1_iii_Z"}
    A1 = A1[A1.role.isin(lab)].assign(score=lambda x: x.role.map(lab) + "@" + x.param)
    o1 = A1.groupby(["kb", "ood", "score"])[["auroc", "op_id_acc", "op_ood_acc", "fpr95"]].mean().reset_index()
    A2 = pd.read_parquet(root / run["a2"] / "tables" / "rows.parquet",
                         columns=["kb", "ood", "m", "corpus", "partition", "family", "param", "auroc", "op_id_acc", "op_ood_acc"])
    A2 = A2[(A2.m == 4) & (A2.corpus == "se") & (A2.partition == "site") & (A2.family == "S_tau") & A2.param.isin(["0.01", "inf"])]
    o2 = A2.assign(score="a2_site@" + A2.param).groupby(["kb", "ood", "score"])[["auroc", "op_id_acc", "op_ood_acc"]].mean().reset_index()
    return pd.concat([o1, o2], ignore_index=True), set(zip(A1.kb, A1.ood))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--res", default=None, help="results directory (smoke)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = Path(args.res) if args.res else p(cfg, "results")
    log = get_logger("A13.02", res / "logs" / "02_report.log")
    mt = cfg["metrics"]
    B, clip, ceil = mt["bootstrap_B"], mt["dprime_clip"], mt["ceiling"]
    dp = lambda a: np.sqrt(2) * norm.ppf(np.clip(a, clip, 1 - clip))
    runs = [r for r in cfg["runs"] if not args.runs or r in args.runs]
    L = ["# A13 — existing KB-gate methods and our comparison sets in one evaluation frame", "",
         "- Plan and forecasts: JOURNAL 2026-10-07 14:52 (fixed before the run). Definitions: config li / kbpca / gauss / qnorm / background / clf / oracle / ours.",
         "- Frame (as A1 / A2): ID = KB test (1,000), OOD = O test (1,000), A1 pairs; operating point = threshold accepting 95% of the KB calibration queries. Common cell set = A1 cells. Draw-based scores are means over 5 draws per cell.",
         "- Columns: AUROC (mean); ΔAUROC, G and Δd′ vs abs (s_KB, top-10 mean cosine) with the KB × O two-way bootstrap 95% and (for ΔAUROC and G) the KB bootstrap 95% (KBs resampled, O fixed; requested 2026-10-07 20:05); G by distance tier (near / mid / far); G = (A − A_abs)/(1 − A_abs) on cells with A_abs ≤ 0.98 (cells kept); ΔOOD-acc = OOD acceptance(abs) − OOD acceptance(score) at the calibrated threshold (> 0 = fewer OOD accepted; with the realised ID acceptance); FPR95. Upper bounds have no calibrated operating point.",
         "- Labels: flats_<bg>@a = −d₁₀(x; KB) + a·d₁₀(x; bg) (FLatS eq. 5, Euclidean 10th-NN distance of unit vectors; the KB documents stand in for FLatS's ID training samples); mix_<bg>@a = s_KB − a·(top-10 mean cosine to bg), mixk = the same with the 10th cosine (FLatS-type approximations); rmd_<bg> = −(Maha_KB − Maha_bg) (RMD-type with an external background, not Ren et al.'s ID-fitted MD_0); bg = ms (MS MARCO; mix / mixk / flats use the A2 cell sample, rmd and clf_ms a run-level sample per draw) or wiki (MedRAG Wikipedia, 3 shards, chunked: less diverse than MS MARCO). kbpca_q / kbpca_d are our ID-only constructions (Triantafyllopoulos's detectors need OOD-labelled queries; their supervised LogReg = tri_sup); knn_cal_full / kbpca_res are diagnostics. a2_site = A2 random 4-site set (label-free). a1_ii_N = level (ii) (O's nearest sites in the set), a1_iii_Z = level (iii) (sites far from O and KB): both are built with knowledge of O (analysis levels, not methods). a1_i_O = level (i) (O's own bank, leak bound). Our rows are existing results, not re-tuned.",
         f"- Pre-specified comparison: FLatS-type rows at a = 0.5 (FLatS default); other a are a sweep (a max over a is post-hoc). The last column is the paired AUROC difference to {REF} (common bootstrap weights across all scores, so row differences are paired).",
         "- Operating point: the realised ID acceptance differs by score; read ΔOOD-acc with the ID-acc next to it (FPR95 fixes the ID acceptance on test).", ""]
    sign_rows = []
    for rn in runs:
        run = cfg["runs"][rn]
        C = pd.read_parquet(res / "tables" / f"cells_{rn}.parquet")
        xp = res / "tables" / f"cells_extra_{rn}.parquet"
        if xp.exists():
            C = pd.concat([C, pd.read_parquet(xp).drop(columns=["m90", "evr15"])], ignore_index=True)
        ours, a1cells = ours_rows(run)
        absm = C[C.score == "abs"].set_index(["kb", "ood"])[["auroc", "op_ood_acc", "op_id_acc", "tier"]]
        ours = ours.merge(absm.rename(columns={"auroc": "auroc_abs", "op_ood_acc": "x1", "op_id_acc": "x2"}).reset_index()[["kb", "ood", "auroc_abs", "tier"]], on=["kb", "ood"])
        C = pd.concat([C, ours], ignore_index=True)
        C = C[[ (a, b) in a1cells for a, b in zip(C.kb, C.ood)]].copy()
        ab = absm.reindex(pd.MultiIndex.from_frame(C[["kb", "ood"]]))
        C["dA"] = C.auroc - C.auroc_abs
        C["ddp"] = dp(C.auroc.to_numpy()) - dp(C.auroc_abs.to_numpy())
        C["G"] = np.where(C.auroc_abs <= ceil, (C.auroc - C.auroc_abs) / (1 - C.auroc_abs), np.nan)
        C["dOOD"] = ab.op_ood_acc.to_numpy() - C.op_ood_acc.to_numpy()
        scores = list(dict.fromkeys(C.score))
        bg = sorted(s_ for s_ in scores if s_.startswith(("flats_", "mix_", "mixk_", "rmd_")))
        refc = C[C.score == REF].set_index(["kb", "ood"])
        mi = pd.MultiIndex.from_frame(C[["kb", "ood"]])
        C["dA_ref"] = C.auroc.to_numpy() - refc.auroc.reindex(mi).to_numpy()
        C["G_ref"] = C.G.to_numpy() - refc.G.reindex(mi).to_numpy()
        ncell = C[C.score == "abs"].shape[0]
        L += [f"## {rn} ({ncell} cells; G cells (A_abs ≤ {ceil}): " + " / ".join(f"{t_} {int(((C.score == 'abs') & (C.tier == t_) & C.G.notna()).sum())}" for t_ in TIERS) + ")", ""]
        rows = []
        seed = lambda q, t=None: stable_seed(cfg["seed"], "A13_boot", rn, q, t)
        for gname, members in GROUPS:
            members = bg if members is None else [x for x in members if x in scores]
            L += [f"### {gname}", "", "| score | AUROC | ΔAUROC two-way 95% | ΔAUROC KB 95% | G two-way 95% (cells) | G KB 95% | G near / mid / far | Δd′ two-way 95% | ΔOOD-acc / ID-acc | FPR95 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for sc in members:
                d = C[C.score == sc]
                mA = d.auroc.mean()
                dA, dAk = twoway(d, "dA", B, seed("dA")), twoway(d, "dA", B, seed("dA", "kb"), kb_only=True)
                gg, ggk = twoway(d, "G", B, seed("G")), twoway(d, "G", B, seed("G", "kb"), kb_only=True)
                dd = twoway(d, "ddp", B, seed("ddp"))
                do = twoway(d, "dOOD", B, seed("dOOD"))
                gt = [d[d.tier == t_].G.mean() for t_ in TIERS]
                at = [d[d.tier == t_].dA.mean() for t_ in TIERS]
                op = f"{do[0]:+.3f} [{do[1]:+.3f}, {do[2]:+.3f}] / {d.op_id_acc.mean():.3f}" if d.op_ood_acc.notna().any() else "—"
                fp = f"{d.fpr95.mean():.3f}" if "fpr95" in d and d.fpr95.notna().any() else "—"
                L.append(f"| {sc} | {mA:.4f} | {fmt(dA)} | {fmt(dAk)} | {fmt(gg, 3)} ({gg[3]}) | {fmt(ggk, 3)} | " + " / ".join(f"{v:+.3f}" for v in gt)
                         + f" | {fmt(dd, 3)} | {op} | {fp} |")
                rows.append({"run": rn, "group": gname, "score": sc, "auroc": mA, "dA": dA[0], "dA_lo": dA[1], "dA_hi": dA[2], "dA_kb_lo": dAk[1], "dA_kb_hi": dAk[2],
                             "G": gg[0], "G_lo": gg[1], "G_hi": gg[2], "G_kb_lo": ggk[1], "G_kb_hi": ggk[2], "G_cells": gg[3],
                             "ddp": dd[0], "ddp_lo": dd[1], "ddp_hi": dd[2], "dOOD": do[0],
                             **{f"G_{t_}": v for t_, v in zip(TIERS, gt)}, **{f"dA_{t_}": v for t_, v in zip(TIERS, at)}})
            L.append("")
        L += [f"### Paired difference to {REF} (FLatS eq. 5, MS MARCO, a = 0.5; pre-specified reference)", "",
              "| score | ΔAUROC two-way 95% | ΔAUROC KB 95% | ΔG two-way 95% | ΔG KB 95% | ΔG near / mid / far |", "|---|---|---|---|---|---|"]
        for gname, members in GROUPS:
            members = bg if members is None else [x for x in members if x in scores]
            for sc in members:
                if sc == REF:
                    continue
                d = C[C.score == sc]
                r1, r2 = twoway(d, "dA_ref", B, seed("dA_ref")), twoway(d, "dA_ref", B, seed("dA_ref", "kb"), kb_only=True)
                r3, r4 = twoway(d, "G_ref", B, seed("G_ref")), twoway(d, "G_ref", B, seed("G_ref", "kb"), kb_only=True)
                L.append(f"| {sc} | {fmt(r1)} | {fmt(r2)} | {fmt(r3, 3)} | {fmt(r4, 3)} | " + " / ".join(f"{d[d.tier == t_].G_ref.mean():+.3f}" for t_ in TIERS) + " |")
                for row in rows:
                    if row["score"] == sc:
                        row.update({"dA_ref": r1[0], "dA_ref_lo": r1[1], "dA_ref_hi": r1[2], "dA_ref_kb_lo": r2[1], "dA_ref_kb_hi": r2[2],
                                    "G_ref": r3[0], "G_ref_lo": r3[1], "G_ref_hi": r3[2], "G_ref_kb_lo": r4[1], "G_ref_kb_hi": r4[2],
                                    **{f"G_ref_{t_}": d[d.tier == t_].G_ref.mean() for t_ in TIERS}})
        L.append("")
        C.to_parquet(res / "tables" / f"cellsG_{rn}.parquet", index=False)
        L += ["### Upper bounds per C (out-of-fold AUROC; cell means) and the selected C", "", "| bound | " + " | ".join(f"C={x:g}" for x in cfg["oracle"]["C_grid"]) + " | max over C | share of cells by best C |", "|---|" + "---|" * (len(cfg["oracle"]["C_grid"]) + 2)]
        for nm in ("oracle", "tri_sup"):
            per = [C[C.score == f"{nm}@C{x:g}"].auroc.mean() for x in cfg["oracle"]["C_grid"]]
            bc = C[C.score == nm].best_C.value_counts(normalize=True).sort_index() if "best_C" in C else pd.Series(dtype=float)
            L.append(f"| {nm} | " + " | ".join(f"{v:.4f}" for v in per) + f" | {C[C.score == nm].auroc.mean():.4f} | " + ", ".join(f"{k_:g}: {v:.2f}" for k_, v in bc.items()) + " |")
        L.append("")
        mcols = ["pi", "T_var", "T_push", "d_abs_m", "dd_m", "rho_id", "rho_ood"]
        dm = C[C.score.isin([s_ for s_ in scores if s_.startswith("mix_") or s_ == "qz_ms"]) & C.pi.notna()] if "pi" in C else C.iloc[:0]
        if len(dm):
            L += ["### Moment decomposition of the subtraction scores (test queries; cell means): d_rel − d_abs = T_var (noise cancellation, π) + T_push (comparison term's ID − OOD mean)", "",
                  "| score | π | T_var | T_push | dd_m | ρ_ID(s,c) | ρ_OOD(s,c) |", "|---|---|---|---|---|---|---|"]
            for sc, g in dm.groupby("score", sort=False):
                L.append(f"| {sc} | {g.pi.mean():+.3f} | {g.T_var.mean():+.3f} | {g.T_push.mean():+.3f} | {g.dd_m.mean():+.3f} | {g.rho_id.mean():+.3f} | {g.rho_ood.mean():+.3f} |")
            if "beta" in C:
                L.append(f"\nqz_ms slope β (KB calibration queries; KB mean, 5 / 50 / 95%): " + " / ".join(f"{v:+.3f}" for v in C[C.score == "qz_ms"].groupby("kb").beta.first().quantile([.05, .5, .95])))
            L.append("")
        S = pd.DataFrame(rows)
        S.to_csv(res / "tables" / f"summary_{rn}.csv", index=False)
        if rn.startswith("a5_"):
            sign_rows.append(S.assign(model=rn[3:]))
    if sign_rows:
        A = pd.concat(sign_rows)
        L += ["## Cross-model signs (A5 subset; ΔAUROC vs abs | G vs abs, two-way 95%: + above 0, − below 0, 0 includes 0)", "",
              "| score | " + " | ".join(A.model.unique()) + " |", "|---|" + "---|" * A.model.nunique()]
        for sc, g in A.groupby("score", sort=False):
            cells = []
            for mdl in A.model.unique():
                x = g[g.model == mdl]
                if len(x) == 0:
                    cells.append("")
                    continue
                x = x.iloc[0]
                sgn = "+" if x.dA_lo > 0 else ("−" if x.dA_hi < 0 else "0")
                sg2 = "+" if x.G_lo > 0 else ("−" if x.G_hi < 0 else "0")
                cells.append(f"{sgn} ({x.dA:+.3f}) \\| {sg2} ({x.G:+.3f})")
            L.append(f"| {sc} | " + " | ".join(cells) + " |")
        L.append("")
    (res / "A13_summary.md").write_text("\n".join(L) + "\n")
    meta = run_metadata(cfg, "A13.02_report")
    meta["inputs"] = fingerprint([res / "tables" / f"cells_{rn}.parquet" for rn in runs] + [x for rn in runs for x in [res / "tables" / f"cells_extra_{rn}.parquet"] if x.exists()]
                                 + [PROJECT_ROOT / cfg["runs"][rn][x] / "tables" / "rows.parquet" for rn in runs for x in ("a1", "a2")])
    save_json(meta, res / "meta_02.json")
    log.info("summary written")


if __name__ == "__main__":
    main()
