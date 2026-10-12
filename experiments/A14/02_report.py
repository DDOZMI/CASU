from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import PROJECT_ROOT, fingerprint, get_logger, load_config, p, run_metadata, save_json, stable_seed

spec = importlib.util.spec_from_file_location("a13_02", PROJECT_ROOT / "experiments" / "A13" / "02_report.py")
R2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R2)
TIERS = ["near", "mid", "far"]
BINS = [(1, 1, "1"), (2, 3, "2-3"), (4, 10, "4-10"), (11, 999, "11+")]


def tier_boot(d, col, B, seed, kb_only=False):
    d = d[["kb", "ood", "tier", col]].dropna()
    kbs, oos = sorted(d.kb.unique()), sorted(set(d.ood.unique()) | set(d.kb.unique()))
    ki = d.kb.map({s: i for i, s in enumerate(kbs)}).to_numpy()
    oi = d.ood.map({s: i for i, s in enumerate(oos)}).to_numpy()
    v = d[col].to_numpy(float)
    rng = np.random.default_rng(seed)
    WK = rng.multinomial(len(kbs), [1 / len(kbs)] * len(kbs), size=B).astype(float)
    WO = rng.multinomial(len(oos), [1 / len(oos)] * len(oos), size=B).astype(float)
    W = WK[:, ki] * (1.0 if kb_only else WO[:, oi])
    out = {}
    for t_ in TIERS:
        msk = (d.tier == t_).to_numpy()
        out[t_] = (W[:, msk] @ v[msk]) / W[:, msk].sum(1)
    out["near-far"] = out["near"] - out["far"]
    res = {}
    for key, bs in out.items():
        msk = d.tier.isin(["near", "far"]).to_numpy() if key == "near-far" else (d.tier == key).to_numpy()
        pt = v[(d.tier == "near").to_numpy()].mean() - v[(d.tier == "far").to_numpy()].mean() if key == "near-far" else v[msk].mean()
        lo, hi = np.nanquantile(bs, [0.025, 0.975])
        res[key] = (float(pt), float(lo), float(hi))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--res", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = Path(args.res) if args.res else p(cfg, "results")
    log = get_logger("A14.02", res / "logs" / "02_report.log")
    B, ceil = cfg["metrics"]["bootstrap_B"], cfg["metrics"]["ceiling"]
    runs = [r for r in cfg["runs"] if not args.runs or r in args.runs]
    L = ["# A14 — O-blind wide comparison bank (StackExchange sites other than KB and O) vs FLatS", "",
         f"- Design: config bank / exclusion / scores (fixed before the run). Reference: {cfg['reference']}.",
         f"- Primary reading (user prediction, JOURNAL 20:05): {cfg['primary']}",
         "- Values: cell means (A1 cells, common with A13); G on cells with A_abs ≤ 0.98. ΔG / ΔAUROC = score − flats_ms@0.5, paired within cell, with the KB × O two-way and the KB bootstrap 95% (common weights across rows). Moments (test queries): T_push = comparison term's ID − OOD mean (neighbour coverage), T_var = noise cancellation (π). ρ_ID = corr(s_KB, c) on KB test queries; partial = given g (A12 factor from all sites but the KB; the bank's sites are in g's banks, so ρ(c,g) is inflated mechanically — read the partial as descriptive).",
         "- Moderator: nn_rank = tag rank, among the sites other than O and the KB, of the bank's site nearest to O (1 = O's nearest eligible site is in the bank); binned per draw.",
         "- At m = all the excl arm has fewer sites than noexcl (84 − |excluded|, KB-dependent): read the exclusion effect from fixed m < all (size-matched, same permutation).", ""]
    for rn in runs:
        R = pd.read_parquet(res / "tables" / f"rows_{rn}.parquet")
        ref = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["a13_results"] / "tables" / f"cellsG_{rn}.parquet")
        ref = ref[ref.score == R2.REF].set_index(["kb", "ood"])[["auroc", "auroc_abs", "G"]]
        num = ["auroc", "op_id_acc", "op_ood_acc", "fpr95", "pi", "T_var", "T_push", "dd_m", "rho_id", "rho_ood", "partial_id", "nn_rank", "nn_d", "n_set"]
        C = R.groupby(["kb", "ood", "tier", "arm", "m", "score"])[num].mean().reset_index()
        C["covered"] = R.assign(cv=R.nn_rank <= 3).groupby(["kb", "ood", "tier", "arm", "m", "score"]).cv.mean().to_numpy()
        C = C[C.set_index(["kb", "ood"]).index.isin(ref.index)].copy()
        rr = ref.reindex(pd.MultiIndex.from_frame(C[["kb", "ood"]]))
        C["auroc_abs"] = rr.auroc_abs.to_numpy()
        C["G"] = np.where(C.auroc_abs <= ceil, (C.auroc - C.auroc_abs) / (1 - C.auroc_abs), np.nan)
        C["dA_ref"] = C.auroc - rr.auroc.to_numpy()
        C["G_ref"] = C.G - rr.G.to_numpy()
        C.to_parquet(res / "tables" / f"cells_{rn}.parquet", index=False)
        ex = pd.read_csv(res / "tables" / f"excluded_{rn}.csv")
        L += [f"## {rn} ({C[['kb', 'ood']].drop_duplicates().shape[0]} cells; excluded sites per KB 5/50/95%: "
              + " / ".join(f"{v:.0f}" for v in ex.groupby("kb").excluded.sum().quantile([.05, .5, .95])) + ")", "",
              f"FLatS reference G (overall / near / mid / far): {ref.G.mean():+.3f} / " + " / ".join(
                  f"{ref.join(C.drop_duplicates(['kb', 'ood']).set_index(['kb', 'ood']).tier)[lambda x: x.tier == t_].G.mean():+.3f}" for t_ in TIERS), "",
              "| arm | m | score | G | ΔG vs FLatS two-way 95% | ΔG KB 95% | ΔG near [two-way 95%] | ΔG mid | ΔG far | ΔG near − far two-way 95% (KB 95%) | ΔAUROC vs FLatS two-way 95% | T_push / T_var | ρ_ID (partial) | coverage(nn ≤ 3) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        seed = lambda q, t=None: stable_seed(cfg["seed"], "A14_boot", rn, q, t)
        for (arm, m, sc), d in C.groupby(["arm", "m", "score"], sort=False):
            g2 = R2.twoway(d, "G_ref", B, seed("G_ref"))
            gk = R2.twoway(d, "G_ref", B, seed("G_ref", "kb"), kb_only=True)
            a2 = R2.twoway(d, "dA_ref", B, seed("dA_ref"))
            mo = f"{d.T_push.mean():+.3f} / {d.T_var.mean():+.3f}" if d.T_push.notna().any() else "—"
            rh = f"{d.rho_id.mean():+.3f} ({d.partial_id.mean():+.3f})" if d.rho_id.notna().any() else "—"
            tb = tier_boot(d, "G_ref", B, seed("tier"))
            tk = tier_boot(d, "G_ref", B, seed("tier", "kb"), kb_only=True)["near-far"]
            L.append(f"| {arm} | {m} | {sc} | {d.G.mean():+.3f} | {R2.fmt(g2, 3)} | {R2.fmt(gk, 3)} | {R2.fmt(tb['near'], 3)} | {R2.fmt(tb['mid'], 3)} | {R2.fmt(tb['far'], 3)} | "
                     f"{R2.fmt(tb['near-far'], 3)} ([{tk[1]:+.3f}, {tk[2]:+.3f}]) | {R2.fmt(a2)} | {mo} | {rh} | {d.covered.mean():.2f} |")
        Rd = R[R.score.str.startswith("S_tau")].merge(ref.reset_index().rename(columns={"auroc": "a_ref", "G": "G_refcell"}), on=["kb", "ood"])
        Rd = Rd[Rd.auroc_abs <= ceil]
        Rd["G_ref_draw"] = (Rd.auroc - Rd.a_ref) / (1 - Rd.auroc_abs)
        L += ["", "### Moderator: ΔG vs FLatS by nn_rank bin, per draw (S_tau@0.01 and S_tau@inf; G cells only; n = draw rows; nn_rank among sites other than O and the KB)", "",
              "| arm | m | score | " + " | ".join(f"nn {b[2]}" for b in BINS) + " |", "|---|---|---|" + "---|" * len(BINS)]
        for (arm, m, sc), d in Rd.groupby(["arm", "m", "score"], sort=False):
            cells = []
            for lo, hi, _ in BINS:
                x = d[(d.nn_rank >= lo) & (d.nn_rank <= hi)]
                cells.append(f"{x.G_ref_draw.mean():+.3f} (n={len(x)})" if len(x) else "")
            L.append(f"| {arm} | {m} | {sc} | " + " | ".join(cells) + " |")
        L += ["", "Per-draw share of draws by nn_rank bin and tier (excl, S_tau@0.01):", "", "| m | tier | " + " | ".join(f"nn {b[2]}" for b in BINS) + " |", "|---|---|" + "---|" * len(BINS)]
        for m_ in [str(x) for x in cfg["runs"][rn]["m_grid"]]:
            d = R[(R.arm == "excl") & (R.m == m_) & (R.score == "S_tau@0.01")]
            for t_ in TIERS:
                x = d[d.tier == t_]
                L.append(f"| {m_} | {t_} | " + " | ".join(f"{((x.nn_rank >= lo) & (x.nn_rank <= hi)).mean():.2f}" for lo, hi, _ in BINS) + " |")
        L += ["", "### Moderator by tier (m = all, excl): mean nn_rank and coverage", "", "| score | near | mid | far |", "|---|---|---|---|"]
        d = C[(C.arm == "excl") & (C.m == "all") & (C.score == "S_tau@0.01")]
        L.append("| nn_rank / covered | " + " | ".join(f"{d[d.tier == t_].nn_rank.mean():.1f} / {d[d.tier == t_].covered.mean():.2f}" for t_ in TIERS) + " |")
        d = C[(C.arm == "noexcl") & (C.m == "all") & (C.score == "S_tau@0.01")]
        L.append("| (noexcl) nn_rank / covered | " + " | ".join(f"{d[d.tier == t_].nn_rank.mean():.1f} / {d[d.tier == t_].covered.mean():.2f}" for t_ in TIERS) + " |")
        L.append("")
        log.info(f"{rn} done")
    (res / "A14_summary.md").write_text("\n".join(L) + "\n")
    meta = run_metadata(cfg, "A14.02_report")
    meta["inputs"] = fingerprint([res / "tables" / f"rows_{rn}.parquet" for rn in runs] + [PROJECT_ROOT / cfg["paths"]["a13_results"] / "tables" / f"cellsG_{rn}.parquet" for rn in runs])
    save_json(meta, res / "meta_02.json")


if __name__ == "__main__":
    main()
