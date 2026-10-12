from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import PROJECT_ROOT, fingerprint, get_logger, load_config, p, run_metadata, save_json, stable_seed


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R8 = _load("a8_report", PROJECT_ROOT / "experiments" / "A8" / "03_report.py")
A3 = _load("a3_rq1", PROJECT_ROOT / "experiments" / "A3" / "01_rq1.py")
twoway = R8.twoway
TIERS = ["near", "mid", "far"]
POOL = ("a2pool", "msunf", "msfil")


def derive(D, clip):
    D = D.copy()
    a = np.where(D.family.isin(POOL), D.param.astype(float), 1.0)
    D["alpha"] = a
    for cl in ("id", "ood"):
        D[f"{cl}_vS"] = D[f"{cl}_vs"] + a ** 2 * D[f"{cl}_vc"] - 2 * a * D[f"{cl}_csc"]
        for x, y, v in (("s", "c", "csc"), ("s", "g", "csg"), ("c", "g", "ccg")):
            vx = D[f"{cl}_vs"] if x == "s" else D[f"{cl}_vc"]
            vy = D[f"{cl}_vc"] if y == "c" else D[f"{cl}_vg"]
            D[f"{cl}_r_{x}{y}"] = D[f"{cl}_{v}"] / np.sqrt(vx * vy)
        D[f"{cl}_r_partial"] = (D[f"{cl}_r_sc"] - D[f"{cl}_r_sg"] * D[f"{cl}_r_cg"]) / np.sqrt((1 - D[f"{cl}_r_sg"] ** 2) * (1 - D[f"{cl}_r_cg"] ** 2))
        D[f"{cl}_r_hat"] = D[f"{cl}_r_sg"] * D[f"{cl}_r_cg"]
        D[f"{cl}_cov_hat"] = D[f"{cl}_csg"] * D[f"{cl}_ccg"] / D[f"{cl}_vg"]
        D[f"{cl}_r_gl"] = D[f"{cl}_cgl"] / np.sqrt(D[f"{cl}_vg"] * D[f"{cl}_vl"])
        D[f"{cl}_r_sl"] = D[f"{cl}_csl"] / np.sqrt(D[f"{cl}_vs"] * D[f"{cl}_vl"])
    sb_s = np.sqrt((D.id_vs + D.ood_vs) / 2)
    sb_c = np.sqrt((D.id_vc + D.ood_vc) / 2)
    sb_S = np.sqrt((D.id_vS + D.ood_vS) / 2)
    dmu_s, dmu_c = D.id_ms - D.ood_ms, a * (D.id_mc - D.ood_mc)
    D["sb_s"], D["sb_c"], D["sb_S"] = sb_s, sb_c, sb_S
    D["d_abs_m"], D["d_rel_m"] = dmu_s / sb_s, (dmu_s - dmu_c) / sb_S
    D["B"], D["mcp_id"], D["mcp_ood"] = dmu_s - dmu_c, a * D.id_mc, a * D.ood_mc
    D["pi"] = sb_s / sb_S - 1
    D["rho_bar"] = (D.id_csc + D.ood_csc) / 2 / (sb_s * sb_c)
    D["thr"] = a * sb_c / (2 * sb_s)
    D["sdr"] = sb_c / sb_s
    D["rho_bar_hat"] = (D.id_cov_hat + D.ood_cov_hat) / 2 / (sb_s * sb_c)
    D["pi_hat"] = sb_s / np.sqrt(sb_s ** 2 + a ** 2 * sb_c ** 2 - 2 * a * D.rho_bar_hat * sb_s * sb_c) - 1
    D["T_push"] = -dmu_c * (1 / sb_s + 1 / sb_S) / 2
    D["T_var"] = (dmu_s - dmu_c / 2) * (1 / sb_S - 1 / sb_s)
    dp = lambda x: np.sqrt(2) * norm.ppf(np.clip(x, clip, 1 - clip))
    D["dp"], D["dp_abs"] = dp(D.auroc), dp(D.auroc_abs)
    D["ddp"] = D.dp - D.dp_abs
    D["R_shape"] = D.ddp - (D.d_rel_m - D.d_abs_m)
    D["dA"] = D.auroc - D.auroc_abs
    D["dA_pred"] = norm.cdf(D.d_rel_m / np.sqrt(2)) - norm.cdf(D.d_abs_m / np.sqrt(2))
    D["dA_var"] = norm.cdf(D.d_abs_m * (1 + D.pi) / np.sqrt(2)) - norm.cdf(D.d_abs_m / np.sqrt(2))
    D["at_clip"] = (D.auroc >= 1 - clip) | (D.auroc_abs >= 1 - clip)
    return D


def cellmean(X, cols, keys=("kb", "ood")):
    return X.groupby(list(keys))[cols].mean().reset_index()


SEED = [20261006]


def tw(cells, col, B, *seed):
    return twoway(cells, col, B, stable_seed(SEED[0], "A10", *seed))


def fmt(t, nd=3):
    return "—" if t[3] == 0 else f"{t[0]:+.{nd}f} [{t[1]:+.{nd}f}, {t[2]:+.{nd}f}]"


def contrast(X, Y, clip):
    h = (1 / X.sb_S.to_numpy() + 1 / Y.sb_S.to_numpy()) / 2
    out = pd.DataFrame({"kb": X.kb.to_numpy(), "ood": X.ood.to_numpy(), "tier": X.tier.to_numpy(), "draw": X.draw.to_numpy(),
                        "a_abs": X.auroc_abs.to_numpy(), "at_clip": X.at_clip.to_numpy() | Y.at_clip.to_numpy()})
    out["ddp"] = X.dp.to_numpy() - Y.dp.to_numpy()
    out["dd_m"] = X.d_rel_m.to_numpy() - Y.d_rel_m.to_numpy()
    out["C_var"] = (X.B.to_numpy() + Y.B.to_numpy()) / 2 * (1 / X.sb_S.to_numpy() - 1 / Y.sb_S.to_numpy())
    out["C_idcost"] = -(X.mcp_id.to_numpy() - Y.mcp_id.to_numpy()) * h
    out["C_oodpush"] = (X.mcp_ood.to_numpy() - Y.mcp_ood.to_numpy()) * h
    out["R_shape"] = out.ddp - out.dd_m
    out["dA"] = X.auroc.to_numpy() - Y.auroc.to_numpy()
    out["dA_pred"] = norm.cdf(X.d_rel_m.to_numpy() / np.sqrt(2)) - norm.cdf(Y.d_rel_m.to_numpy() / np.sqrt(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--res", default=None)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--allow-smoke", action="store_true", help="scratch smoke only: accept stats from --smoke-pairs runs")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = Path(args.res) if args.res else p(cfg, "results")
    (res / "tables").mkdir(parents=True, exist_ok=True)
    log = get_logger("A10.02", res / "logs" / "02_analyze.log")
    clip, B = cfg["dprime_clip"], cfg["bootstrap_B"]
    SEED[0] = cfg["seed"]
    runs = list(args.runs or cfg["runs"])
    missing = [r for r in runs if not (res / "tables" / f"stats_{r}.parquet").exists()]
    assert not missing, f"missing stats for runs {missing} (pass --runs to analyse a subset)"
    for r in runs:
        mt = res / f"meta_01_{r}.json"
        assert args.allow_smoke or not mt.exists() or not __import__("json").load(open(mt)).get("smoke_pairs"), f"{r}: stats come from a smoke run"
    S = {r: derive(pd.read_parquet(res / "tables" / f"stats_{r}.parquet"), clip) for r in runs}
    L = ["# A10 — structural closure: B factor model, A four-term accounting, C aggregation relations, D AUROC transfer", "",
         "- Plan and pre-run forecasts: JOURNAL 2026-10-07 13:03. All quantities from per-class sufficient statistics (01); "
         "means over (KB, O) cells of the draw mean; [ ] KB × O two-way bootstrap 95% over the run's sites (B = " + str(B) + ").", ""]
    idmax = max(float(np.nanmax(np.abs(X.d_rel_m - X.d_rel_direct))) for X in S.values())
    assert idmax < 1e-9, idmax
    L += [f"- Identity check (the only closure check here that could fail): d_rel from the sufficient statistics vs the moment d of S computed directly, max |diff| = {idmax:.1e} over {sum(len(X) for X in S.values())} rows.",
          "- Definitional (cannot fail, shown as accounting): A's four terms sum to Δd′_AUC because R_shape is the remainder; C's inequalities (asserted in 01) and the τ = ∞ 1/m law (same fixed members across roles) hold by algebra.",
          "- g is rebuilt per comparison set (left-out banks exclude the set's own banks), so B quantities compared across sets combine a change of c with a (small) change of g; sets sharing g: site / random / pooled (same 4 banks), MS unf / fil. "
          "PC1 share depends on the number of left-out banks (main ≈ 80, A5 ≈ 24–28, A8 ≈ 80): compare within a run. g is an estimated PC1, so the one-factor prediction is attenuated and the partial correlation inflated by its measurement error; a residual partial ρ is not by itself evidence of a second component (review 2026-10-07).",
          "- A1 (ii)−(iii) and (i)−(iii) parts are the mean of the per-slot-pair midpoint splits (totals equal mean(N) − mean(Z) exactly). Cell sets differ between families in A5 runs (A1-dropped pairs have A2/MS rows only); n cells are listed.", ""]

    L += ["## B. One-factor model of the noise-cancellation term", "",
          "g = PC1 of the KB calibration queries' similarities to the left-out banks (not KB, O, or the set's own banks). Cell medians; Spearman over cells.", ""]
    hdr = ["run", "set", "τ/α", "left-out banks", "corr(w, KB-nearness)", "PC1 share", "ρ_ID(s,c)", "ρ_OOD(s,c)", "ρ̄", "ρ̄ one-factor", "Spearman(ρ̄̂, ρ̄)", "partial ρ_ID(s,c|g)", "partial ρ_OOD",
           "ρ_ID(s,g)", "ρ_OOD(s,g)", "ρ_ID(c,g)", "ρ_OOD(c,g)", "π", "π̂ (one-factor)", "Spearman(π̂, π)", "sign agree π̂/π", "ρ_ID(g, log len)", "ρ_ID(s, log len)"]
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    brows = []
    cols = ["n_left_out", "w_kbnear_corr", "pc1_share", "id_r_sc", "ood_r_sc", "rho_bar", "rho_bar_hat", "id_r_partial", "ood_r_partial", "id_r_sg", "ood_r_sg", "id_r_cg", "ood_r_cg",
            "pi", "pi_hat", "id_r_gl", "id_r_sl"]
    for r, X in S.items():
        X = X.assign(set=np.where(X.family == "a1", "a1:" + X.role.astype(str).str.replace(r"\d$", "", regex=True), X.family))
        X = X[X.set.isin(["a1:Z", "a1:N", "a2site", "a2pool", "a2rand", "msunf", "msfil"]) | X.family.eq("a4")]
        X = X.assign(set=np.where(X.family == "a4", "a4:" + X.role.astype(str), X.set))
        for (st, par), g in X.groupby(["set", "param"], sort=False):
            c = cellmean(g, cols)
            sp1 = spearmanr(c.rho_bar_hat, c.rho_bar).statistic
            sp2 = spearmanr(c.pi_hat, c.pi).statistic
            agree = float((np.sign(c.pi_hat) == np.sign(c.pi)).mean())
            med = c[cols].median()
            brows.append({"run": r, "set": st, "param": par, **med.to_dict(), "sp_rho": sp1, "sp_pi": sp2, "sign_agree": agree, "n_cells": len(c)})
            L.append(f"| {r} | {st} | {par} | {med.n_left_out:.0f} | {med.w_kbnear_corr:+.2f} | {med.pc1_share:.2f} | {med.id_r_sc:+.2f} | {med.ood_r_sc:+.2f} | {med.rho_bar:+.2f} | {med.rho_bar_hat:+.2f} | {sp1:.2f} | "
                     f"{med.id_r_partial:+.2f} | {med.ood_r_partial:+.2f} | {med.id_r_sg:+.2f} | {med.ood_r_sg:+.2f} | {med.id_r_cg:+.2f} | {med.ood_r_cg:+.2f} | "
                     f"{med.pi:+.3f} | {med.pi_hat:+.3f} | {sp2:.2f} | {agree:.2f} | {med.id_r_gl:+.2f} | {med.id_r_sl:+.2f} |")
    pd.DataFrame(brows).to_csv(res / "tables" / "B_factor.csv", index=False)
    L += ["", "MS MARCO filter (A9 arm) through the factor: paired fil − unf per cell (α-free quantities), mean [two-way 95%]", "",
          "| run | Δρ̄ | Δρ_ID(c,g) | Δρ_OOD(c,g) | Δpartial ρ_ID(s,c|g) | Δρ_ID(s,c) |", "|---|---|---|---|---|---|"]
    for r, X in S.items():
        if "msfil" not in set(X.family):
            continue
        u = X[(X.family == "msunf") & (X.param == "1")].set_index(["kb", "ood", "draw"])
        f_ = X[(X.family == "msfil") & (X.param == "1")].set_index(["kb", "ood", "draw"]).loc[u.index]
        dd = pd.DataFrame({k_: f_[k_] - u[k_] for k_ in ("rho_bar", "id_r_cg", "ood_r_cg", "id_r_partial", "id_r_sc")}).reset_index()
        cl = cellmean(dd, ["rho_bar", "id_r_cg", "ood_r_cg", "id_r_partial", "id_r_sc"])
        L.append(f"| {r} | " + " | ".join(fmt(tw(cl, k_, B, r, "Bms", k_)) for k_ in ("rho_bar", "id_r_cg", "ood_r_cg", "id_r_partial", "id_r_sc")) + " |")

    L += ["", "## A. Four-term accounting of every reported contrast", "",
          "Single score vs abs: Δd′_AUC = T_var (noise cancellation) + T_push (mean push) + R_shape. Two scores X vs Y: Δd′_AUC = C_var + C_idcost + C_oodpush + R_shape "
          "(C_idcost > 0 = less ID cost in X; C_oodpush > 0 = more OOD push in X). R_shape = AUROC-based minus moment d (binormal transfer remainder).", ""]
    arows = []

    def account(r, name, cells_draw, kind):
        cells_draw = cells_draw.assign(at_clip=cells_draw.at_clip.astype(float))
        c = cellmean(cells_draw, [k_ for k_ in ("ddp", "dd_m", "T_var", "T_push", "C_var", "C_idcost", "C_oodpush", "R_shape", "dA", "dA_pred", "a_abs", "at_clip") if k_ in cells_draw])
        c = c.merge(cells_draw.drop_duplicates(["kb", "ood"])[["kb", "ood", "tier"]], on=["kb", "ood"])
        parts = ["T_var", "T_push"] if kind == "single" else ["C_var", "C_idcost", "C_oodpush"]
        row = {"run": r, "contrast": name}
        cells = []
        for k_ in ["ddp", *parts, "R_shape"]:
            t = tw(c, k_, B, r, "A", name, k_)
            row.update({k_: t[0], f"{k_}_lo": t[1], f"{k_}_hi": t[2]})
            cells.append(fmt(t))
        tot = float(np.abs(c.ddp).mean())
        row["R_share"] = float(np.abs(c.R_shape).mean()) / tot if tot > 0 else np.nan
        nc_ = c[c.at_clip == 0]
        row["clip_draw_share"] = float(cells_draw.at_clip.mean())
        row["R_share_noclip"] = float(np.abs(nc_.R_shape).mean() / max(np.abs(nc_.ddp).mean(), 1e-12)) if len(nc_) else np.nan
        row["n_cells_noclip"] = len(nc_)
        for t_ in TIERS:
            q = c[c.tier == t_]
            row.update({f"{k_}_{t_}": float(q[k_].mean()) if len(q) else np.nan for k_ in ["ddp", *parts, "R_shape"]})
        row["n_cells"] = len(c)
        arows.append(row)
        pad = cells[:1] + (cells[1:3] + ["—"] if kind == "single" else cells[1:4]) + cells[-1:]
        tiers = " / ".join(f"{row.get('R_shape_' + t_, np.nan):+.2f}" for t_ in TIERS)
        L.append(f"| {r} | {name} | " + " | ".join(pad) + f" | {row['R_share']:.2f} | {row['clip_draw_share']:.3f} | {row['R_share_noclip']:.2f} ({len(nc_)}) | {tiers} | {len(c)} |")
        return c

    L += ["| run | contrast | Δd′_AUC | T_var / C_var | T_push / C_idcost | — / C_oodpush | R_shape | mean|R|/mean|Δd′| | draws at clip | mean|R|/mean|Δd′| on cells without clipped draws (n) | R_shape near / mid / far | cells |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    keyd = ["kb", "ood", "draw"]
    a_cells = {}
    for r, X in S.items():
        for t in cfg["taus"]:
            a1 = X[(X.family == "a1") & (X.param == t)].set_index([*keyd, "role"])
            if len(a1):
                roles = a1.index.get_level_values("role")
                sl = lambda ro: a1[roles == ro].reset_index(level="role", drop=True)
                Z1, Z2 = sl("Z1"), sl("Z2")
                ix = Z1.index.intersection(Z2.index)
                Zs = [Z1.loc[ix].reset_index(), Z2.loc[ix].reset_index()]
                off = pd.concat(Zs).assign(at_clip=lambda x_: x_.at_clip.astype(float)).groupby(keyd + ["tier"])[["ddp", "d_rel_m", "d_abs_m", "T_var", "T_push", "R_shape", "dA", "dA_pred", "auroc_abs", "at_clip"]].mean().reset_index()
                off["at_clip"] = off.at_clip > 0
                off["dd_m"] = off.d_rel_m - off.d_abs_m
                off["a_abs"] = off.auroc_abs
                account(r, f"A1 (iii)−abs @{t}", off, "single")
                N1, N2 = sl("N1").loc[ix].reset_index(), sl("N2").loc[ix].reset_index()
                pairs_ = [contrast(x, z, clip) for x in (N1, N2) for z in Zs]
                main_ = pd.concat(pairs_).groupby(keyd + ["tier"]).agg(lambda v: v.any() if v.dtype == bool else v.mean()).reset_index()
                a_cells[(r, f"main@{t}")] = account(r, f"A1 (ii)−(iii) @{t}", main_, "two")
                Zm, O = sl("Zm").loc[ix].reset_index(), sl("O").loc[ix].reset_index()
                account(r, f"A1 N1−Zm @{t}", contrast(N1, Zm, clip), "two")
                leak = pd.concat([contrast(O, z, clip) for z in Zs]).groupby(keyd + ["tier"]).agg(lambda v: v.any() if v.dtype == bool else v.mean()).reset_index()
                account(r, f"A1 (i)−(iii) @{t}", leak, "two")
            for fam in ("a2site", "a2rand"):
                q = X[(X.family == fam) & (X.param == t)].copy()
                if len(q):
                    q["dd_m"], q["a_abs"] = q.d_rel_m - q.d_abs_m, q.auroc_abs
                    a_cells[(r, f"{fam}@{t}")] = account(r, f"A2 {fam[2:]} vs abs @{t}", q, "single")
            st_ = X[(X.family == "a2site") & (X.param == t)].set_index(keyd)
            rd_ = X[(X.family == "a2rand") & (X.param == t)].set_index(keyd)
            if len(st_) and len(rd_):
                ix2 = st_.index.intersection(rd_.index)
                account(r, f"A2 site − random @{t}", contrast(st_.loc[ix2].reset_index(), rd_.loc[ix2].reset_index(), clip), "two")
            a4 = X[(X.family == "a4") & (X.param == t)].set_index([*keyd, "role"])
            if len(a4):
                rl = a4.index.get_level_values("role")
                L0 = a4[rl == "L0"].reset_index(level="role", drop=True)
                for lv in ("L1", "L2", "L3"):
                    Lv = a4[rl == lv].reset_index(level="role", drop=True)
                    ix3 = Lv.index.intersection(L0.index)
                    account(r, f"A4 {lv}−L0 @{t}", contrast(Lv.loc[ix3].reset_index(), L0.loc[ix3].reset_index(), clip), "two")
        for a in ("0.5", "1"):
            for fam, nm in (("a2pool", "A2 SE pooled mix"), ("msunf", "A2 MS pooled mix")):
                q = X[(X.family == fam) & (X.param == a)].copy()
                if len(q):
                    q["dd_m"], q["a_abs"] = q.d_rel_m - q.d_abs_m, q.auroc_abs
                    a_cells[(r, f"{fam}@{a}")] = account(r, f"{nm} vs abs @{a}", q, "single")
            u = X[(X.family == "msunf") & (X.param == a)].set_index(keyd)
            f_ = X[(X.family == "msfil") & (X.param == a)].set_index(keyd)
            if len(f_):
                account(r, f"A9 filter − unfiltered @{a}", contrast(f_.loc[u.index].reset_index(), u.reset_index(), clip), "two")
    pd.DataFrame(arows).to_csv(res / "tables" / "A_terms.csv", index=False)

    L += ["", "## C. Aggregation relations", "",
          "Checked in 01 on every query (assert): pooled top-10 mean ≥ best-group top-10 mean with equality when concentrated; max_j c_j − τ log m ≤ c_0.01 ≤ max_j c_j (A1 slots).", "",
          "| run | set | concentrated share ID / OOD | mean (pooled − max group) ID / OOD |", "|---|---|---|---|"]
    crow = []
    for r, X in S.items():
        for fam in ("a2site", "a2rand", "a1", "a4"):
            q = X[(X.family == fam) & (X.param == "inf")]
            if not len(q):
                continue
            grp = [("all", q)] if fam in ("a2site", "a2rand") else [(ro_, g_) for ro_, g_ in q.groupby("role") if fam == "a4" or ro_ in ("N1", "Z1", "K")]
            for ro, g in grp:
                c = cellmean(g, ["id_conc", "ood_conc", "id_gap_pool_max", "ood_gap_pool_max"])
                crow.append({"run": r, "family": fam, "role": ro, **c[["id_conc", "ood_conc", "id_gap_pool_max", "ood_gap_pool_max"]].mean().to_dict()})
                L.append(f"| {r} | {fam}{'' if ro == 'all' else ':' + ro} | {c.id_conc.mean():.2f} / {c.ood_conc.mean():.2f} | {c.id_gap_pool_max.mean():.4f} / {c.ood_gap_pool_max.mean():.4f} |")
    pd.DataFrame(crow).to_csv(res / "tables" / "C_concentration.csv", index=False)
    L += ["", "Cell level (τ = 0.01): Spearman of the score difference with the concentration share (ID + OOD mean); mean values", "",
          "| run | contrast | mean Δd′_AUC | mean conc (site) | mean conc (random) | Spearman(Δd′, Δconc or conc) |", "|---|---|---|---|---|---|"]
    for r, X in S.items():
        st_ = X[(X.family == "a2site") & (X.param == "0.01")].set_index(keyd)
        pl_ = X[(X.family == "a2pool") & (X.param == "1")].set_index(keyd)
        rd_ = X[(X.family == "a2rand") & (X.param == "0.01")].set_index(keyd)
        if not len(st_):
            continue
        cs = (st_.id_conc + st_.ood_conc) / 2
        d1 = (pl_.dp - st_.dp).rename("d").to_frame().assign(conc=cs).reset_index()
        c1 = cellmean(d1, ["d", "conc"])
        L.append(f"| {r} | SE pooled mix@1 − site@0.01 | {c1.d.mean():+.3f} | {c1.conc.mean():.2f} | — | {spearmanr(c1.conc, c1.d).statistic:+.2f} (with site conc) |")
        if len(rd_):
            cr = (rd_.id_conc + rd_.ood_conc) / 2
            d2 = (rd_.dp - st_.dp).rename("d").to_frame().assign(dconc=cs - cr, cr=cr, cs=cs).reset_index()
            c2 = cellmean(d2, ["d", "dconc", "cr", "cs"])
            L.append(f"| {r} | random − site @0.01 | {c2.d.mean():+.3f} | {c2.cs.mean():.2f} | {c2.cr.mean():.2f} | {spearmanr(c2.dconc, c2.d).statistic:+.2f} (with conc site − random) |")
    L += ["", "A1 slot contrasts in raw cosine units (mean over cells): comparison-term change N − Z on ID / OOD vs the closed forms", "",
          "| run | class | τ = ∞ observed | (s_slot,N − s_slot,Z)/m | max |diff| | τ = 0.01 observed | exceedance law E[(s_N−M_F)₊ − (s_Z−M_F)₊] | Spearman over cells |", "|---|---|---|---|---|---|---|---|"]
    m = cfg["m"]
    for r, X in S.items():
        a1 = X[X.family == "a1"]
        if not len(a1):
            continue
        for cl in ("id", "ood"):
            v = {}
            for t in ("inf", "0.01"):
                q = a1[a1.param == t].pivot_table(index=keyd, columns="role", values=[f"{cl}_mc", f"{cl}_sslot", f"{cl}_exc"])
                obs = (q[(f"{cl}_mc", "N1")] + q[(f"{cl}_mc", "N2")] - q[(f"{cl}_mc", "Z1")] - q[(f"{cl}_mc", "Z2")]) / 2
                lin = (q[(f"{cl}_sslot", "N1")] + q[(f"{cl}_sslot", "N2")] - q[(f"{cl}_sslot", "Z1")] - q[(f"{cl}_sslot", "Z2")]) / 2 / m
                exc = (q[(f"{cl}_exc", "N1")] + q[(f"{cl}_exc", "N2")] - q[(f"{cl}_exc", "Z1")] - q[(f"{cl}_exc", "Z2")]) / 2
                v[t] = pd.DataFrame({"obs": obs, "lin": lin, "exc": exc}).reset_index().groupby(["kb", "ood"]).mean(numeric_only=True)
            L.append(f"| {r} | {cl.upper()} | {v['inf'].obs.mean():+.4f} | {v['inf'].lin.mean():+.4f} | {np.abs(v['inf'].obs - v['inf'].lin).max():.1e} | "
                     f"{v['0.01'].obs.mean():+.4f} | {v['0.01'].exc.mean():+.4f} | {spearmanr(v['0.01'].obs, v['0.01'].exc).statistic:.2f} |")

    L += ["", "## D. AUROC transfer and the RQ1 slope", "",
          "Predicted ΔAUROC = Φ(d_rel/√2) − Φ(d_abs/√2) from the moment d (two-score: Φ(d_X/√2) − Φ(d_Y/√2)). Slope = A3 interval of record (crossed random intercepts + slopes) on the KB–O tag rank per 1 SD; > 0 = near gains less.", "",
          "| run | outcome | ΔA obs near / mid / far | ΔA pred near / mid / far | Spearman(pred, obs) cells | slope ΔA obs | slope ΔA pred | Δd′ obs near / mid / far | slope Δd′ obs (reference) |", "|---|---|---|---|---|---|---|---|---|"]
    drows = []
    outcomes = [("a2site@0.01", "A2 site @0.01"), ("a2site@inf", "A2 site @inf"), ("a2pool@1", "A2 SE mix@1"), ("a2pool@0.5", "A2 SE mix@0.5"),
                ("main@0.01", "A1 (ii)−(iii) @0.01"), ("main@inf", "A1 (ii)−(iii) @inf")]
    for r in S:
        rank = pd.read_csv(PROJECT_ROOT / cfg["runs"][r]["a1"] / "tables" / "pairs.csv")[["kb", "ood", "rank_kb_o"]]
        for key, nm in outcomes:
            if (r, key) not in a_cells:
                continue
            c = a_cells[(r, key)].merge(rank, on=["kb", "ood"])
            tm = c.groupby("tier")[["dA", "dA_pred", "ddp"]].mean().reindex(TIERS)
            so, spd, sdd = A3.mixed_slope(c, "dA", "rank_kb_o"), A3.mixed_slope(c, "dA_pred", "rank_kb_o"), A3.mixed_slope(c, "ddp", "rank_kb_o")
            sp = spearmanr(c.dA_pred, c.dA).statistic
            drows.append({"run": r, "outcome": key, **{f"dA_{t_}": tm.loc[t_, "dA"] for t_ in TIERS}, **{f"dApred_{t_}": tm.loc[t_, "dA_pred"] for t_ in TIERS},
                          "spearman": sp, "slope_obs": so["slope"], "slope_obs_lo": so["lo"], "slope_obs_hi": so["hi"], "slope_pred": spd["slope"],
                          "slope_pred_lo": spd["lo"], "slope_pred_hi": spd["hi"], "conv_obs": so["converged"], "conv_pred": spd["converged"], "slope_ddp": sdd["slope"], "slope_ddp_lo": sdd["lo"], "slope_ddp_hi": sdd["hi"]})
            ff = lambda s_: f"{s_['slope']:+.4f} [{s_['lo']:+.4f}, {s_['hi']:+.4f}]" + ("" if s_["converged"] else " (nc)")
            L.append(f"| {r} | {nm} | " + " / ".join(f"{v:+.3f}" for v in tm.dA) + " | " + " / ".join(f"{v:+.3f}" for v in tm.dA_pred) +
                     f" | {sp:.2f} | {ff(so)} | {ff(spd)} | " + " / ".join(f"{v:+.2f}" for v in tm.ddp) + f" | {ff(sdd)} |")
    pd.DataFrame(drows).to_csv(res / "tables" / "D_transfer.csv", index=False)
    L += ["", "π by tier (A2 site sets; cell medians): π < 0 ⇔ ρ̄ < α σ̄_c/(2σ̄_s) = thr", "",
          "| run | τ | π near / mid / far | ρ̄ near / mid / far | σ̄_c/σ̄_s near / mid / far | thr near / mid / far | ρ̄ one-factor near / mid / far | d_abs near / mid / far |", "|---|---|---|---|---|---|---|---|"]
    for r, X in S.items():
        for t in cfg["taus"]:
            q = X[(X.family == "a2site") & (X.param == t)]
            if not len(q):
                continue
            c = cellmean(q, ["pi", "rho_bar", "sdr", "thr", "rho_bar_hat", "d_abs_m"]).merge(q.drop_duplicates(["kb", "ood"])[["kb", "ood", "tier"]], on=["kb", "ood"])
            md = c.groupby("tier")[["pi", "rho_bar", "sdr", "thr", "rho_bar_hat", "d_abs_m"]].median().reindex(TIERS)
            L.append(f"| {r} | {t} | " + " | ".join(" / ".join(f"{v:+.3f}" if k_ == "pi" else f"{v:.2f}" for v in md[k_]) for k_ in ["pi", "rho_bar", "sdr", "thr", "rho_bar_hat", "d_abs_m"]) + " |")
    dpk = cfg["d_peak"]
    edges = [float("inf") if e == "inf" else float(e) for e in dpk["bins"]]
    L += ["", "Constant-π curve, variance part only (A2 site sets): ΔA_var = Φ(d_abs(1+π)/√2) − Φ(d_abs/√2) ≈ φ(d/√2)·π·d/√2, peaking at d_abs = √2 (A_abs ≈ 0.84). "
          f"Per d_abs bin: ratio of means ΣΔA_var/Σπ vs the curve at the bin's mean d; observed total ΔA alongside (includes push and R_shape). Cells with |π| ≤ {dpk['pi_min_abs']} dropped (counted).", "",
          "| run | τ | d_abs bin | n (dropped) | A_abs | π mean | ΣΔA_var/Σπ | φ(d/√2)·d/√2 | ΔA_var mean | ΔA obs mean |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r, X in S.items():
        for t in cfg["taus"]:
            q = X[(X.family == "a2site") & (X.param == t)]
            if not len(q):
                continue
            c = cellmean(q, ["pi", "d_abs_m", "dA", "dA_var", "auroc_abs"])
            c["bin"] = pd.cut(c.d_abs_m, edges, include_lowest=True)
            for b_, g in c.groupby("bin", observed=True):
                k_ = g[np.abs(g.pi) > dpk["pi_min_abs"]]
                if not len(k_):
                    continue
                dm = k_.d_abs_m.mean()
                L.append(f"| {r} | {t} | {b_} | {len(k_)} ({len(g) - len(k_)}) | {k_.auroc_abs.mean():.3f} | {k_.pi.mean():+.3f} | {k_.dA_var.sum() / k_.pi.sum():+.3f} | "
                         f"{norm.pdf(dm / np.sqrt(2)) * dm / np.sqrt(2):.3f} | {k_.dA_var.mean():+.4f} | {k_.dA.mean():+.4f} |")
    (res / "A10_summary.md").write_text("\n".join(L) + "\n")
    meta = run_metadata(cfg, "A10.02_analyze")
    meta["inputs"] = fingerprint([res / "tables" / f"stats_{r}.parquet" for r in runs] + [PROJECT_ROOT / "experiments" / "A8" / "03_report.py", PROJECT_ROOT / "experiments" / "A3" / "01_rq1.py"])
    meta["outputs"] = fingerprint([res / "A10_summary.md"] + sorted((res / "tables").glob("[ABCD]_*.csv")))
    save_json(meta, res / "meta_02.json")
    log.info("A10_summary.md written")


if __name__ == "__main__":
    main()
