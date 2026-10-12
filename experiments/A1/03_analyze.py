from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json, stable_seed

CMETRICS = ["ddp", "dA", "G", "dFPR", "dOP"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--res", default=None)
    ap.add_argument("--emb-smoke", action="store_true")
    ap.add_argument("--retriever", default=None, help="default: config detector")
    args = ap.parse_args()
    cfg = load_config(args.config)
    r = args.retriever or cfg["detector"]
    res = Path(args.res) if args.res else (p(cfg, "results", r) if cfg.get("results_per_retriever") else p(cfg, "results"))
    log = get_logger("A1.03", res / "logs" / "03_analyze.log")
    B = cfg["metrics"]["bootstrap_B"]
    edir = p(cfg, "emb", "_smoke" if args.emb_smoke else "", r)
    sites = load_json(edir / "sites.json")
    sidx = {s: i for i, s in enumerate(sites)}
    D = pd.read_csv(p(cfg, "a0_processed", cfg.get("distance_file", "tag_js.csv")), index_col=0).loc[sites, sites].to_numpy()
    rankm = np.argsort(np.argsort(D + np.eye(len(sites)) * 9, axis=1, kind="stable"), axis=1, kind="stable") + 1
    r2 = cfg["second_distance_encoder"][r]
    C2 = np.load(edir / f"centroids_{r2}.npy")
    D2 = 1 - C2 @ C2.T
    bands = load_json(p(cfg, "a0_processed", "bands.json"))
    q_nb = bands["q_neighbor_value"]
    cat = {s: v["category"] for s, v in load_json(p(cfg, "a0_processed", "sites.json")).items()}
    pairs = pd.read_csv(res / "tables" / "pairs.csv")
    df = pd.read_parquet(res / "tables" / "rows.parquet")
    df["param"] = df["param"].astype(str)
    df["score"] = df.family + "@" + df.param
    df["dA"] = df.auroc - df.auroc_abs
    df["dFPR"] = df.fpr95 - df.fpr95_abs
    df["dOP"] = df.op_ood_acc - df.op_ood_acc_abs

    key = ["score", "kb", "ood", "draw"]
    W = df.pivot_table(index=key, columns="role", values=CMETRICS + ["mu_id", "mu_ood", "sd_id"], aggfunc="first")
    C = pd.DataFrame(index=W.index)
    for mt in CMETRICS:
        z = (W[(mt, "Z1")] + W[(mt, "Z2")]) / 2
        C[f"main_{mt}"] = (W[(mt, "N1")] + W[(mt, "N2")]) / 2 - z
        C[f"off_{mt}"] = z
        C[f"leak_{mt}"] = W[(mt, "O")] - z
        C[f"kcost_{mt}"] = W[(mt, "K")] - z
        C[f"matched_{mt}"] = W[(mt, "N1")] - W[(mt, "Zm")]
        C[f"N_{mt}"] = (W[(mt, "N1")] + W[(mt, "N2")]) / 2
        C[f"O_{mt}"] = W[(mt, "O")]
        if "Op" in W[mt].columns:
            C[f"leakp_{mt}"] = W[(mt, "Op")] - z
            C[f"opmo_{mt}"] = W[(mt, "Op")] - W[(mt, "O")]
    zsd = (W[("sd_id", "Z1")] + W[("sd_id", "Z2")]) / 2
    for side in ("id", "ood"):
        C[f"{side}_shift"] = ((W[(f"mu_{side}", "N1")] + W[(f"mu_{side}", "N2")]) - (W[(f"mu_{side}", "Z1")] + W[(f"mu_{side}", "Z2")])) / 2 / zsd
    clip = cfg["metrics"]["dprime_clip"]
    A_ = df.pivot_table(index=key, columns="role", values="auroc", aggfunc="first")
    A0_ = df.pivot_table(index=key, columns="role", values="auroc_abs", aggfunc="first")["O"]
    C["at_clip"] = ((A_[["N1", "N2", "Z1", "Z2"]] >= 1 - clip).any(axis=1) | (A0_ >= 1 - clip)).to_numpy()
    C["a_abs"] = A0_.to_numpy()
    C = C.reset_index()
    dr = pd.read_csv(res / "tables" / "draws.csv")
    n1 = dr[dr.role == "N1"].set_index(["kb", "ood", "draw"]).slot
    C["N1"] = [n1[(a, b, c)] for a, b, c in zip(C.kb, C.ood, C.draw)]
    zrel = dr.drop_duplicates(["kb", "ood", "draw"]).set_index(["kb", "ood", "draw"]).z_relaxed.astype(int)
    C["z_relaxed"] = [zrel[(a, b, c)] for a, b, c in zip(C.kb, C.ood, C.draw)]
    same_cat = cfg.get("nb_same_category", True)
    C["n1_in_band"] = [(cat[s] == cat[o] or not same_cat) and D[sidx[s], sidx[o]] <= q_nb for s, o in zip(C.N1, C.ood)]
    C["n1_same_cat_o"] = [cat[s] == cat[o] for s, o in zip(C.N1, C.ood)]
    pinfo = pairs.set_index(["kb", "ood"])
    C["tier"] = [pinfo.loc[(a, b), "tier"] for a, b in zip(C.kb, C.ood)]
    C["kb_o_same_cat"] = [cat[a] == cat[b] for a, b in zip(C.kb, C.ood)]
    keep_cat = cfg.get("ood_cat_keep", ("technology", "culturerecreation", "lifearts", "science"))
    C["ood_cat"] = [cat[b] if cat[b] in keep_cat else "other" for b in C.ood]
    C.to_parquet(res / "tables" / "contrasts.parquet", index=False)

    rng = np.random.default_rng(stable_seed(cfg["seed"], "A1_boot2"))
    nS = len(sites)
    set_mode = cfg["pairs"].get("mode") == "set"
    if set_mode:
        ui = np.array([sidx[s_] for s_ in cfg["pairs"]["kb_set"]])
        nU = len(ui)
        CK, CO = np.zeros((B, nS)), np.zeros((B, nS))
        CK[:, ui] = rng.multinomial(nU, [1 / nU] * nU, size=B)
        CO[:, ui] = rng.multinomial(nU, [1 / nU] * nU, size=B)
    else:
        CK = rng.multinomial(nS, [1 / nS] * nS, size=B).astype(float)
        CO = rng.multinomial(nS, [1 / nS] * nS, size=B).astype(float)
    clus = None
    if cfg.get("cluster_bootstrap") == "subset_triples":
        groups = load_json(p(cfg, "a0_processed", "subset.json"))["groups"]
        gi = {s_: i for i, g_ in enumerate(groups) for s_ in g_["sites"]}
        clus = np.array([gi[s_] for s_ in sites])
        nG = len(groups)
        rngc = np.random.default_rng(stable_seed(cfg["seed"], "A1_boot_cluster"))
        CKc = rngc.multinomial(nG, [1 / nG] * nG, size=B).astype(float)
        COc = rngc.multinomial(nG, [1 / nG] * nG, size=B).astype(float)
    rng1 = np.random.default_rng(stable_seed(cfg["seed"], "A1_boot_kb"))
    if set_mode:
        CK1 = np.zeros((B, nS))
        CK1[:, ui] = rng1.multinomial(nU, [1 / nU] * nU, size=B)
    else:
        CK1 = rng1.multinomial(nS, [1 / nS] * nS, size=B).astype(float)

    def cells(sub, col):
        s = sub.dropna(subset=[col])
        cell = s.groupby(["kb", "ood"])[col].mean()
        return cell, np.array([sidx[a] for a, _ in cell.index]), np.array([sidx[b] for _, b in cell.index])

    def boot(sub, col, num=None):
        cell, ki, oi = cells(sub, col)
        if len(cell) == 0:
            return (np.nan,) * 3 + (0,) + (np.nan,) * 2
        v = cell.to_numpy()
        Wt = CK[:, ki] * CO[:, oi]
        Wk = CK1[:, ki]
        if num is None:
            est = (Wt @ v) / np.maximum(Wt.sum(1), 1e-12)
            estk = (Wk @ v) / np.maximum(Wk.sum(1), 1e-12)
            pt = v.mean()
        else:
            cn, _, _ = cells(sub, num)
            u = cn.reindex(cell.index).to_numpy()
            est = (Wt @ u) / (Wt @ v)
            estk = (Wk @ u) / (Wk @ v)
            pt = u.mean() / v.mean()
        ok = Wt.sum(1) > 0
        return (pt, *np.quantile(est[ok], [0.025, 0.975]), len(cell), *np.quantile(estk, [0.025, 0.975]))

    def boot_cluster(sub, col):
        if clus is None:
            return np.nan, np.nan
        cell, ki, oi = cells(sub, col)
        v = cell.to_numpy()
        Wc = CKc[:, clus[ki]] * COc[:, clus[oi]]
        ok = Wc.sum(1) > 0
        est = (Wc[ok] @ v) / Wc[ok].sum(1)
        return tuple(np.quantile(est, [0.025, 0.975]))

    def f(t, d=3):
        return "—" if t[3] == 0 else f"{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]"

    nr = df[~df.role.isin(["O", "Op"])].drop_duplicates(["score", "kb", "ood", "draw", "slot"]).copy()
    ko, ok_ = nr.kb.map(sidx).to_numpy(), nr.ood.map(sidx).to_numpy()
    sl = nr.slot.map(sidx).to_numpy()
    nr["x_rank_O"], nr["x_rank_KB"] = rankm[ok_, sl], rankm[ko, sl]
    nr["x_tag_O"], nr["x_tag_KB"] = D[ok_, sl], D[ko, sl]
    nr["x_emb_O"], nr["x_emb_KB"] = D2[ok_, sl], D2[ko, sl]

    def regress(g, var):
        x = g[[f"x_{var}_O", f"x_{var}_KB"]].to_numpy(float)
        x = x / x.std(0)
        dmy = pd.get_dummies(g.slot).to_numpy(float)
        X = np.column_stack([x, dmy[:, 1:]])
        y = g.ddp.to_numpy()
        grp = g.groupby(["kb", "ood", "draw"]).ngroup().to_numpy()
        cnt = np.bincount(grp)

        def dm(A):
            A = A.reshape(len(A), -1)
            mu = np.zeros((grp.max() + 1, A.shape[1]))
            np.add.at(mu, grp, A)
            return A - (mu / cnt[:, None])[grp]
        Xd, yd = dm(X), dm(y)[:, 0]
        keep = np.abs(Xd).sum(0) > 0
        Xd = Xd[:, keep]
        cid = g.groupby(["kb", "ood"]).ngroup().to_numpy()
        nc = cid.max() + 1
        XtX = np.zeros((nc, Xd.shape[1], Xd.shape[1]))
        Xty = np.zeros((nc, Xd.shape[1]))
        order = np.argsort(cid, kind="stable")
        bounds = np.searchsorted(cid[order], np.arange(nc + 1))
        for c in range(nc):
            ix = order[bounds[c]:bounds[c + 1]]
            XtX[c] = Xd[ix].T @ Xd[ix]
            Xty[c] = Xd[ix].T @ yd[ix]
        first = g.groupby(cid)[["kb", "ood"]].first()
        ki, oi = first.kb.map(sidx).to_numpy(), first.ood.map(sidx).to_numpy()
        coef = np.linalg.lstsq(XtX.sum(0), Xty.sum(0), rcond=None)[0][:2]
        Wt = CK[:, ki] * CO[:, oi]
        A = np.tensordot(Wt, XtX, axes=(1, 0))
        bvec = Wt @ Xty
        bs = np.array([np.linalg.lstsq(A[b], bvec[b], rcond=None)[0][:2] for b in range(B)])
        lo, hi = np.quantile(bs, 0.025, axis=0), np.quantile(bs, 0.975, axis=0)
        return [(coef[j], lo[j], hi[j], 1) for j in range(2)]

    scores = list(dict.fromkeys(df.score))
    est_rows, reg_rows, strata_rows = [], [], []
    L = ["# A1 — RQ3 intervention (StackExchange, detector " + r + ")", ""]
    L.append(f"- pairs {len(pairs)} (dropped for a too small fixed-member pool: {int(pairs.dropped.sum())}); KB sites {pairs.kb.nunique()}, R = {cfg['comparison']['R']}, m = {cfg['comparison']['m']}")
    a = df[(df.score == scores[0]) & (df.role == "O")].drop_duplicates(["kb", "ood"]).merge(pairs[["kb", "ood", "tier"]], on=["kb", "ood"])
    L.append("- A_abs (top-10 mean) by tier: " + "; ".join(f"{t} median {g.auroc_abs.median():.3f} (≤ {cfg['metrics']['ceiling']}: {(g.auroc_abs <= cfg['metrics']['ceiling']).mean():.2f})" for t, g in a.groupby("tier")))
    L.append("- values: mean over (KB, O) cells of the draw mean; [ ] two-way (KB x O) bootstrap 95%, B = " + str(B) + ". d′ differences are to the score's own absolute baseline."
             + (f" Set mode: the bootstrap resamples the {len(cfg['pairs']['kb_set'])} kb_set sites (KB and O independently)." if set_mode else ""))
    L += ["", "## Main estimands (Δd′)", "",
          "| score | (ii)−(iii) | (iii)−abs | (i)−(iii) | recovery | N1−Zm (KB-matched) | id_shift | ood_shift | γ_O rank | γ_KB rank |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in scores:
        g = C[C.score == s]
        e = {n: boot(g, f"{n}_ddp") for n in ("main", "off", "leak", "kcost", "matched")}
        rec = boot(g, "leak_ddp", num="main_ddp")
        rg = regress(nr[nr.score == s], "rank")
        for n, t in {**e, "recovery": rec}.items():
            lc, hc = boot_cluster(g, f"{n}_ddp") if n != "recovery" else (np.nan, np.nan)
            est_rows.append({"score": s, "estimand": n, "metric": "ddp", "point": t[0], "lo": t[1], "hi": t[2], "n_cells": t[3], "lo_kbonly": t[4], "hi_kbonly": t[5],
                             "lo_cluster": lc, "hi_cluster": hc})
        tz = boot(g[g.z_relaxed == 0], "main_ddp")
        est_rows.append({"score": s, "estimand": "main_z_strict", "metric": "ddp", "point": tz[0], "lo": tz[1], "hi": tz[2],
                         "n_cells": tz[3], "lo_kbonly": tz[4], "hi_kbonly": tz[5]})
        for mt in ("dA", "G", "dFPR", "dOP"):
            for n in ("main", "off", "leak", "kcost", "matched"):
                t = boot(g, f"{n}_{mt}")
                lc, hc = boot_cluster(g, f"{n}_{mt}")
                est_rows.append({"score": s, "estimand": n, "metric": mt, "point": t[0], "lo": t[1], "hi": t[2], "n_cells": t[3], "lo_kbonly": t[4], "hi_kbonly": t[5],
                                 "lo_cluster": lc, "hi_cluster": hc})
        for var in ("rank", "tag", "emb"):
            rr = rg if var == "rank" else regress(nr[nr.score == s], var)
            for j, nm in enumerate(("gamma_O", "gamma_KB")):
                reg_rows.append({"score": s, "distance": var, "coef": nm, "point": rr[j][0], "lo": rr[j][1], "hi": rr[j][2]})
        L.append(f"| {s} | {f(e['main'])} | {f(e['off'])} | {f(e['leak'])} | {rec[0]:.2f} [{rec[1]:.2f}, {rec[2]:.2f}] | {f(e['matched'])} | "
                 f"{boot(g, 'id_shift')[0]:+.2f} | {boot(g, 'ood_shift')[0]:+.2f} | {f(rg[0])} | {f(rg[1])} |")
        if "leakp_ddp" in g:
            for mt in CMETRICS:
                for n in ("leakp", "opmo"):
                    t = boot(g, f"{n}_{mt}")
                    est_rows.append({"score": s, "estimand": n, "metric": mt, "point": t[0], "lo": t[1], "hi": t[2], "n_cells": t[3], "lo_kbonly": t[4], "hi_kbonly": t[5]})
            t = boot(g, "leakp_ddp", num="main_ddp")
            est_rows.append({"score": s, "estimand": "recoveryp", "metric": "ddp", "point": t[0], "lo": t[1], "hi": t[2], "n_cells": t[3], "lo_kbonly": t[4], "hi_kbonly": t[5]})
        t98 = boot(g[g.a_abs <= cfg["metrics"]["ceiling"]], "main_ddp")
        est_rows.append({"score": s, "estimand": "main_Aabs_le_ceiling", "metric": "ddp", "point": t98[0], "lo": t98[1], "hi": t98[2],
                         "n_cells": t98[3], "lo_kbonly": t98[4], "hi_kbonly": t98[5]})
    E = pd.DataFrame(est_rows)
    R_ = pd.DataFrame(reg_rows)
    E.to_csv(res / "tables" / "estimands.csv", index=False)
    R_.to_csv(res / "tables" / "regression.csv", index=False)

    def etab(metric, title):
        ns = ("main", "off", "leak", "matched", "kcost")
        rows = ["", f"## {title}", "", "| score | (ii)−(iii) | (iii)−abs | (i)−(iii) | N1−Zm | K−(iii) (descriptive) |", "|---|---|---|---|---|---|"]
        for s in scores:
            t = {n: E[(E.score == s) & (E.estimand == n) & (E.metric == metric)].iloc[0] for n in ns}
            rows.append(f"| {s} | " + " | ".join(f"{t[n].point:+.3f} [{t[n].lo:+.3f}, {t[n].hi:+.3f}] ({int(t[n].n_cells)})" for n in ns) + " |")
        return rows
    L += etab("ddp", "Δd′: all contrasts (K−(iii) is descriptive: K is also closer to O than Z; KB cost = γ_KB)")
    L += ["", "## Ceiling diagnostics (Δd′ uses AUROC clipped at 1 − " + str(clip) + ")", "",
          "| score | share of draws at the clip (abs, N or Z) | share of cells A_abs > 0.995 | (ii)−(iii) on cells A_abs ≤ ceiling |", "|---|---|---|---|"]
    for s in scores:
        g = C[C.score == s]
        t = E[(E.score == s) & (E.estimand == "main_Aabs_le_ceiling")].iloc[0]
        cl = g.groupby(["kb", "ood"]).a_abs.first()
        L.append(f"| {s} | {g.at_clip.mean():.3f} | {(cl > 0.995).mean():.3f} | {t.point:+.3f} [{t.lo:+.3f}, {t.hi:+.3f}] ({int(t.n_cells)}) |")
    L += etab("dA", "ΔAUROC")
    L += etab("dOP", "Operating point: Δ OOD acceptance at the threshold accepting 95% of KB calibration queries")
    op = df[df.role.isin(["N1", "Z1"])].groupby(["score", "role"]).op_id_acc.mean().unstack()
    opa = df.drop_duplicates(["score", "kb", "ood"]).groupby("score").op_id_acc_abs.mean()
    L += ["", "Realised ID acceptance on test at the calibrated threshold (target 0.95): " + "; ".join(
        f"{s}: abs {opa[s]:.3f}, N1 {op.loc[s, 'N1']:.3f}, Z1 {op.loc[s, 'Z1']:.3f}" for s in scores[:6]) + " (all scores: tables/rows.parquet)"]
    L += etab("dFPR", "ΔFPR95 (oracle)")
    L += etab("G", "Secondary: G (A_abs ≤ ceiling; n = cells)")
    L += ["", "## Dose response (Δd′ per 1 SD; slot-site and (KB, O, draw) fixed effects)", "",
          "| score | γ_O rank | γ_KB rank | γ_O tag | γ_KB tag | γ_O 2nd-enc | γ_KB 2nd-enc |", "|---|---|---|---|---|---|---|"]
    for s in scores:
        q = R_[R_.score == s].set_index(["distance", "coef"])
        L.append(f"| {s} | " + " | ".join(f"{q.loc[(v, c)].point:+.3f} [{q.loc[(v, c)].lo:+.3f}, {q.loc[(v, c)].hi:+.3f}]"
                                          for v in ("rank", "tag", "emb") for c in ("gamma_O", "gamma_KB")) + " |")
    L += ["", "## Strata of (ii)−(iii) Δd′ (n = cells)", ""]
    strata = [("tier near", C.tier == "near"), ("tier mid", C.tier == "mid"), ("tier far", C.tier == "far"),
              ("N1 in O's neighbour band", C.n1_in_band), ("N1 not in band", ~C.n1_in_band),
              ("O technology", C.ood_cat == "technology"), ("O culture/recreation", C.ood_cat == "culturerecreation"),
              ("O life/arts", C.ood_cat == "lifearts"), ("O science", C.ood_cat == "science"), ("O other (prof./business)", C.ood_cat == "other"),
              ("KB–O same cat & N1 same cat as O", C.kb_o_same_cat & C.n1_same_cat_o),
              ("KB–O same cat & N1 other cat", C.kb_o_same_cat & ~C.n1_same_cat_o),
              ("KB–O diff cat & N1 same cat as O", ~C.kb_o_same_cat & C.n1_same_cat_o),
              ("KB–O diff cat & N1 other cat", ~C.kb_o_same_cat & ~C.n1_same_cat_o)]
    L += ["| score | " + " | ".join(n for n, _ in strata) + " |", "|---" * (len(strata) + 1) + "|"]
    for s in scores:
        g = C[C.score == s]
        cellsv = []
        for n, msk in strata:
            t = boot(g[msk[C.score == s]], "main_ddp")
            strata_rows.append({"score": s, "stratum": n, "point": t[0], "lo": t[1], "hi": t[2], "n_cells": t[3]})
            cellsv.append(f"{f(t)} ({t[3]})")
        L.append(f"| {s} | " + " | ".join(cellsv) + " |")
    L += ["", "## Post hoc: N1 − Zm (KB-proximity-matched) Δd′ by KB–O tier (added after the A1 results were seen)", "",
          "| score | tier near | tier mid | tier far |", "|---|---|---|---|"]
    dsl = dr[dr.role.isin(["N1", "Zm", "Z1", "Z2"])].pivot_table(index=["kb", "ood", "draw"], columns="role", values="slot", aggfunc="first").reset_index()
    dsl["tier"] = [pinfo.loc[(a, b), "tier"] for a, b in zip(dsl.kb, dsl.ood)]
    dsl["gap"] = [rankm[sidx[k_], sidx[z]] - rankm[sidx[k_], sidx[n]] for k_, z, n in zip(dsl.kb, dsl.Zm, dsl.N1)]
    bal = dsl.groupby("tier").gap.agg(lambda x: f"median {np.median(x):.0f}, IQR {np.quantile(x, .25):.0f}–{np.quantile(x, .75):.0f}, |gap| ≤ 3: {np.mean(np.abs(x) <= 3):.2f}")
    L.append("| matching balance: KB-rank(Zm) − KB-rank(N1) | " + " | ".join(bal.get(t_, "—") for t_ in ("near", "mid", "far")) + " |")
    for s in scores:
        g = C[C.score == s]
        cellsv = []
        for tn in ("near", "mid", "far"):
            t = boot(g[g.tier == tn], "matched_ddp")
            strata_rows.append({"score": s, "stratum": f"matched: tier {tn}", "point": t[0], "lo": t[1], "hi": t[2], "n_cells": t[3]})
            cellsv.append(f"{f(t)} ({t[3]})")
        L.append(f"| {s} | " + " | ".join(cellsv) + " |")
    pd.DataFrame(strata_rows).to_csv(res / "tables" / "strata.csv", index=False)
    d1 = C[C.score == scores[0]]
    dd0 = dr.drop_duplicates(["kb", "ood", "draw"])
    zr = (dd0.z_relaxed.astype(int) > 0).mean()
    zr2 = (dd0.z_relaxed.astype(int) == 2).mean()
    zmr = dd0.zm_relaxed.mean() if "zm_relaxed" in dd0 else 0.0
    T = np.load(edir / "topk.npy", mmap_mode="r")
    qi = pd.read_parquet(edir / "q_index.parquet")
    rows_of = {kk: g.index.to_numpy() for kk, g in qi.groupby(["site", "role"])}
    dd = dr[dr.role.isin(["N1", "Z1", "Zm"])].drop_duplicates(["kb", "ood", "draw", "role"])
    top1 = {}
    for (o, sl), g in dd.groupby(["ood", "slot"]):
        top1[(o, sl)] = np.asarray(T[rows_of[(o, "test")], sidx[sl], 0])
    dup = {ro: [np.mean(top1[(o, sl)] >= th) for o, sl in zip(g.ood, g.slot)] for ro, g in dd.groupby("role") for th in [0.95]}
    L += ["", "- near-duplicate check (share of O test queries whose top-1 cosine in the slot's bank is ≥ 0.95; mean over draws): "
          + ", ".join(f"{ro} {np.mean(v):.4f}" for ro, v in dup.items())]
    L += ["", f"- N1 inside O's neighbour band: {d1.n1_in_band.mean():.2f} of draws; N1 same category as O: {d1.n1_same_cat_o.mean():.2f}; "
          f"Z drawn from a relaxed pool: {zr:.3f} of draws (level 2, farthest by rank: {zr2:.3f}); Zm outside the far band: {zmr:.3f}; "
          f"pairs with fewer than R draws: {int((pairs.n_draws < cfg['comparison']['R']).sum())}",
          "- KB-only bootstrap intervals for the main estimands: tables/estimands.csv (lo_kbonly, hi_kbonly)."]
    (res / "A1_summary.md").write_text("\n".join(L) + "\n")
    meta = run_metadata(cfg, "A1.03_analyze")
    meta["inputs"] = fingerprint([res / "tables" / "rows.parquet", res / "tables" / "pairs.csv", res / "tables" / "draws.csv",
                                  edir / "topk.npy", edir / f"centroids_{r2}.npy", p(cfg, "a0_processed", cfg.get("distance_file", "tag_js.csv")),
                                  p(cfg, "a0_processed", "bands.json"), p(cfg, "a0_processed", "sites.json")])
    meta["outputs"] = fingerprint([res / "A1_summary.md", res / "tables" / "estimands.csv", res / "tables" / "regression.csv",
                                   res / "tables" / "strata.csv"])
    save_json(meta, res / "meta_03.json")
    log.info("summary written")


if __name__ == "__main__":
    main()
