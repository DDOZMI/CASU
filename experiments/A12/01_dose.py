from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import PROJECT_ROOT, fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json, set_determinism, stable_seed

set_determinism()
import torch

DEV = "cuda"


def unit_t(X):
    X = torch.as_tensor(np.asarray(X, np.float32), device=DEV)
    return X / X.norm(dim=1, keepdim=True)


def factor(Lsel, nt, U):
    X = Lsel[:, U]
    cal = X[nt:]
    mu, sd = cal.mean(0), cal.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    _, sv, vt = np.linalg.svd(Z[nt:] - Z[nt:].mean(0), full_matrices=False)
    w = vt[0] * (1 if vt[0].sum() >= 0 else -1)
    wk = float(np.corrcoef(w, cal.mean(0))[0, 1]) if len(U) > 2 else np.nan
    return Z @ w, float(sv[0] ** 2 / (sv ** 2).sum()), wk


def corr_stats(s, c, g):
    R = np.corrcoef(np.vstack([s, c, g]))
    rsc, rsg, rcg = R[0, 1], R[0, 2], R[1, 2]
    part = (rsc - rsg * rcg) / np.sqrt((1 - rsg ** 2) * (1 - rcg ** 2))
    return {"rho_sc": rsc, "rho_sg": rsg, "rho_cg": rcg, "partial": part, "sd_ratio": c.std() / s.std(), "mean_c": c.mean()}


def arm_after_removal(sample, stream, z, n_rm):
    if n_rm == 0:
        return sample
    order = np.argsort(-z[sample], kind="stable")
    removed, kept = sample[order[:n_rm]], sample[order[n_rm:]]
    cut = z[removed].min()
    top = stream[z[stream] < cut][:n_rm]
    assert len(top) == n_rm, "pool too small to top up"
    return np.concatenate([kept, top])


def run_one(rn, run, cfg, log, smoke_kbs=0):
    k, dc = cfg["k"], cfg["dose"]
    n, R = dc["n_sample"], dc["R"]
    edir = PROJECT_ROOT / run["emb"]
    sites = load_json(edir / "sites.json")
    sidx = {s: i for i, s in enumerate(sites)}
    T = np.asarray(np.load(edir / "topk.npy", mmap_mode="r"))
    Lm = T.astype(np.float64).mean(2)
    qi = pd.read_parquet(edir / "q_index.parquet")
    rows_of = {kk: g.index.to_numpy() for kk, g in qi.groupby(["site", "role"])}
    r = run["retriever"]
    a0 = p(cfg, "a0_emb", r)
    MS = unit_t(np.load(p(cfg, "p9_emb", r, "pool.npy")))
    nP = MS.shape[0]
    MQ = unit_t(np.load(p(cfg, "emb", r, "msq.npy")))
    em = load_json(p(cfg, "emb", r, "meta.json"))
    assert em["n_msq"] == MQ.shape[0] == 20000 and not em["upstream_fingerprint"]["smoke"], em
    hub = (MS @ MQ.mean(0)).double().cpu().numpy()
    D = pd.read_csv(p(cfg, "a0_processed", "tag_js.csv"), index_col=0).loc[sites, sites].to_numpy()
    q_far = load_json(p(cfg, "a0_processed", "bands.json"))["q_far_value"]
    kbs = sites[:smoke_kbs] if smoke_kbs else sites
    recs = []
    t0 = time.time()
    for ki, kb in enumerate(kbs):
        i_t, i_c = rows_of[(kb, "test")], rows_of[(kb, "cal")]
        sel = np.r_[i_t, i_c]
        nt = len(i_t)
        Lsel, Tsel = Lm[sel], T[sel]
        s = Lsel[:nt, sidx[kb]]
        Qt = unit_t(np.load(a0 / f"{kb}__test.npy"))
        Qc = unit_t(np.load(a0 / f"{kb}__cal.npy"))
        assert Qt.shape[0] == nt and Qc.shape[0] == len(i_c)
        Bk = unit_t(np.load(a0 / f"{kb}__bank.npy"))
        s_chk = (Qt @ Bk.T).topk(k, dim=1).values.double().mean(1).cpu().numpy()
        assert np.abs(s_chk - s).max() < 1e-5, (kb, np.abs(s_chk - s).max())
        near = (Qc @ MS.T).max(0).values.double().cpu().numpy()
        St = Qt @ MS.T
        U_ms = [j for j in range(len(sites)) if j != sidx[kb]]
        g_ms, sh_ms, wk_ms = factor(Lsel, nt, U_ms)
        g_ms = g_ms[:nt]
        base = {"run": rn, "kb": kb}
        others = [x for x in sites if x != kb]
        far = sorted(x for x in others if D[sidx[kb], sidx[x]] > q_far)
        far_relaxed = len(far) < dc["se_m"]
        if far_relaxed:
            far = sorted(sorted(others, key=lambda x: (-D[sidx[kb], sidx[x]], x))[: dc["se_m"]])
        for d in range(R):
            perm = np.random.default_rng(stable_seed(cfg["seed"], "A12_ms", kb, d)).permutation(nP)
            sample, stream = perm[:n], perm[n:]
            gq = np.random.default_rng(stable_seed(cfg["seed"], "A12_gen", kb, d)).choice(MQ.shape[0], dc["n_gen"], replace=False)
            zs = {"near": near, "gen": (MQ[torch.as_tensor(gq, device=DEV)] @ MS.T).max(0).values.double().cpu().numpy(),
                  "hub": hub, "rand": np.random.default_rng(stable_seed(cfg["seed"], "A12_rand", kb, d)).random(nP)}
            o_far = far[int(np.random.default_rng(stable_seed(cfg["seed"], "A12_nother", kb, d)).integers(len(far)))]
            Qo_ = unit_t(np.load(a0 / f"{o_far}__cal.npy"))
            zs["near_other"] = (Qo_ @ MS.T).max(0).values.double().cpu().numpy()
            for rule in dc["rules"]:
                for f in dc["fracs"]:
                    n_rm = int(round(f * n))
                    arm = arm_after_removal(sample, stream, zs[rule], n_rm)
                    assert len(arm) == n and len(np.unique(arm)) == n
                    c = St[:, torch.as_tensor(arm, device=DEV)].topk(k, dim=1).values.double().mean(1).cpu().numpy()
                    recs.append({**base, "draw": d, "set": "ms", "rule": rule, "frac": f, "n_removed": n_rm, "pc1_share": sh_ms, "w_kbnear_corr": wk_ms,
                                 **corr_stats(s, c, g_ms), "arm_near_mean": near[arm].mean(), "arm_hub_mean": hub[arm].mean(),
                                 "arm_near_q99": float(np.quantile(near[arm], 0.99)), "near_other_site": o_far if rule == "near_other" else ""})
            for nm, pool, key in (("se_far", far, "A12_far"), ("se_any", sorted(others), "A12_any")):
                C = list(np.random.default_rng(stable_seed(cfg["seed"], key, kb, d)).choice(pool, dc["se_m"], replace=False))
                Ci = [sidx[x] for x in C]
                top = -np.sort(-Tsel[:nt][:, Ci, :].reshape(nt, -1), axis=1)[:, :k]
                c = top.astype(np.float64).mean(1)
                g, sh, wk = factor(Lsel, nt, [j for j in U_ms if j not in Ci])
                recs.append({**base, "draw": d, "set": nm, "rule": "none", "frac": 0.0, "n_removed": 0, "pc1_share": sh, "w_kbnear_corr": wk,
                             **corr_stats(s, c, g[:nt]), "far_relaxed": far_relaxed if nm == "se_far" else False,
                             "n_far_pool": len(far), "members": "+".join(C)})
        if ki % 10 == 0:
            log.info(f"{rn}: KB {ki + 1}/{len(kbs)} ({(time.time() - t0) / 60:.1f} min)")
    D_ = pd.DataFrame(recs)
    f0 = D_[(D_.set == "ms") & (D_.frac == 0)].groupby(["kb", "draw"]).rho_sc.agg(["min", "max"])
    assert (f0["max"] - f0["min"]).abs().max() < 1e-12, "f = 0 must be the same arm for every rule"
    return D_


def kb_boot(v, B, seed):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), size=(B, len(v)))].mean(1)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return v.mean(), lo, hi


def summarise(all_rows, cfg, res, sfx):
    dc, B = cfg["dose"], cfg["bootstrap_B"]
    L = ["# A12 — where rho_ID comes from: dose-response of a general background's documents, and KB-far reference sets", "",
         f"- Design (fixed before the run): {dc['_source']}", f"- g: {cfg['factor']}", f"- Measures: {cfg['measures']}",
         "- Reading rule (JOURNAL 14:52, revised after the pre-run review, before the run): the near curve alone cannot separate 'KB-adjacent content' from a selection effect of removing documents close to queries. The deciding contrast is within MS MARCO at a matched dose (Δ mean c): near (the KB's own calibration queries) vs near_other (the same operation with a KB-far StackExchange site's calibration queries: same query type, unrelated topic) vs rand. gen / hub barely move c (MS MARCO queries do not reach SE queries' top-10) and serve only as weak placebos. se_far / se_any are a cross-corpus reference (g is built from SE banks, so SE comparison terms load on it more than MS documents); se_far partials are negative without any selection (A10), so a negative partial alone does not indicate selection.",
         "- Fractions are nested for the removed documents only; top-up documents change between fractions (a lower cut drops earlier top-ups). Common random numbers hold.",
         "- Values: KB means of the per-KB draw means [KB bootstrap 95%]. f = removal fraction of the 12,000-document sample. ρ_sc = corr(s_KB, c) on KB test queries.", ""]
    for rn, D in all_rows.items():
        kbm = D.groupby(["set", "rule", "frac", "kb"]).mean(numeric_only=True).reset_index()
        nkb = kbm.kb.nunique()
        L += [f"## {rn} ({nkb} KBs, {D.draw.nunique()} draws)", ""]
        ref = {}
        L += ["Reference sets (StackExchange, 4 banks pooled):", "", "| set | ρ_sc | partial ρ(s,c\\|g) | ρ(s,g) | ρ(c,g) | sd(c)/sd(s) | PC1 share | corr(w, KB-nearness) |", "|---|---|---|---|---|---|---|---|"]
        for nm in ("se_far", "se_any"):
            sub = kbm[kbm.set == nm]
            m_, lo, hi = kb_boot(sub.rho_sc, B, stable_seed(cfg["seed"], "A12_boot", rn, nm))
            ref[nm] = m_
            pm, plo, phi = kb_boot(sub.partial, B, stable_seed(cfg["seed"], "A12_boot_p", rn, nm))
            L.append(f"| {nm} | {m_:+.3f} [{lo:+.3f}, {hi:+.3f}] | {pm:+.3f} [{plo:+.3f}, {phi:+.3f}] | {sub.rho_sg.mean():+.3f} | {sub.rho_cg.mean():+.3f} | {sub.sd_ratio.mean():.3f} | {sub.pc1_share.mean():.3f} | {sub.w_kbnear_corr.mean():+.3f} |")
        relax = D[D.set == "se_far"].groupby("kb").far_relaxed.max()
        L += ["", f"se_far relaxed (fewer than 4 far-band sites; 4 farthest used) for {int(relax.sum())}/{len(relax)} KBs.", ""]
        for q, lab in (("rho_sc", "ρ_sc = corr(s_KB, c)"), ("partial", "partial ρ(s,c|g)"), ("rho_cg", "ρ(c,g)"), ("sd_ratio", "sd(c)/sd(s)"),
                       ("arm_near_mean", "arm mean near score"), ("arm_hub_mean", "arm mean hub score")):
            L += [f"### {lab} by rule and removal fraction", "", "| f | " + " | ".join(dc["rules"]) + " |", "|---|" + "---|" * len(dc["rules"])]
            for f in dc["fracs"]:
                cells = []
                for rule in dc["rules"]:
                    sub = kbm[(kbm.set == "ms") & (kbm.rule == rule) & (kbm.frac == f)]
                    if q in ("rho_sc", "partial"):
                        m_, lo, hi = kb_boot(sub[q], B, stable_seed(cfg["seed"], "A12_boot", rn, q, rule, f))
                        cells.append(f"{m_:+.3f} [{lo:+.3f}, {hi:+.3f}]")
                    else:
                        cells.append(f"{sub[q].mean():+.4f}")
                L.append(f"| {f:g} | " + " | ".join(cells) + " |")
            L.append("")
        base0 = kbm[(kbm.set == "ms") & (kbm.frac == 0) & (kbm.rule == "near")].set_index("kb").rho_sc
        L += ["### Paired change Δρ_sc(f) − ρ_sc(0) per KB [KB bootstrap 95%]", "", "| f | " + " | ".join(dc["rules"]) + " |", "|---|" + "---|" * len(dc["rules"])]
        for f in dc["fracs"][1:]:
            cells = []
            for rule in dc["rules"]:
                sub = kbm[(kbm.set == "ms") & (kbm.rule == rule) & (kbm.frac == f)].set_index("kb").rho_sc
                m_, lo, hi = kb_boot((sub - base0.reindex(sub.index)).to_numpy(), B, stable_seed(cfg["seed"], "A12_boot_d", rn, rule, f))
                cells.append(f"{m_:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            L.append(f"| {f:g} | " + " | ".join(cells) + " |")
        c0 = kbm[(kbm.set == "ms") & (kbm.frac == 0) & (kbm.rule == "near")].set_index("kb").mean_c
        L += ["", "### Common dose axis: Δ mean c (vs f = 0) | ρ_sc, KB means", "", "| f | " + " | ".join(dc["rules"]) + " |", "|---|" + "---|" * len(dc["rules"])]
        curves = {}
        for rule in dc["rules"]:
            xs, ys = [], []
            for f in dc["fracs"]:
                sub = kbm[(kbm.set == "ms") & (kbm.rule == rule) & (kbm.frac == f)].set_index("kb")
                xs.append(float((sub.mean_c - c0.reindex(sub.index)).mean()))
                ys.append(float(sub.rho_sc.mean()))
            curves[rule] = (np.array(xs), np.array(ys))
        for i, f in enumerate(dc["fracs"]):
            L.append(f"| {f:g} | " + " | ".join(f"{curves[r_][0][i]:+.4f} \\| {curves[r_][1][i]:+.3f}" for r_ in dc["rules"]) + " |")
        L += ["", "ρ_sc of each rule interpolated at the near rule's Δ mean c (blank = outside that rule's dose range):", "", "| f (near) | Δ mean c | " + " | ".join(dc["rules"]) + " |", "|---|---|" + "---|" * len(dc["rules"])]
        for i, f in enumerate(dc["fracs"]):
            x0 = curves["near"][0][i]
            cells = []
            for r_ in dc["rules"]:
                xs, ys = curves[r_]
                o_ = np.argsort(xs)
                cells.append(f"{np.interp(x0, xs[o_], ys[o_]):+.3f}" if xs.min() - 1e-9 <= x0 <= xs.max() + 1e-9 else "")
            L.append(f"| {f:g} | {x0:+.4f} | " + " | ".join(cells) + " |")
        far_kb = kbm[kbm.set == "se_far"].set_index("kb").rho_sc
        below = []
        for f in dc["fracs"]:
            sub = kbm[(kbm.set == "ms") & (kbm.rule == "near") & (kbm.frac == f)].set_index("kb").rho_sc
            below.append(f"{f:g}: {(sub < far_kb.reindex(sub.index)).mean():.2f}")
        L += ["", "Share of KBs whose near-rule ρ_sc is below their own se_far ρ_sc, by f: " + ", ".join(below), ""]
    (res / f"A12_summary{sfx}.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--smoke-kbs", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    assert not args.smoke_kbs or args.out, "--smoke-kbs requires --out"
    cfg = load_config(args.config)
    res = Path(args.out) if args.out else p(cfg, "results")
    (res / "tables").mkdir(parents=True, exist_ok=True)
    log = get_logger("A12.01", res / "logs" / "01_dose.log")
    runs = {k_: v for k_, v in cfg["runs"].items() if not args.runs or k_ in args.runs}
    all_rows = {}
    for rn, run in runs.items():
        fpath = res / "tables" / f"rows_{rn}.parquet"
        if not args.summary_only:
            t0 = time.time()
            D = run_one(rn, run, cfg, log, args.smoke_kbs)
            D.to_parquet(fpath, index=False)
            log.info(f"{rn}: {len(D)} rows in {(time.time() - t0) / 60:.1f} min")
        all_rows[rn] = pd.read_parquet(fpath)
    sfx = "" if not args.runs else "_" + "_".join(runs)
    summarise(all_rows, cfg, res, sfx)
    if not args.summary_only:
        meta = run_metadata(cfg, "A12.01_dose")
        ins = [PROJECT_ROOT / r["emb"] / f for r in runs.values() for f in ("topk.npy", "q_index.parquet", "sites.json")]
        ins += [p(cfg, "p9_emb", r, "pool.npy") for r in sorted({v["retriever"] for v in runs.values()})]
        ins += [p(cfg, "emb", r, "msq.npy") for r in sorted({v["retriever"] for v in runs.values()})]
        ins += [p(cfg, "a0_processed", "tag_js.csv"), p(cfg, "a0_processed", "bands.json")]
        ins += sorted({p(cfg, "a0_emb", r["retriever"], f"{s_}__{ro}.npy") for r in runs.values()
                       for s_ in load_json(PROJECT_ROOT / r["emb"] / "sites.json") for ro in ("test", "cal", "bank")})
        meta["inputs"] = fingerprint(ins)
        meta["smoke_kbs"] = args.smoke_kbs
        save_json(meta, res / f"meta_01{sfx}.json")


if __name__ == "__main__":
    main()
