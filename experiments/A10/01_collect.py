from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import PROJECT_ROOT, fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json, set_determinism, stable_seed

set_determinism()
import torch

DEV = "cuda"
ROLES = ["O", "N1", "N2", "M", "Z1", "Z2", "Zm", "K", "NK"]


def unit_t(X):
    X = torch.as_tensor(np.asarray(X, np.float32), device=DEV)
    return X / X.norm(dim=1, keepdim=True)


def agg(A, t, m):
    return A.mean(1) if t == "inf" else float(t) * (logsumexp(A / float(t), axis=1) - np.log(m))


def auroc(S, n_id, n_ood):
    X = -S[: n_id + n_ood]
    r = rankdata(X)
    return float((r[n_id:].sum() - n_ood * (n_ood + 1) / 2) / (n_ood * n_id))


def norm_par(x):
    s = str(x)
    return s[:-2] if s.endswith(".0") else s


class Factor:

    def __init__(self, L_sel, n_test):
        self.L, self.nt, self.cache = L_sel, n_test, {}

    def __call__(self, U):
        key = tuple(U)
        if key not in self.cache:
            X = self.L[:, U]
            cal = X[self.nt:]
            mu, sd = cal.mean(0), cal.std(0)
            sd[sd == 0] = 1.0
            Z = (X - mu) / sd
            _, sv, vt = np.linalg.svd(Z[self.nt:] - Z[self.nt:].mean(0), full_matrices=False)
            w = vt[0] * (1 if vt[0].sum() >= 0 else -1)
            wk = float(np.corrcoef(w, cal.mean(0))[0, 1]) if len(U) > 2 else np.nan
            self.cache[key] = (Z @ w, float(sv[0] ** 2 / (sv ** 2).sum()), wk)
        return self.cache[key]


def class_stats(s, c, g, ll, n_id, n_ood):
    out = {}
    s, c, g, ll = (np.asarray(x, np.float64) for x in (s, c, g, ll))
    for cl, sl in (("id", slice(0, n_id)), ("ood", slice(n_id, n_id + n_ood))):
        S, C, G, L = s[sl], c[sl], g[sl], ll[sl]
        cv = np.cov(np.vstack([S, C, G, L]), bias=True)
        out.update({f"{cl}_n": len(S), f"{cl}_ms": S.mean(), f"{cl}_mc": C.mean(), f"{cl}_vs": cv[0, 0], f"{cl}_vc": cv[1, 1],
                    f"{cl}_csc": cv[0, 1], f"{cl}_vg": cv[2, 2], f"{cl}_csg": cv[0, 2], f"{cl}_ccg": cv[1, 2], f"{cl}_cgl": cv[2, 3],
                    f"{cl}_vl": cv[3, 3], f"{cl}_csl": cv[0, 3]})
    return out


def group_diag(Tg, n_id, n_ood):
    Tg = np.asarray(Tg, np.float64)
    gm = Tg.mean(2)
    jmax = gm.argmax(1)
    kth_best = Tg[np.arange(len(Tg)), jmax, -1]
    first = Tg[:, :, 0].copy()
    first[np.arange(len(Tg)), jmax] = -np.inf
    conc = kth_best >= first.max(1)
    pooled = -np.sort(-Tg.reshape(len(Tg), -1), axis=1)[:, : Tg.shape[2]].mean(1)
    cmax = gm.max(1)
    assert (pooled >= cmax - 1e-9).all() and (np.abs(pooled - cmax)[conc] < 1e-9).all()
    out = {}
    for cl, sl in (("id", slice(0, n_id)), ("ood", slice(n_id, n_id + n_ood))):
        out.update({f"{cl}_conc": conc[sl].mean(), f"{cl}_gap_pool_max": (pooled - cmax)[sl].mean(), f"{cl}_cmax": cmax[sl].mean()})
    return out, pooled


def run_one(rn, run, cfg, log, smoke_pairs=0):
    k, m, cap, R = cfg["k"], cfg["m"], cfg["cap"], cfg["R"]
    n = m * cap
    taus, alphas = cfg["taus"], cfg["alphas"]
    root = PROJECT_ROOT
    edir = root / run["emb"]
    sites = load_json(edir / "sites.json")
    sidx = {s: i for i, s in enumerate(sites)}
    T = np.asarray(np.load(edir / "topk.npy", mmap_mode="r"))
    Lm = T.astype(np.float64).mean(2)
    qi = pd.read_parquet(edir / "q_index.parquet")
    rows_of = {kk: g.index.to_numpy() for kk, g in qi.groupby(["site", "role"])}
    if run["queries"] == "a0":
        it = pd.read_parquet(root / run["items"] if run.get("items") else p(cfg, "a0_items"), columns=["site", "role", "pos", "title"])
        it = it[it.role.isin(["test", "cal"])].set_index(["site", "role", "pos"]).title
        titles = it.reindex(pd.MultiIndex.from_frame(qi[["site", "role", "pos"]])).to_numpy()
    else:
        P4 = pd.read_parquet(p(cfg, "p4_queries"))
        tq = dict(zip(zip(P4.source, P4.qid.astype(str)), P4.text))
        titles = np.array([tq[(s_, str(q_))] for s_, q_ in zip(qi.source, qi.qid)])
    assert not pd.isna(titles).any()
    LL = np.log1p(np.array([len(str(t).split()) for t in titles], float))
    tw = {}
    if run["queries"] == "cqa":
        tw = {c: v["twin"] for c, v in load_json(p(cfg, "a8_processed", "sites.json")).items() if "twin" in v}
    base = [s for s in sites if s not in tw]

    pairs = pd.read_csv(root / run["a1"] / "tables" / "pairs.csv")[["kb", "ood", "tier"]]
    if smoke_pairs:
        pairs = pairs.iloc[:: max(1, len(pairs) // smoke_pairs)].head(smoke_pairs)
    draws = pd.read_csv(root / run["a1"] / "tables" / "draws.csv")
    draws_of = {kk: g for kk, g in draws.groupby(["kb", "ood"])}
    R1 = pd.read_parquet(root / run["a1"] / "tables" / "rows.parquet", columns=["kb", "ood", "draw", "role", "family", "param", "auroc"])
    R1 = R1[R1.family == "S_tau"].assign(param=lambda x: x.param.map(norm_par)).set_index(["kb", "ood", "draw", "role", "param"]).auroc
    if not run.get("check", True):
        R1 = R1.iloc[:0]
    chk = {"a1": [0.0, 0]}
    emitted = {}
    have_a2 = "a2" in run
    if have_a2:
        sets = pd.read_csv(root / run["a2"] / "tables" / "sets.csv")
        sets_of = {kk: g for kk, g in sets.groupby(["kb", "ood"])}
        R2 = pd.read_parquet(root / run["a2"] / "tables" / "rows.parquet", columns=["kb", "ood", "draw", "m", "corpus", "partition", "family", "param", "auroc"])
        R2 = R2[R2.m == m].assign(param=lambda x: x.param.map(norm_par)).set_index(["kb", "ood", "draw", "corpus", "partition", "family", "param"]).auroc
        if not run.get("check", True):
            R2 = R2.iloc[:0]
        chk.update({"a2site": [0.0, 0], "a2pool": [0.0, 0], "a2rand": [0.0, 0], "msunf": [0.0, 0]})
    have_a4 = "a4" in run
    if have_a4:
        lev = pd.read_csv(root / run["a4"] / "tables" / "levels.csv")
        lev_of = {kk: g for kk, g in lev.groupby(["kb", "ood"])}
        T3 = np.asarray(np.load(root / run["a4_l3"], mmap_mode="r"))
        L3 = T3.astype(np.float64).mean(2)
        R4 = pd.read_parquet(root / run["a4"] / "tables" / "rows.parquet", columns=["kb", "ood", "draw", "level", "family", "param", "auroc"])
        R4 = R4[R4.family == "S_tau"].assign(param=lambda x: x.param.map(norm_par)).set_index(["kb", "ood", "draw", "level", "param"]).auroc
        chk["a4"] = [0.0, 0]
    doc = run["doc"]
    if doc:
        a0 = p(cfg, "a0_emb", run["retriever"])
        Q = unit_t(np.concatenate([np.load(a0 / f"{s}__{ro}.npy") for s in sites for ro in ("test", "cal")]))
        assert Q.shape[0] == len(qi)
        Bd = unit_t(np.concatenate([np.load(a0 / f"{s}__bank.npy") for s in sites]))
        MS = unit_t(np.load(p(cfg, "p9_emb", run["retriever"], "pool.npy")))
        nP = MS.shape[0]
        sample = lambda kb, o, d: np.random.default_rng(stable_seed(cfg["seed"], "A2_ms", kb, o, d)).permutation(nP)[: cfg["mmax_a2"] * cap][:n]
        allpairs = pd.read_csv(root / run["a1"] / "tables" / "pairs.csv")[["kb", "ood"]]
        dmax, crowd = {}, []
        for kb in allpairs.kb.unique():
            Qc = Q[torch.as_tensor(rows_of[(kb, "cal")], device=DEV)]
            dmax[kb] = (Qc @ MS.T).max(0).values.cpu().numpy()
            for o in allpairs[allpairs.kb == kb].ood:
                for d in range(R):
                    idx = torch.as_tensor(sample(kb, o, d), device=DEV)
                    crowd.append((Qc @ MS[idx].T).topk(k, dim=1).values.mean(1).cpu().numpy())
        t_common = float(np.quantile(np.concatenate(crowd), cfg["a9_p"]))
        log.info(f"{rn}: A9 common threshold recomputed {t_common:.6f}")
        A9c = pd.read_csv(root / run["a9"]).set_index(["kb", "ood"])

    ms_only = run.get("ms_only", False)
    if ms_only:
        assert not doc and not have_a2
        qd = root / run["q_emb"]
        Q = unit_t(np.concatenate([np.load(qd / f"{s_}__{ro}.npy") for s_ in sites for ro in ("test", "cal")]))
        assert Q.shape[0] == len(qi)
        MS = unit_t(np.load(p(cfg, "p9_emb", run["retriever"], "pool.npy")))
        nP = MS.shape[0]
        sample = lambda kb, o, d: np.random.default_rng(stable_seed(cfg["seed"], "A2_ms", kb, o, d)).permutation(nP)[: cfg["mmax_a2"] * cap][:n]

    recs, a9chk = [], []
    t0 = time.time()
    for pi_, (kb, o, tier) in enumerate(zip(pairs.kb, pairs.ood, pairs.tier)):
        i_id, i_ood, i_cal = rows_of[(kb, "test")], rows_of[(o, "test")], rows_of[(kb, "cal")]
        sel = np.r_[i_id, i_ood, i_cal]
        n_id, n_ood = len(i_id), len(i_ood)
        Lsel, Tsel, ll = Lm[sel], T[sel], LL[sel]
        s = Lsel[:, sidx[kb]]
        a_abs = auroc(s, n_id, n_ood)
        fac = Factor(Lsel, n_id + n_ood)
        excl0 = {kb, o, tw.get(kb), tw.get(o)}
        Ufor = lambda C_names: [sidx[x] for x in base if x not in excl0 and x not in C_names]
        common = {"run": rn, "kb": kb, "ood": o, "tier": tier, "auroc_abs": a_abs}

        def emit(family, role, param, d, c, U, extra=None, stored=None, ckey=None):
            g, share, wk = fac(U)
            S = s - (float(param) if family in ("a2pool", "msunf", "msfil") else 1.0) * c
            A = auroc(S, n_id, n_ood)
            Si, So = S[:n_id], S[n_id:n_id + n_ood]
            d_direct = (Si.mean() - So.mean()) / np.sqrt((Si.var() + So.var()) / 2)
            if ckey is not None:
                emitted[ckey] = emitted.get(ckey, 0) + 1
            if stored is not None:
                chk[ckey][0] = max(chk[ckey][0], abs(A - stored))
                chk[ckey][1] += 1
            recs.append({**common, "draw": d, "family": family, "role": role, "param": str(param), "auroc": A, "pc1_share": share, "w_kbnear_corr": wk,
                         "n_left_out": len(U), "d_rel_direct": d_direct, **class_stats(s, c, g, ll, n_id, n_ood), **(extra or {})})

        for (d, fx), gg in draws_of.get((kb, o), draws.iloc[:0]).groupby(["draw", "fixed"]):
            F = fx.split("+")
            Fi = [sidx[x] for x in F]
            MF = Lsel[:, Fi].max(1)
            for role, slot in zip(gg.role, gg.slot):
                if pd.isna(slot):
                    continue
                C = [*F, slot]
                Ci = [sidx[x] for x in C]
                ss_ = Lsel[:, sidx[slot]]
                gd, _ = group_diag(Tsel[:, Ci, :], n_id, n_ood)
                ex = {**gd}
                for cl, sl in (("id", slice(0, n_id)), ("ood", slice(n_id, n_id + n_ood))):
                    ex.update({f"{cl}_mF": MF[sl].mean(), f"{cl}_exc": np.maximum(ss_ - MF, 0)[sl].mean(), f"{cl}_sslot": ss_[sl].mean(),
                               f"{cl}_mfix": Lsel[sl][:, Fi].mean()})
                for t in taus:
                    c = agg(Lsel[:, Ci], t, m)
                    if t != "inf":
                        cm = np.maximum(MF, ss_)
                        assert (c <= cm + 1e-12).all() and (c >= cm - float(t) * np.log(m) - 1e-12).all()
                    emit("a1", role, t, d, c, Ufor(C), ex, R1.get((kb, o, d, role, t)), "a1")
        if have_a2:
            for d, order in zip(sets_of[(kb, o)].draw, sets_of[(kb, o)].order):
                C = order.split("+")[:m]
                Ci = [sidx[x] for x in C]
                U = Ufor(C)
                gd, pooled = group_diag(Tsel[:, Ci, :], n_id, n_ood)
                for t in taus:
                    emit("a2site", "site", t, d, agg(Lsel[:, Ci], t, m), U, gd, R2.get((kb, o, d, "se", "site", "S_tau", t)), "a2site")
                for a in alphas:
                    emit("a2pool", "pooled", norm_par(a), d, pooled, U, gd, R2.get((kb, o, d, "se", "pooled", "mix", norm_par(a))), "a2pool")
                if doc:
                    Qp = Q[torch.as_tensor(sel, device=DEV)]
                    docs = torch.cat([torch.arange(sidx[x] * cap, (sidx[x] + 1) * cap, device=DEV) for x in C])
                    Sd = (Qp @ Bd[docs].T).double()
                    pr = torch.as_tensor(np.random.default_rng(stable_seed(cfg["seed"], "A2_rand", kb, o, d, m)).permutation(n), device=DEV)
                    Tg = torch.stack([Sd[:, pr[j * cap:(j + 1) * cap]].topk(k, dim=1).values for j in range(m)], dim=1).cpu().numpy()
                    gr, pooled_r = group_diag(Tg, n_id, n_ood)
                    assert np.abs(pooled_r - pooled).max() < 1e-6
                    for t in taus:
                        emit("a2rand", "random", t, d, agg(Tg.mean(2), t, m), U, gr, R2.get((kb, o, d, "se", "random", "S_tau", t)), "a2rand")
                    unf = sample(kb, o, d)
                    rng = np.random.default_rng(stable_seed(cfg["seed"], "A9_arms", kb, o, d))
                    keepm = dmax[kb] < t_common
                    kept = unf[keepm[unf]]
                    rest = np.setdiff1d(np.flatnonzero(keepm), kept)
                    fil = np.concatenate([kept, rng.choice(rest, n - len(kept), replace=False)])
                    Ums = Ufor([])
                    for arm, ix in (("msunf", unf), ("msfil", fil)):
                        c = (Qp @ MS[torch.as_tensor(ix, device=DEV)].T).double().topk(k, dim=1).values.mean(1).cpu().numpy()
                        for a in alphas:
                            stv = R2.get((kb, o, d, "ms", "pooled", "mix", norm_par(a))) if arm == "msunf" else None
                            emit(arm, "pooled", norm_par(a), d, c, Ums, {"frac_removed": float((~keepm[unf]).mean())}, stv, "msunf" if arm == "msunf" else None)
                        a9chk.append({"kb": kb, "ood": o, "draw": d, "arm": arm, "rho_test": float(np.corrcoef(s[:n_id], c[:n_id])[0, 1]),
                                      **{f"auc@{norm_par(a)}": recs[-len(alphas) + j]["auroc"] for j, a in enumerate(alphas)}})
        if ms_only:
            Qp = Q[torch.as_tensor(sel, device=DEV)]
            for d in range(R):
                c = (Qp @ MS[torch.as_tensor(sample(kb, o, d), device=DEV)].T).double().topk(k, dim=1).values.mean(1).cpu().numpy()
                for a in alphas:
                    emit("msunf", "pooled", norm_par(a), d, c, Ufor([]))
        if have_a4 and (kb, o) in lev_of:
            fx_of = draws_of[(kb, o)].drop_duplicates("draw").set_index("draw").fixed
            for d, lv, slot in zip(lev_of[(kb, o)].draw, lev_of[(kb, o)].level, lev_of[(kb, o)].slot):
                if pd.isna(slot):
                    continue
                F = fx_of[d].split("+")
                Fi = [sidx[x] for x in F]
                if lv == "L3":
                    A = np.column_stack([Lsel[:, Fi], L3[sel, sidx[kb]]])
                    Tg = np.concatenate([Tsel[:, Fi, :], T3[sel, sidx[kb]][:, None, :]], axis=1)
                    C = F
                else:
                    A = Lsel[:, Fi + [sidx[slot]]]
                    Tg = Tsel[:, Fi + [sidx[slot]], :]
                    C = [*F, slot]
                gd, _ = group_diag(Tg, n_id, n_ood)
                for t in taus:
                    emit("a4", lv, t, d, agg(A, t, m), Ufor(C), gd, R4.get((kb, o, d, lv, t)), "a4")
        if pi_ % 100 == 0:
            log.info(f"{rn}: pair {pi_}/{len(pairs)} ({(time.time() - t0) / 60:.1f} min)")
    for kk, (e, nchk) in (chk.items() if run.get("check", True) else []):
        log.info(f"{rn}: {kk}: max |AUROC(recomputed) - stored| = {e:.2e} over {nchk} checks of {emitted.get(kk, 0)} emitted rows")
        assert nchk > 0 and nchk == emitted.get(kk, 0) and e < 1e-5, (rn, kk, e, nchk, emitted.get(kk))
    D = pd.DataFrame(recs)
    if doc:
        a9all = pd.DataFrame(a9chk).groupby(["kb", "ood", "arm"]).mean(numeric_only=True)
        a9 = a9all.rho_test.unstack()
        e9 = max(float(np.abs(a9["msunf"] - A9c.loc[a9.index, "rho_test_unf"]).max()), float(np.abs(a9["msfil"] - A9c.loc[a9.index, "rho_test_fil"]).max()))
        e9a = max(float(np.abs(a9all[f"auc@{norm_par(a)}"].unstack()[arm] - A9c.loc[a9.index, f"auc_{nm}@{a}"]).max())
                  for a in alphas for arm, nm in (("msunf", "unf"), ("msfil", "fil")))
        log.info(f"{rn}: A9 reproduction over {len(a9)} cells: rho_test max diff {e9:.2e}, cell-mean AUROC (unf, fil) max diff {e9a:.2e}")
        assert e9 < 1e-6 and e9a < 1e-5, (rn, e9, e9a)
        D.attrs["t_common"] = t_common
    return D, {kk: v for kk, v in chk.items()}, (t_common if doc else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--smoke-pairs", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    assert not args.smoke_pairs or args.out, "--smoke-pairs requires --out (never write smoke output into the results directory)"
    cfg = load_config(args.config)
    res = Path(args.out) if args.out else p(cfg, "results")
    (res / "tables").mkdir(parents=True, exist_ok=True)
    log = get_logger("A10.01", res / "logs" / "01_collect.log")
    for rn in args.runs or list(cfg["runs"]):
        run = cfg["runs"][rn]
        t0 = time.time()
        D, chk, tc = run_one(rn, run, cfg, log, args.smoke_pairs)
        D.to_parquet(res / "tables" / f"stats_{rn}.parquet", index=False)
        meta = run_metadata(cfg, f"A10.01_collect:{rn}")
        root = PROJECT_ROOT
        ins = [root / run["emb"] / f for f in ("topk.npy", "q_index.parquet", "sites.json")] + [root / run["a1"] / "tables" / f for f in ("pairs.csv", "draws.csv", "rows.parquet")]
        if "a2" in run:
            ins += [root / run["a2"] / "tables" / f for f in ("sets.csv", "rows.parquet")]
        if "a4" in run:
            ins += [root / run["a4"] / "tables" / f for f in ("levels.csv", "rows.parquet")] + [root / run["a4_l3"]]
        if run["doc"]:
            ins += [root / run["a9"], p(cfg, "p9_emb", run["retriever"], "pool.npy")]
            ins += [p(cfg, "a0_emb", run["retriever"], f"{s_}__{ro}.npy") for s_ in load_json(root / run["emb"] / "sites.json") for ro in ("test", "cal", "bank")]
        if run.get("ms_only"):
            ins += [p(cfg, "p9_emb", run["retriever"], "pool.npy")] + [root / run["q_emb"] / f"{s_}__{ro}.npy" for s_ in load_json(root / run["emb"] / "sites.json") for ro in ("test", "cal")]
        ins += ([root / run["items"]] if run.get("items") else [p(cfg, "a0_items")]) if run["queries"] == "a0" else [p(cfg, "p4_queries"), p(cfg, "a8_processed", "sites.json")]
        meta["inputs"] = fingerprint(ins)
        meta["checks"] = {kk: {"max_abs_diff": v[0], "n": v[1]} for kk, v in chk.items()}
        meta["t_common_a9"], meta["n_rows"], meta["minutes"], meta["smoke_pairs"] = tc, len(D), (time.time() - t0) / 60, args.smoke_pairs
        save_json(meta, res / f"meta_01_{rn}.json")
        log.info(f"{rn}: {len(D)} rows in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
