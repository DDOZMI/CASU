from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import PROJECT_ROOT, fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json, set_determinism, stable_seed

set_determinism()
import torch

DEV = "cuda"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A2 = _load("a2_eval", PROJECT_ROOT / "experiments" / "A2" / "02_eval.py")
D3 = _load("a3_diag", PROJECT_ROOT / "experiments" / "A3" / "02_diagnose.py")


def unit_t(X):
    X = torch.as_tensor(np.asarray(X, np.float32), device=DEV)
    return X / X.norm(dim=1, keepdim=True)


def ledoit_wolf(X):
    mu = X.mean(0)
    Xc = (X - mu).double()
    n, pdim = Xc.shape
    emp = Xc.T @ Xc / n
    tr = torch.diagonal(emp)
    m = tr.sum() / pdim
    X2 = Xc ** 2
    beta_ = (X2.T @ X2).sum()
    delta_ = (emp ** 2).sum()
    beta = (beta_ / n - delta_) / (pdim * n)
    delta = (delta_ - 2 * m * tr.sum() + pdim * m ** 2) / pdim
    beta = torch.minimum(beta, delta)
    sh = 0.0 if float(beta) == 0 else float(beta / delta)
    cov = (1 - sh) * emp
    cov.diagonal().add_(sh * m)
    return mu.double(), cov, sh


class Gauss:
    def __init__(self, X):
        self.mu, cov, self.shrink = ledoit_wolf(X)
        self.P = torch.linalg.inv(cov)

    def maha(self, Q):
        D = Q.double() - self.mu
        return ((D @ self.P) * D).sum(1)


def logreg(X, y, C, max_iter, balanced=True):
    y = y.double()
    n1 = float(y.sum())
    n0 = len(y) - n1
    sw = torch.where(y > 0, len(y) / (2 * n1), len(y) / (2 * n0)) if balanced else torch.ones_like(y)
    sw = sw.float()
    ys = (2 * y - 1).float()
    w = torch.zeros(X.shape[1], device=DEV, requires_grad=True)
    b = torch.zeros(1, device=DEV, requires_grad=True)
    scale = 1.0 / (C * float(sw.sum()))
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=max_iter, tolerance_grad=1e-9, tolerance_change=1e-12, history_size=20,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = X @ w + b
        loss = 0.5 * scale * (w * w).sum() + (sw * torch.nn.functional.softplus(-ys * z)).sum() * scale * C
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def logreg_oof(X, y, folds, Cs, max_iter):
    F, nC = len(folds), len(Cs)
    ys = torch.as_tensor(2 * y - 1, device=DEV, dtype=torch.float32)
    Z, SW, TE = [], [], []
    for tr, te in folds:
        tr_t = torch.as_tensor(tr, device=DEV)
        mu_, sd_ = X[tr_t].mean(0), X[tr_t].std(0) + 1e-8
        Z.append((X - mu_) / sd_)
        n1, n0 = float((y[tr] == 1).sum()), float((y[tr] == 0).sum())
        sw = np.zeros(len(y), np.float32)
        sw[tr] = np.where(y[tr] == 1, len(tr) / (2 * n1), len(tr) / (2 * n0))
        SW.append(sw)
        TE.append(te)
    Z = torch.stack(Z)
    SW = torch.as_tensor(np.stack(SW), device=DEV)
    Cv = torch.as_tensor(Cs, device=DEV, dtype=torch.float32)
    W = torch.zeros(F, nC, X.shape[1], device=DEV, requires_grad=True)
    b = torch.zeros(F, nC, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=max_iter, tolerance_grad=1e-9, tolerance_change=1e-12, history_size=20,
                            line_search_fn="strong_wolfe")
    ssum = SW.sum(1)

    def closure():
        opt.zero_grad()
        zz = torch.einsum("fnd,fcd->fcn", Z, W) + b[:, :, None]
        data = (SW[:, None, :] * torch.nn.functional.softplus(-ys[None, None, :] * zz)).sum(2) / ssum[:, None]
        reg = 0.5 * (W * W).sum(2) / (Cv[None, :] * ssum[:, None])
        loss = (data + reg).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        zz = (torch.einsum("fnd,fcd->fcn", Z, W) + b[:, :, None]).double().cpu().numpy()
    out = {}
    for ci, C in enumerate(Cs):
        dec = np.empty(len(y))
        for fi, te in enumerate(TE):
            dec[te] = zz[fi, ci, te]
        out[C] = dec
    return out


def auroc_np(S, n_id, n_ood):
    X = -np.asarray(S, np.float64)
    r = rankdata(X)
    return float((r[n_id:n_id + n_ood].sum() - n_ood * (n_ood + 1) / 2) / (n_ood * n_id))


def fpr95(Sc, n_id, n_ood):
    Si = torch.sort(Sc[:, :n_id].double(), dim=1).values
    t = Si[:, int(np.floor(0.05 * n_id))]
    return (Sc[:, n_id:n_id + n_ood].double() >= t[:, None]).double().mean(1).cpu().numpy()


def run_one(rn, run, cfg, log, smoke_pairs=0):
    k, m, cap, R = cfg["k"], cfg["m"], cfg["cap"], cfg["R"]
    n = m * cap
    li, kp, cl, orc, mt = cfg["li"], cfg["kbpca"], cfg["clf"], cfg["oracle"], cfg["metrics"]
    r = run["retriever"]
    edir = PROJECT_ROOT / run["emb"]
    sites = load_json(edir / "sites.json")
    sidx = {s: i for i, s in enumerate(sites)}
    T = np.asarray(np.load(edir / "topk.npy", mmap_mode="r"))
    Lm = T.astype(np.float64).mean(2)
    qi = pd.read_parquet(edir / "q_index.parquet")
    rows_of = {kk: g.index.to_numpy() for kk, g in qi.groupby(["site", "role"])}
    a0 = p(cfg, "a0_emb", r)
    Q = unit_t(np.concatenate([np.load(a0 / f"{s}__{ro}.npy") for s in sites for ro in ("test", "cal")]))
    assert Q.shape[0] == len(qi)
    Bd = unit_t(np.concatenate([np.load(a0 / f"{s}__bank.npy") for s in sites]))
    assert Bd.shape[0] == len(sites) * cap
    MS = unit_t(np.load(p(cfg, "p9_emb", r, "pool.npy")))
    WK = unit_t(np.load(p(cfg, "a12_emb", r, "wiki.npy")))
    MQ = unit_t(np.load(p(cfg, "a12_emb", r, "msq.npy")))
    em = load_json(p(cfg, "a12_emb", r, "meta.json"))
    assert em["n_msq"] == MQ.shape[0] == 20000 and em["n_wiki"] == WK.shape[0] and not em["upstream_fingerprint"]["smoke"], em
    rq = torch.cat([(Q[i:i + 8192] @ MQ.T).topk(k, dim=1).values.double().mean(1) for i in range(0, len(Q), 8192)]).cpu().numpy()
    pairs = pd.read_csv(PROJECT_ROOT / run["a1"] / "tables" / "pairs.csv")[["kb", "ood", "tier"]]
    if smoke_pairs:
        pairs = pairs.iloc[:: max(1, len(pairs) // smoke_pairs)].head(smoke_pairs)
    sets = pd.read_csv(PROJECT_ROOT / run["a2"] / "tables" / "sets.csv")
    sets_of = {(a, b, d): o for a, b, d, o in zip(sets.kb, sets.ood, sets.draw, sets.order)}
    R2 = pd.read_parquet(PROJECT_ROOT / run["a2"] / "tables" / "rows.parquet", columns=["kb", "ood", "draw", "m", "corpus", "partition", "family", "param", "auroc"])
    R2 = R2[(R2.m == m) & (R2.corpus == "ms") & (R2.partition == "pooled")].set_index(["kb", "ood", "draw", "family", "param"]).auroc
    a1abs = pd.read_parquet(PROJECT_ROOT / run["a1"] / "tables" / "rows.parquet", columns=["kb", "ood", "family", "auroc_abs"])
    a1abs = a1abs[a1abs.family == "S_tau"].groupby(["kb", "ood"]).auroc_abs.first()
    alphas = cfg["alphas"]
    bg_idx = {(bgn, d): np.random.default_rng(stable_seed(cfg["seed"], "A13_bg", bgn, d)).permutation(BG.shape[0])[:n]
              for bgn, BG in (("ms", MS), ("wiki", WK)) for d in range(R)}
    bg_gauss = {key: Gauss((MS if key[0] == "ms" else WK)[torch.as_tensor(ix, device=DEV)]) for key, ix in bg_idx.items()}
    out, chk, kbcache = [], {"a2_ms": [0.0, 0], "a1_abs": [0.0, 0], "s_kb": [0.0, 0]}, {}
    draws = []
    t0 = time.time()
    for pi_, (kb, o, tier) in enumerate(zip(pairs.kb, pairs.ood, pairs.tier)):
        if kb not in kbcache:
            kbcache.clear()
            Bk = Bd[sidx[kb] * cap:(sidx[kb] + 1) * cap]
            gk = Gauss(Bk)
            Bc = (Bk - Bk.mean(0)).double()
            _, _, Vt = torch.linalg.svd(Bc, full_matrices=False)
            V = Vt[: kp["m"]].float()
            i_c = rows_of[(kb, "cal")]
            Pc = Q[torch.as_tensor(i_c, device=DEV)] @ V.T
            Pd = Bk @ V.T
            sc, rc_ = Lm[i_c, sidx[kb]], rq[i_c]
            beta = float(np.cov(sc, rc_, bias=True)[0, 1] / rc_.var())
            clfms = {}
            for d in range(R):
                NEG = MS[torch.as_tensor(bg_idx[("ms", d)], device=DEV)]
                clfms[d] = logreg(torch.cat([Bk, NEG]), torch.cat([torch.ones(len(Bk), device=DEV), torch.zeros(len(NEG), device=DEV)]), cl["C"], cl["max_iter"])
            kbcache[kb] = (Bk, gk, V, Pc, Pd, beta, clfms)
        Bk, gk, V, Pc, Pd, beta, clfms = kbcache[kb]
        i_id, i_ood, i_cal = rows_of[(kb, "test")], rows_of[(o, "test")], rows_of[(kb, "cal")]
        sel = np.r_[i_id, i_ood, i_cal]
        n_id, n_ood, n_cal = len(i_id), len(i_ood), len(i_cal)
        Qs = Q[torch.as_tensor(sel, device=DEV)]
        Skb = Qs @ Bk.T
        T32 = Skb.topk(li["k"], dim=1).values.double()
        s = T32[:, :k].mean(1)
        absk = T32[:, k - 1]
        e = float((s.cpu().numpy() - Lm[sel, sidx[kb]]).__abs__().max())
        chk["s_kb"] = [max(chk["s_kb"][0], e), chk["s_kb"][1] + 1]
        tau = li["tau"]
        P32 = torch.softmax(T32 / tau, dim=1)
        P_ = Qs @ V.T
        Dq = torch.cdist(P_, Pc)
        Dq[torch.arange(n_id + n_ood, n_id + n_ood + n_cal, device=DEV), torch.arange(n_cal, device=DEV)] = float("inf")
        Dd = torch.cdist(P_, Pd)
        Dfull = torch.cdist(Qs, Q[torch.as_tensor(i_cal, device=DEV)])
        Dfull[torch.arange(n_id + n_ood, n_id + n_ood + n_cal, device=DEV), torch.arange(n_cal, device=DEV)] = float("inf")
        rsel = torch.as_tensor(rq[sel], device=DEV, dtype=torch.float64)
        pair_scores = {
            "abs": s, "li_mss": T32[:, 0], "li_knn10": absk, "li_knn32": T32[:, li["k"] - 1], "li_avg32": T32.mean(1),
            "li_energy32": tau * torch.logsumexp(T32 / tau, dim=1), "li_entropy32": (P32 * torch.log(P32)).sum(1),
            "maha_kb": -gk.maha(Qs), "kbpca_q": -Dq.topk(kp["knn_k"], dim=1, largest=False).values[:, -1].double(),
            "kbpca_d": -Dd.topk(kp["knn_k"], dim=1, largest=False).values[:, -1].double(),
            "qz_kb": (s - Skb.double().mean(1)) / Skb.double().std(1), "qz_ms": s - beta * rsel,
            "knn_cal_full": -Dfull.topk(kp["knn_k"], dim=1, largest=False).values[:, -1].double(),
            "kbpca_res": -(Qs - P_ @ V).norm(dim=1).double()}
        names = list(pair_scores)
        Sc = torch.stack([pair_scores[x] for x in names])
        auc, ida, ooda = A2.gpu_metrics(Sc, n_id, n_ood, mt["op_accept"])
        fp = fpr95(Sc, n_id, n_ood)
        base = {"run": rn, "kb": kb, "ood": o, "tier": tier}
        recs = {nm: {"auroc": a_, "op_id_acc": ia, "op_ood_acc": oa, "fpr95": f_} for nm, a_, ia, oa, f_ in zip(names, auc, ida, ooda, fp)}
        if (kb, o) in a1abs.index:
            chk["a1_abs"] = [max(chk["a1_abs"][0], abs(recs["abs"]["auroc"] - a1abs[(kb, o)])), chk["a1_abs"][1] + 1]
        sn = s.cpu().numpy()
        sl_i, sl_o = slice(0, n_id), slice(n_id, n_id + n_ood)

        def mom(c):
            mo = D3.moments(sn[sl_i], sn[sl_o], c[sl_i], c[sl_o])
            return {"pi": mo["pi"], "T_var": mo["T_var"], "T_push": mo["T_push"], "d_abs_m": mo["d_abs_m"], "dd_m": mo["dd_m"],
                    "rho_id": float(np.corrcoef(sn[sl_i], c[sl_i])[0, 1]), "rho_ood": float(np.corrcoef(sn[sl_o], c[sl_o])[0, 1])}
        recs["qz_ms"].update(mom(beta * rq[sel]))
        recs["qz_ms"]["beta"] = beta
        y = np.r_[np.zeros(n_id), np.ones(n_ood)]
        skf = StratifiedKFold(orc["folds"], shuffle=True, random_state=stable_seed(cfg["seed"], "A13_oracle", kb, o))
        folds = list(skf.split(np.zeros(len(y)), y))
        for nm, X in (("oracle", Qs[: n_id + n_ood]), ("tri_sup", P_[: n_id + n_ood])):
            decs = logreg_oof(X, y, folds, orc["C_grid"], orc["max_iter"])
            per = {C: auroc_np(-dv, n_id, n_ood) for C, dv in decs.items()}
            recs[nm] = {"auroc": max(per.values()), "best_C": max(per, key=per.get)}
            for C, a_ in per.items():
                recs[f"{nm}@C{C:g}"] = {"auroc": a_}
        mk = gk.maha(Qs)
        acc = {}
        for d in range(R):
            ms_idx = np.random.default_rng(stable_seed(cfg["seed"], "A2_ms", kb, o, d)).permutation(MS.shape[0])[:n]
            wk_idx = np.random.default_rng(stable_seed(cfg["seed"], "A13_wiki", kb, o, d)).permutation(WK.shape[0])[:n]
            dn, vec = [], []
            for bgn, BG, ix in (("ms", MS, ms_idx), ("wiki", WK, wk_idx)):
                X = BG[torch.as_tensor(ix, device=DEV)]
                top = (Qs @ X.T).topk(k, dim=1).values.double()
                cm, ck = top.mean(1), top[:, k - 1]
                for a in alphas:
                    dn += [(f"mix_{bgn}", str(a)), (f"mixk_{bgn}", str(a))]
                    vec += [s - a * cm, absk - a * ck]
                for a in cfg["flats_alphas"]:
                    dn.append((f"flats_{bgn}", str(a)))
                    vec.append(-torch.sqrt(torch.clamp(2 - 2 * absk, min=0)) + a * torch.sqrt(torch.clamp(2 - 2 * ck, min=0)))
                dn.append((f"rmd_{bgn}", ""))
                vec.append(-(mk - bg_gauss[(bgn, d)].maha(Qs)))
                cmn = cm.cpu().numpy()
                for a in cfg["moment_alphas"]:
                    acc.setdefault((f"mix_{bgn}", str(a)), []).append(mom(a * cmn))
            order = sets_of[(kb, o, d)].split("+")[:m]
            se_docs = torch.cat([Bd[sidx[x] * cap:(sidx[x] + 1) * cap] for x in order])
            w, b = logreg(torch.cat([Bk, se_docs]), torch.cat([torch.ones(len(Bk), device=DEV), torch.zeros(len(se_docs), device=DEV)]), cl["C"], cl["max_iter"])
            for cn, (w_, b_) in (("clf_ms", clfms[d]), ("clf_se", (w, b))):
                dn.append((cn, ""))
                vec.append((Qs @ w_ + b_).double())
            Sd = torch.stack(vec)
            auc, ida, ooda = A2.gpu_metrics(Sd, n_id, n_ood, mt["op_accept"])
            fp = fpr95(Sd, n_id, n_ood)
            for (nm, par), a_, ia, oa, f_ in zip(dn, auc, ida, ooda, fp):
                if nm in ("mix_ms", "mixk_ms"):
                    key = (kb, o, d, nm.split("_")[0], par)
                    if key in R2.index:
                        chk["a2_ms"] = [max(chk["a2_ms"][0], abs(a_ - R2[key])), chk["a2_ms"][1] + 1]
                acc.setdefault((nm, par), []).append({"auroc": a_, "op_id_acc": ia, "op_ood_acc": oa, "fpr95": f_})
                draws.append({**base, "draw": d, "score": f"{nm}@{par}" if par else nm, "auroc": a_, "op_id_acc": ia, "op_ood_acc": oa, "fpr95": f_})
        for (nm, par), lst in acc.items():
            full = f"{nm}@{par}" if par else nm
            agg = pd.DataFrame(lst).mean(numeric_only=True).to_dict()
            recs.setdefault(full, {}).update(agg)
        a_abs = recs["abs"]["auroc"]
        for nm, rr in recs.items():
            out.append({**base, "score": nm, "auroc_abs": a_abs, **rr})
        if pi_ % 50 == 0:
            log.info(f"{rn}: pair {pi_}/{len(pairs)} ({(time.time() - t0) / 60:.1f} min)")
    for kk, (e, nchk) in chk.items():
        log.info(f"{rn}: check {kk}: max |diff| = {e:.2e} over {nchk}")
    assert chk["s_kb"][1] > 0 and chk["s_kb"][0] < 1e-5, chk["s_kb"]
    assert (chk["a1_abs"][1] > 0 or smoke_pairs) and chk["a1_abs"][0] < 1e-5, chk["a1_abs"]
    assert chk["a2_ms"][1] == len(pairs) * R * 2 * len(alphas) and chk["a2_ms"][0] < 1e-5, chk["a2_ms"]
    return pd.DataFrame(out), chk, pd.DataFrame(draws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--smoke-pairs", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    assert not args.smoke_pairs or args.out, "--smoke-pairs requires --out"
    cfg = load_config(args.config)
    res = Path(args.out) if args.out else p(cfg, "results")
    (res / "tables").mkdir(parents=True, exist_ok=True)
    log = get_logger("A13.01", res / "logs" / "01_baselines.log")
    for rn in args.runs or list(cfg["runs"]):
        run = cfg["runs"][rn]
        t0 = time.time()
        D, chk, DR = run_one(rn, run, cfg, log, args.smoke_pairs)
        D.to_parquet(res / "tables" / f"cells_{rn}.parquet", index=False)
        DR.to_parquet(res / "tables" / f"draws_{rn}.parquet", index=False)
        meta = run_metadata(cfg, f"A13.01_baselines:{rn}")
        r = run["retriever"]
        ins = [PROJECT_ROOT / run["emb"] / f for f in ("topk.npy", "q_index.parquet", "sites.json")]
        ins += [PROJECT_ROOT / run["a1"] / "tables" / f for f in ("pairs.csv", "rows.parquet")] + [PROJECT_ROOT / run["a2"] / "tables" / f for f in ("sets.csv", "rows.parquet")]
        ins += [p(cfg, "p9_emb", r, "pool.npy"), p(cfg, "a12_emb", r, "wiki.npy"), p(cfg, "a12_emb", r, "msq.npy")]
        ins += [p(cfg, "a0_emb", r, f"{s_}__{ro}.npy") for s_ in load_json(PROJECT_ROOT / run["emb"] / "sites.json") for ro in ("test", "cal", "bank")]
        meta["inputs"] = fingerprint(ins)
        meta["imported_code_sha256"] = fingerprint([PROJECT_ROOT / "experiments" / "A2" / "02_eval.py", PROJECT_ROOT / "experiments" / "A3" / "02_diagnose.py"])
        meta["checks"] = {kk: {"max_abs_diff": v[0], "n": v[1]} for kk, v in chk.items()}
        meta["n_rows"], meta["minutes"], meta["smoke_pairs"] = len(D), (time.time() - t0) / 60, args.smoke_pairs
        save_json(meta, res / f"meta_01_{rn}.json")
        log.info(f"{rn}: {len(D)} rows in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
