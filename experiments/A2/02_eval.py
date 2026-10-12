from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json, set_determinism, stable_seed

set_determinism()
import torch

DEV = "cuda"


def unit_t(X):
    X = torch.as_tensor(np.asarray(X, np.float32), device=DEV)
    return X / X.norm(dim=1, keepdim=True)


def balanced_kmeans(X, k, cap, seed, iters):
    n = X.shape[0]
    assert n == k * cap
    if k == 1:
        return [torch.arange(n, device=DEV)]
    rng = np.random.default_rng(seed)
    Xc = X.cpu().numpy().astype(np.float64)
    cent = [int(rng.integers(n))]
    dmin = 1 - Xc @ Xc[cent[0]]
    for _ in range(1, k):
        pr = np.maximum(dmin, 0) ** 2
        c = int(rng.choice(n, p=pr / pr.sum()))
        cent.append(c)
        dmin = np.minimum(dmin, 1 - Xc @ Xc[c])
    C = X[cent].clone()
    for _ in range(iters):
        S = X @ C.T
        asg = torch.full((n,), -1, dtype=torch.long, device=DEV)
        rem = torch.full((k,), cap, dtype=torch.long, device=DEV)
        while bool((asg < 0).any()):
            un = torch.nonzero(asg < 0).squeeze(1)
            Su = S[un].clone()
            Su[:, rem == 0] = float("-inf")
            best = Su.argmax(1)
            for c in range(k):
                prop = un[best == c]
                if len(prop) == 0 or int(rem[c]) == 0:
                    continue
                take = min(int(rem[c]), len(prop))
                sv = S[prop, c]
                order = torch.argsort(-sv, stable=True)[:take]
                asg[prop[order]] = c
                rem[c] -= take
        C = torch.stack([X[asg == c].mean(0) for c in range(k)])
        C = C / C.norm(dim=1, keepdim=True)
    return [torch.nonzero(asg == c).squeeze(1) for c in range(k)]


def assign_rounds_np(S, k, cap):
    n = S.shape[0]
    asg = np.full(n, -1)
    rem = np.full(k, cap)
    while (asg < 0).any():
        un = np.flatnonzero(asg < 0)
        Su = S[un].copy()
        Su[:, rem == 0] = -np.inf
        best = Su.argmax(1)
        for c in range(k):
            prop = un[best == c]
            if len(prop) == 0 or rem[c] == 0:
                continue
            take = min(rem[c], len(prop))
            order = np.argsort(-S[prop, c], kind="stable")[:take]
            asg[prop[order]] = c
            rem[c] -= take
    return asg


def balanced_kmeans_tags(M, k, cap, seed, iters):
    n = M.shape[0]
    if k == 1:
        return [torch.arange(n, device=DEV)]
    M = M[:, np.flatnonzero(np.asarray(M.sum(0)).ravel() > 0)].tocsr()
    dfreq = np.asarray((M > 0).sum(0)).ravel()
    idf = np.log((1 + n) / (1 + dfreq)) + 1
    X = M.multiply(idf[None, :]).tocsr()
    X = sp.diags(1 / np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())) @ X
    rng = np.random.default_rng(seed)
    cent = [int(rng.integers(n))]
    dmin = 1 - np.asarray((X @ X[cent[0]].T).todense()).ravel()
    for _ in range(1, k):
        pr = np.maximum(dmin, 0) ** 2
        c = int(rng.choice(n, p=pr / pr.sum()))
        cent.append(c)
        dmin = np.minimum(dmin, 1 - np.asarray((X @ X[c].T).todense()).ravel())
    C = np.asarray(X[cent].todense())
    for _ in range(iters):
        S = np.asarray(X @ C.T)
        asg = assign_rounds_np(S, k, cap)
        C = np.asarray((sp.csr_matrix((np.ones(n), (asg, np.arange(n))), shape=(k, n)) @ X).todense())
        C /= np.linalg.norm(C, axis=1, keepdims=True)
    return [torch.as_tensor(np.flatnonzero(asg == c), device=DEV) for c in range(k)]


def purity(groups, site_of):
    return float(np.mean([np.bincount(site_of[g].cpu().numpy()).max() / len(g) for g in groups]))


def group_logits(S, groups, k):
    return torch.stack([S[:, g].topk(k, dim=1).values.mean(1) for g in groups], dim=1)


def s_tau(s_kb, G, t):
    m = G.shape[1]
    if t == "inf":
        return s_kb - G.mean(1)
    return s_kb - t * (torch.logsumexp(G / t, dim=1) - np.log(m))


def gpu_metrics(Sc, n_id, n_ood, acc):
    X = -Sc[:, : n_id + n_ood]
    srt, idx = torch.sort(X, dim=1, stable=True)
    ranks = torch.empty_like(X)
    ar = torch.arange(1, X.shape[1] + 1, device=DEV, dtype=torch.float64).expand_as(X)
    ranks.scatter_(1, idx, ar)
    eq = torch.cat([torch.zeros_like(srt[:, :1], dtype=torch.bool), srt[:, 1:] == srt[:, :-1]], dim=1)
    if bool(eq.any()):
        Xn = X.cpu().numpy()
        from scipy.stats import rankdata
        ranks = torch.as_tensor(np.stack([rankdata(r) for r in Xn]), device=DEV)
    rp = ranks[:, n_id:].sum(1)
    auc = (rp - n_ood * (n_ood + 1) / 2) / (n_ood * n_id)
    cal = torch.sort(Sc[:, n_id + n_ood:].double(), dim=1).values
    t = cal[:, int(np.floor((1 - acc) * (cal.shape[1] - 1)))]
    id_acc = (Sc[:, :n_id].double() >= t[:, None]).double().mean(1)
    ood_acc = (Sc[:, n_id:n_id + n_ood].double() >= t[:, None]).double().mean(1)
    return auc.cpu().numpy(), id_acc.cpu().numpy(), ood_acc.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--smoke-pairs", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--retriever", default=None, help="default: config detector (A5: per model)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    r = args.retriever or cfg["detector"]
    rc = cfg.get("cross_encoder_by_detector", {}).get(r, cfg["cross_encoder"])
    res = Path(args.out) if args.out else (p(cfg, "results", r) if cfg.get("results_per_retriever") else p(cfg, "results"))
    (res / "tables").mkdir(parents=True, exist_ok=True)
    log = get_logger("A2.02", res / "logs" / "02_eval.log")
    t0 = time.time()
    k, cap = cfg["k"], 3000
    sc, pc = cfg["scores"], cfg["partitions"]
    acc = cfg["metrics"]["id_accept"]
    edir = p(cfg, "a1_emb", r)
    sites = load_json(edir / "sites.json")
    sidx = {s: i for i, s in enumerate(sites)}
    qi = pd.read_parquet(edir / "q_index.parquet")
    rows_of = {kk: g.index.to_numpy() for kk, g in qi.groupby(["site", "role"])}
    T = np.load(edir / "topk.npy", mmap_mode="r")
    pairs = pd.read_csv(p(cfg, "a1_results", "tables", "pairs.csv"))[["kb", "ood", "tier"]]
    if args.smoke_pairs:
        pairs = pairs.iloc[:: max(1, len(pairs) // args.smoke_pairs)].head(args.smoke_pairs)
    a0 = p(cfg, "a0_emb")
    Q = unit_t(np.concatenate([np.load(a0 / r / f"{s}__{ro}.npy") for s in sites for ro in ("test", "cal")]))
    assert Q.shape[0] == len(qi)
    Bd = unit_t(np.concatenate([np.load(a0 / r / f"{s}__bank.npy") for s in sites]))
    Bc = unit_t(np.concatenate([np.load(a0 / rc / f"{s}__bank.npy") for s in sites]))
    TM = sp.load_npz(p(cfg, "emb", "tag_incidence.npz")).tocsr()
    assert Bd.shape[0] == Bc.shape[0] == TM.shape[0] == len(sites) * cap
    MSd = unit_t(np.load(p(cfg, "p9_emb", r, "pool.npy")))
    MSc = unit_t(np.load(p(cfg, "p9_emb", rc, "pool.npy")))
    m_grid, R = cfg["sets"]["m_grid"], cfg["sets"]["R"]
    mmax = max(m_grid)
    taus, alphas = sc["tau_grid"], sc["alpha_grid"]
    rows, sets_rows = [], []
    for pi, (kb, o, tier) in enumerate(zip(pairs.kb, pairs.ood, pairs.tier)):
        i_id, i_ood, i_cal = rows_of[(kb, "test")], rows_of[(o, "test")], rows_of[(kb, "cal")]
        sel = np.r_[i_id, i_ood, i_cal]
        n_id, n_ood = len(i_id), len(i_ood)
        Tsel = torch.as_tensor(np.asarray(T[sel, sidx[kb]]), device=DEV).double()
        s_kb, absk = Tsel.mean(1), Tsel[:, k - 1]
        Qp = Q[torch.as_tensor(sel, device=DEV)]
        ab = torch.stack([s_kb, absk])
        a_auc, a_id, a_ood = gpu_metrics(ab, n_id, n_ood, acc)
        abs_m = {"abs": (a_auc[0], a_id[0], a_ood[0]), "absk": (a_auc[1], a_id[1], a_ood[1])}
        elig = [s for s in sites if s not in (kb, o)]
        for d in range(R):
            order = [elig[i] for i in np.random.default_rng(stable_seed(cfg["seed"], "A2_sets", kb, o, d)).permutation(len(elig))[:mmax]]
            docs = torch.cat([torch.arange(sidx[s] * cap, (sidx[s] + 1) * cap, device=DEV) for s in order])
            S = (Qp @ Bd[docs].T).double()
            ms_idx = torch.as_tensor(np.random.default_rng(stable_seed(cfg["seed"], "A2_ms", kb, o, d)).permutation(MSd.shape[0])[: mmax * cap], device=DEV)
            Sm = (Qp @ MSd[ms_idx].T).double()
            sets_rows.append({"kb": kb, "ood": o, "draw": d, "order": "+".join(order)})
            for m in m_grid:
                n = m * cap
                Sn, Smn = S[:, :n], Sm[:, :n]
                parts = {"site": [torch.arange(j * cap, (j + 1) * cap, device=DEV) for j in range(m)]}
                site_of = torch.arange(m, device=DEV).repeat_interleave(cap)
                if m == 1:
                    for nm in ("tag", "cross", "same", "random"):
                        parts[nm] = parts["site"]
                    ms_parts = {"random": parts["site"], "cross": parts["site"]}
                else:
                    pr = torch.as_tensor(np.random.default_rng(stable_seed(cfg["seed"], "A2_rand", kb, o, d, m)).permutation(n), device=DEV)
                    parts["random"] = [pr[j * cap:(j + 1) * cap] for j in range(m)]
                    dm = docs[:n]
                    parts["tag"] = balanced_kmeans_tags(TM[dm.cpu().numpy()], m, cap, stable_seed(cfg["seed"], "A2_km", "tag", kb, o, d, m), pc["kmeans_iters"])
                    for nm, X in (("cross", Bc), ("same", Bd)):
                        parts[nm] = balanced_kmeans(X[dm], m, cap, stable_seed(cfg["seed"], "A2_km", nm, kb, o, d, m), pc["kmeans_iters"])
                    prm = torch.as_tensor(np.random.default_rng(stable_seed(cfg["seed"], "A2_msrand", kb, o, d, m)).permutation(n), device=DEV)
                    ms_parts = {"random": [prm[j * cap:(j + 1) * cap] for j in range(m)],
                                "cross": balanced_kmeans(MSc[ms_idx[:n]], m, cap, stable_seed(cfg["seed"], "A2_km_ms", kb, o, d, m), pc["kmeans_iters"])}
                names, vecs = [], []
                for src, PP, SS in (("se", parts, Sn), ("ms", ms_parts, Smn)):
                    for nm, groups in PP.items():
                        G = group_logits(SS, groups, k)
                        for t in taus:
                            names.append((src, nm, "S_tau", str(t)))
                            vecs.append(s_tau(s_kb, G, t))
                    top = SS.topk(k, dim=1).values
                    for a in alphas:
                        names.append((src, "pooled", "mix", str(a)))
                        vecs.append(s_kb - a * top.mean(1))
                        names.append((src, "pooled", "mixk", str(a)))
                        vecs.append(absk - a * top[:, k - 1])
                pur = {nm: purity(gr, site_of) for nm, gr in parts.items()}
                auc, ida, ooda = gpu_metrics(torch.stack(vecs), n_id, n_ood, acc)
                for (src, nm, fam, par), A, ia, oa in zip(names, auc, ida, ooda):
                    b = "absk" if fam == "mixk" else "abs"
                    rows.append({"kb": kb, "ood": o, "tier": tier, "draw": d, "m": m, "corpus": src, "partition": nm, "family": fam,
                                 "param": par, "auroc": float(A), "op_id_acc": float(ia), "op_ood_acc": float(oa),
                                 "auroc_abs": float(abs_m[b][0]), "op_ood_acc_abs": float(abs_m[b][2]), "op_id_acc_abs": float(abs_m[b][1]),
                                 "purity": pur.get(nm, np.nan) if src == "se" else np.nan})
        if pi % 50 == 0:
            log.info(f"pair {pi}/{len(pairs)} ({(time.time() - t0) / 60:.1f} min)")
    df = pd.DataFrame(rows)
    df.to_parquet(res / "tables" / "rows.parquet", index=False)
    pd.DataFrame(sets_rows).to_csv(res / "tables" / "sets.csv", index=False)
    meta = run_metadata(cfg, "A2.02_eval")
    meta["inputs"] = fingerprint([edir / "topk.npy", edir / "q_index.parquet", p(cfg, "a1_results", "tables", "pairs.csv"),
                                  p(cfg, "emb", "tag_incidence.npz"), p(cfg, "p9_emb", r, "pool.npy"), p(cfg, "p9_emb", rc, "pool.npy"),
                                  p(cfg, "a0_processed", "items.parquet")]
                                 + [a0 / x / f"meta_{part}.json" for x in sorted({r, rc}) for part in ("main", "subset") if (a0 / x / f"meta_{part}.json").exists()])
    meta["n_rows"], meta["minutes"] = len(df), (time.time() - t0) / 60
    save_json(meta, res / "meta_02.json")
    log.info(f"rows {len(df)} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
