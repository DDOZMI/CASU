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


B01 = _load("a13_01", Path(__file__).with_name("01_baselines.py"))


def auroc_t(dec, yy):
    yv = yy.cpu().numpy()
    n1, n0 = int(yv.sum()), int((~yv).sum())
    r = rankdata(dec.double().cpu().numpy(), axis=-1)
    return (r[..., yv].sum(-1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def fit_grid(Z, SW, ys, mask, Cs, max_iter):
    G, n, D = Z.shape
    M, nC = mask.shape[0], len(Cs)
    Cv = torch.as_tensor(Cs, device=DEV, dtype=torch.float32)
    W = torch.zeros(G, M, nC, D, device=DEV, requires_grad=True)
    b = torch.zeros(G, M, nC, device=DEV, requires_grad=True)
    ssum = SW.sum(1)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=max_iter, tolerance_grad=1e-9, tolerance_change=1e-12, history_size=20,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        Wm = W * mask[None, :, None, :]
        zz = torch.einsum("gnd,gmcd->gmcn", Z, Wm) + b[..., None]
        data = (SW[:, None, None, :] * torch.nn.functional.softplus(-ys[None, None, None, :] * zz)).sum(3) / ssum[:, None, None]
        reg = 0.5 * (Wm * Wm).sum(3) / (Cv[None, None, :] * ssum[:, None, None])
        loss = (data + reg).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        Wm = (W * mask[None, :, None, :]).detach()
        return torch.einsum("gnd,gmcd->gmcn", Z, Wm) + b.detach()[..., None], Wm, b.detach()


def order_cols(P, tr, y, rule):
    if rule == "evr":
        return torch.arange(P.shape[1], device=DEV)
    X = P[tr].double()
    yt = torch.as_tensor(y[tr], device=DEV)
    a, c = X[yt == 0], X[yt == 1]
    na, nc = a.shape[0], c.shape[0]
    sp = ((na - 1) * a.var(0) + (nc - 1) * c.var(0)) / (na + nc - 2)
    t = (a.mean(0) - c.mean(0)) / torch.sqrt(sp * (1 / na + 1 / nc) + 1e-30)
    return torch.argsort(-t.abs(), stable=True)


def run_one(rn, run, cfg, log, smoke_pairs=0):
    ts, cap = cfg["trisup"], cfg["cap"]
    K, m_grid, Cs, rules, F = ts["k_pcs"], ts["m_grid"], ts["C_grid"], ts["rules"], ts["inner_folds"]
    c_lit = Cs.index(ts["C_literal"])
    r = run["retriever"]
    edir = PROJECT_ROOT / run["emb"]
    sites = load_json(edir / "sites.json")
    sidx = {s: i for i, s in enumerate(sites)}
    qi = pd.read_parquet(edir / "q_index.parquet")
    rows_of = {kk: g.index.to_numpy() for kk, g in qi.groupby(["site", "role"])}
    a0 = p(cfg, "a0_emb", r)
    Q = B01.unit_t(np.concatenate([np.load(a0 / f"{s}__{ro}.npy") for s in sites for ro in ("test", "cal")]))
    assert Q.shape[0] == len(qi)
    Bd = B01.unit_t(np.concatenate([np.load(a0 / f"{s}__bank.npy") for s in sites]))
    assert Bd.shape[0] == len(sites) * cap
    assert K <= Bd.shape[1] and max(m_grid) <= K
    pairs = pd.read_csv(PROJECT_ROOT / run["a1"] / "tables" / "pairs.csv")[["kb", "ood", "tier"]]
    if smoke_pairs:
        pairs = pairs.iloc[:: max(1, len(pairs) // smoke_pairs)].head(smoke_pairs)
    cells13 = pd.read_parquet(PROJECT_ROOT / "experiments" / "A13" / "results" / "tables" / f"cells_{rn}.parquet",
                              columns=["kb", "ood", "score", "auroc"])
    abs13 = cells13[cells13.score == "abs"].set_index(["kb", "ood"]).auroc
    mask = torch.stack([(torch.arange(K, device=DEV) < m_).float() for m_ in m_grid])
    out, grid, kbcache, chk = [], [], {}, [0.0, 0]
    t0 = time.time()
    for pi_, (kb, o, tier) in enumerate(zip(pairs.kb, pairs.ood, pairs.tier)):
        if kb not in kbcache:
            kbcache.clear()
            Bk = Bd[sidx[kb] * cap:(sidx[kb] + 1) * cap]
            _, _, Vt = torch.linalg.svd((Bk - Bk.mean(0)).double(), full_matrices=False)
            V = Vt[:K].float()
            i_cal = rows_of[(kb, "cal")]
            pool = np.concatenate([rows_of[(s, "cal")] for s in sites if s != kb])
            rng = np.random.default_rng(stable_seed(cfg["seed"], "A13_trisup_neg", rn, kb))
            i_neg = np.sort(rng.choice(pool, size=len(i_cal), replace=False))
            y = np.r_[np.zeros(len(i_cal)), np.ones(len(i_neg))].astype(int)
            n_tr = len(y)
            P = Q[torch.as_tensor(np.r_[i_cal, i_neg], device=DEV)] @ V.T
            skf = StratifiedKFold(F, shuffle=True, random_state=stable_seed(cfg["seed"], "A13_trisup_cv", rn, kb))
            splits = list(skf.split(np.zeros(n_tr), y))
            fits = [tr for tr, _ in splits] + [np.arange(n_tr)]
            Z, SW, tf = [], [], []
            for rule in rules:
                for tr in fits:
                    cols = order_cols(P, tr, y, rule)
                    Pt = P[:, cols]
                    trt = torch.as_tensor(tr, device=DEV)
                    mu_, sd_ = Pt[trt].mean(0), Pt[trt].std(0) + 1e-8
                    Z.append((Pt - mu_) / sd_)
                    tf.append((cols, mu_, sd_))
                    n1, n0 = float((y[tr] == 1).sum()), float((y[tr] == 0).sum())
                    sw = np.zeros(n_tr, np.float32)
                    sw[tr] = np.where(y[tr] == 1, len(tr) / (2 * n1), len(tr) / (2 * n0))
                    SW.append(sw)
            ys = torch.as_tensor(2 * y - 1, device=DEV, dtype=torch.float32)
            dec, W, b = fit_grid(torch.stack(Z), torch.as_tensor(np.stack(SW), device=DEV), ys, mask, Cs, ts["max_iter"])
            dec = dec.view(len(rules), F + 1, len(m_grid), len(Cs), n_tr)
            cv = np.zeros((len(rules), len(m_grid), len(Cs)))
            for f, (_, te) in enumerate(splits):
                cv += auroc_t(dec[:, f][..., torch.as_tensor(te, device=DEV)], torch.as_tensor(y[te] == 1, device=DEV))
            cv /= F
            full = [(tf[a * (F + 1) + F], W[a * (F + 1) + F], b[a * (F + 1) + F]) for a in range(len(rules))]
            kbcache[kb] = (Bk, V, cv, full, i_neg)
        Bk, V, cv, full, i_neg = kbcache[kb]
        i_id, i_ood = rows_of[(kb, "test")], rows_of[(o, "test")]
        n_from_o = int(np.isin(i_neg, rows_of[(o, "cal")]).sum())
        Xte = Q[torch.as_tensor(np.r_[i_id, i_ood], device=DEV)]
        yte = torch.as_tensor(np.r_[np.zeros(len(i_id)), np.ones(len(i_ood))] == 1, device=DEV)
        s_abs = (Xte @ Bk.T).topk(cfg["k"], dim=1).values.double().mean(1)
        a_abs = float(auroc_t(-s_abs, yte))
        chk = [max(chk[0], abs(a_abs - abs13[(kb, o)])), chk[1] + 1]
        Pte = Xte @ V.T
        ten = np.stack([auroc_t(torch.einsum("nd,mcd->mcn", (Pte[:, cols] - mu_) / sd_, Wr) + br[..., None], yte)
                        for (cols, mu_, sd_), Wr, br in full])
        base = {"run": rn, "kb": kb, "ood": o, "tier": tier, "auroc_abs": a_abs, "n_neg_from_O": n_from_o}

        def pick(name, sub, idx_map, src):
            k_ = np.unravel_index(int(np.argmax(sub)), sub.shape)
            a, b_, c = idx_map(k_)
            out.append({**base, "score": name, "auroc": float(ten[a, b_, c]), "rule": rules[a], "m": m_grid[b_], "C": Cs[c],
                        "cv_auroc": float(cv[a, b_, c]), "selected_by": src})
        pick("tri_sup_cv", cv, lambda k_: k_, "inner CV over rule x m x C")
        for a, rule in enumerate(rules):
            pick(f"tri_sup_{rule}_c{ts['C_literal']:g}", cv[a, :, c_lit], lambda k_, a=a: (a, k_[0], c_lit), "inner CV over m")
        pick("tri_sup_max", ten, lambda k_: k_, "test AUROC (envelope)")
        for a, rule in enumerate(rules):
            for b_, m_ in enumerate(m_grid):
                for c, C in enumerate(Cs):
                    grid.append({"kb": kb, "ood": o, "rule": rule, "m": m_, "C": C, "test_auroc": float(ten[a, b_, c]), "cv_auroc": float(cv[a, b_, c])})
        if pi_ % 50 == 0:
            log.info(f"{rn}: pair {pi_}/{len(pairs)} ({(time.time() - t0) / 60:.1f} min)")
    log.info(f"{rn}: check abs AUROC vs A13 cells: max |diff| = {chk[0]:.2e} over {chk[1]}")
    assert chk[1] == len(pairs) and chk[0] < 1e-12, chk
    return pd.DataFrame(out), pd.DataFrame(grid), chk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--smoke-pairs", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    assert not args.smoke_pairs or args.out, "--smoke-pairs requires --out"
    cfg = load_config(args.config)
    res = Path(args.out) if args.out else PROJECT_ROOT / "experiments" / "A13" / "results_trisup"
    (res / "tables").mkdir(parents=True, exist_ok=True)
    log = get_logger("A13.06", res / "logs" / "06_trisup.log")
    for rn in args.runs or list(cfg["runs"]):
        run = cfg["runs"][rn]
        t0 = time.time()
        D, Gd, chk = run_one(rn, run, cfg, log, args.smoke_pairs)
        D.to_parquet(res / "tables" / f"cells_trisup_{rn}.parquet", index=False)
        Gd.to_parquet(res / "tables" / f"grid_trisup_{rn}.parquet", index=False)
        meta = run_metadata(cfg, f"A13.06_trisup:{rn}")
        r = run["retriever"]
        ins = [PROJECT_ROOT / run["emb"] / f for f in ("q_index.parquet", "sites.json")] + [PROJECT_ROOT / run["a1"] / "tables" / "pairs.csv"]
        ins += [PROJECT_ROOT / "experiments" / "A13" / "results" / "tables" / f"cells_{rn}.parquet"]
        ins += [p(cfg, "a0_emb", r, f"{s_}__{ro}.npy") for s_ in load_json(PROJECT_ROOT / run["emb"] / "sites.json") for ro in ("test", "cal", "bank")]
        meta["inputs"] = fingerprint(ins)
        meta["imported_code_sha256"] = fingerprint([Path(__file__).with_name("01_baselines.py")])
        meta["checks"] = {"abs_auroc_vs_a13": {"max_abs_diff": chk[0], "n": chk[1]}}
        meta["n_rows"], meta["minutes"], meta["smoke_pairs"] = len(D), (time.time() - t0) / 60, args.smoke_pairs
        save_json(meta, res / f"meta_06_{rn}.json")
        log.info(f"{rn}: {len(D)} rows in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
