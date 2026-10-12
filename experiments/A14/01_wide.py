from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import PROJECT_ROOT, fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json, stable_seed


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A1 = _load("a1_eval", PROJECT_ROOT / "experiments" / "A1" / "02_eval.py")
D3 = _load("a3_diag", PROJECT_ROOT / "experiments" / "A3" / "02_diagnose.py")
G_ = {}


def factor(Lsel, nt, U):
    X = Lsel[:, U]
    cal = X[nt:]
    mu, sd = cal.mean(0), cal.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    _, _, vt = np.linalg.svd(Z[nt:] - Z[nt:].mean(0), full_matrices=False)
    w = vt[0] * (1 if vt[0].sum() >= 0 else -1)
    return Z @ w


def partial(s, c, g):
    R = np.corrcoef(np.vstack([s, c, g]))
    return (R[0, 1] - R[0, 2] * R[1, 2]) / np.sqrt((1 - R[0, 2] ** 2) * (1 - R[1, 2] ** 2))


def eval_kb(kb):
    cfg, sites, sidx, rows_of, Lm, T = G_["cfg"], G_["sites"], G_["sidx"], G_["rows_of"], G_["Lm"], G_["T"]
    k, R, acc, a = cfg["k"], cfg["R"], cfg["metrics"]["op_accept"], cfg["scores"]["alpha"]
    D, rankO, excl = G_["D"], G_["rankO"], G_["excl"][kb]
    rankOK = {(o, kb): {x: r + 1 for r, x in enumerate(sorted((y for y in G_["sites"] if y not in (o, kb)), key=lambda y: (D[G_["sidx"][o], G_["sidx"][y]], y)))}
              for o, _ in G_["pairs_of"][kb]}
    out = []
    i_id_, i_cal_ = rows_of[(kb, "test")], rows_of[(kb, "cal")]
    g_id = factor(Lm[np.r_[i_id_, i_cal_]], len(i_id_), [j for j in range(len(sites)) if j != sidx[kb]])[: len(i_id_)]
    for o, tier in G_["pairs_of"][kb]:
        i_id, i_ood, i_cal = i_id_, rows_of[(o, "test")], i_cal_
        sel = np.r_[i_id, i_ood, i_cal]
        n1, n2 = len(i_id), len(i_id) + len(i_ood)
        Lsel, Tsel = Lm[sel], T[sel]
        s = Lsel[:, sidx[kb]]
        absk = Tsel[:, sidx[kb], k - 1].astype(np.float64)
        elig = sorted(x for x in sites if x not in (kb, o))
        for d in range(R):
            perm = [elig[i] for i in np.random.default_rng(stable_seed(cfg["seed"], "A14_order", kb, o, d)).permutation(len(elig))]
            orders = {"noexcl": perm, "excl": [x for x in perm if x not in excl]}
            for arm, order in orders.items():
                for m in G_["m_grid"]:
                    if m == "all":
                        if d > 0:
                            continue
                        Cn = sorted(order)
                    else:
                        if m > len(order):
                            continue
                        Cn = order[:m]
                    Ci = [sidx[x] for x in Cn]
                    A = Lsel[:, Ci]
                    U = Tsel[:, Ci, :].reshape(len(sel), -1).astype(np.float64)
                    top = -np.partition(-U, k - 1, axis=1)[:, :k]
                    pm, pk = top.mean(1), top.min(1)
                    terms = {f"S_tau@{t}": (A.mean(1) if t == "inf" else float(t) * (logsumexp(A / float(t), axis=1) - np.log(len(Ci))))
                             for t in cfg["scores"]["taus"]}
                    terms[f"mix@{a:g}"] = a * pm
                    nn = min(rankOK[(o, kb)][x] for x in Cn)
                    nn_all = min(rankO[o][x] for x in Cn)
                    base = {"kb": kb, "ood": o, "tier": tier, "arm": arm, "m": str(m), "n_set": len(Ci), "draw": d,
                            "nn_rank": nn, "nn_rank_all": nn_all, "nn_d": float(min(D[sidx[o], sidx[x]] for x in Cn)), "n_excl": len(excl)}
                    for nm, c in terms.items():
                        S = s - c
                        mt = A1.metrics(S[:n1], S[n1:n2], S[n2:], acc)
                        mo = D3.moments(s[:n1], s[n1:n2], c[:n1], c[n1:n2])
                        out.append({**base, "score": nm, **mt, "pi": mo["pi"], "T_var": mo["T_var"], "T_push": mo["T_push"], "dd_m": mo["dd_m"],
                                    "rho_id": float(np.corrcoef(s[:n1], c[:n1])[0, 1]), "rho_ood": float(np.corrcoef(s[n1:n2], c[n1:n2])[0, 1]),
                                    "partial_id": float(partial(s[:n1], c[:n1], g_id))})
                    S = -np.sqrt(np.clip(2 - 2 * absk, 0, None)) + a * np.sqrt(np.clip(2 - 2 * pk, 0, None))
                    out.append({**base, "score": f"flatsp@{a:g}", **A1.metrics(S[:n1], S[n1:n2], S[n2:], acc)})
    return out


def run_one(rn, run, cfg, log, workers, smoke_kbs=0):
    edir = PROJECT_ROOT / run["emb"]
    sites = load_json(edir / "sites.json")
    sidx = {s: i for i, s in enumerate(sites)}
    T = np.asarray(np.load(edir / "topk.npy", mmap_mode="r"))
    Lm = T.astype(np.float64).mean(2)
    qi = pd.read_parquet(edir / "q_index.parquet")
    rows_of = {kk: g.index.to_numpy() for kk, g in qi.groupby(["site", "role"])}
    D = pd.read_csv(p(cfg, "a0_processed", "tag_js.csv"), index_col=0).loc[sites, sites].to_numpy()
    rankO = {o: {x: r + 1 for r, x in enumerate(sorted((y for y in sites if y != o), key=lambda y: (D[sidx[o], sidx[y]], y)))} for o in sites}
    crowd = {(kb, j): float(Lm[rows_of[(kb, "cal")], sidx[j]].mean()) for kb in sites for j in sites if j != kb}
    thr = float(np.quantile(list(crowd.values()), cfg["exclusion"]["quantile"]))
    excl = {kb: {j for j in sites if j != kb and crowd[(kb, j)] >= thr} for kb in sites}
    pd.DataFrame([{"kb": kb, "site": j, "crowd": crowd[(kb, j)], "excluded": crowd[(kb, j)] >= thr} for kb in sites for j in sites if j != kb]).to_csv(
        G_["res"] / "tables" / f"excluded_{rn}.csv", index=False)
    log.info(f"{rn}: common threshold {thr:.4f}; excluded sites per KB 5/50/95% " + " / ".join(f"{v:.0f}" for v in np.quantile([len(v) for v in excl.values()], [.05, .5, .95])))
    pairs = pd.read_csv(PROJECT_ROOT / run["a1"] / "tables" / "pairs.csv")[["kb", "ood", "tier"]]
    kbs = list(dict.fromkeys(pairs.kb))[: smoke_kbs or None]
    pairs_of = {kb: list(zip(g.ood, g.tier)) for kb, g in pairs.groupby("kb")}
    G_.update({"cfg": cfg, "sites": sites, "sidx": sidx, "rows_of": rows_of, "Lm": Lm, "T": T, "D": D, "rankO": rankO, "excl": excl,
               "pairs_of": pairs_of, "m_grid": run["m_grid"]})
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, o_ in enumerate(ex.map(eval_kb, kbs)):
            rows += o_
            if i % 10 == 0:
                log.info(f"{rn}: KB {i + 1}/{len(kbs)}")
    return pd.DataFrame(rows), thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--smoke-kbs", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    assert not args.smoke_kbs or args.out, "--smoke-kbs requires --out"
    cfg = load_config(args.config)
    res = Path(args.out) if args.out else p(cfg, "results")
    (res / "tables").mkdir(parents=True, exist_ok=True)
    G_["res"] = res
    log = get_logger("A14.01", res / "logs" / "01_wide.log")
    for rn in args.runs or list(cfg["runs"]):
        run = cfg["runs"][rn]
        t0 = time.time()
        Dd, thr = run_one(rn, run, cfg, log, args.workers, args.smoke_kbs)
        Dd.to_parquet(res / "tables" / f"rows_{rn}.parquet", index=False)
        meta = run_metadata(cfg, f"A14.01_wide:{rn}")
        meta["inputs"] = fingerprint([PROJECT_ROOT / run["emb"] / f for f in ("topk.npy", "q_index.parquet", "sites.json")]
                                     + [PROJECT_ROOT / run["a1"] / "tables" / "pairs.csv", p(cfg, "a0_processed", "tag_js.csv")])
        meta["imported_code_sha256"] = fingerprint([PROJECT_ROOT / "experiments" / "A1" / "02_eval.py", PROJECT_ROOT / "experiments" / "A3" / "02_diagnose.py"])
        meta["threshold"], meta["n_rows"], meta["minutes"], meta["smoke_kbs"] = thr, len(Dd), (time.time() - t0) / 60, args.smoke_kbs
        save_json(meta, res / f"meta_01_{rn}.json")
        log.info(f"{rn}: {len(Dd)} rows in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
