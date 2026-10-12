from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json, stable_seed

G_ = {}


def auroc(neg, pos):
    r = rankdata(np.concatenate([neg, pos]))
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def metrics(S_id, S_ood, S_cal, acc, ov, abs_rej=None):
    A = auroc(-S_id, -S_ood)
    t_cal = np.quantile(S_cal, 1 - acc, method="lower")
    rej = S_id < t_cal
    out = {"auroc": A, "op_id_acc": float(np.mean(~rej)), "op_ood_acc": float(np.mean(S_ood >= t_cal)),
           "mu_id": float(S_id.mean()), "sd_id": float(S_id.std()), "mu_ood": float(S_ood.mean())}
    if ov is not None:
        out["fr_overlap_share"] = float(ov[rej].mean()) if rej.any() else np.nan
        out["base_overlap_share"] = float(ov.mean())
        out["fr_overlap_share_abs"] = float(ov[abs_rej].mean()) if abs_rej is not None and abs_rej.any() else np.nan
    return out


def eval_kb(kb):
    cfg, sidx, rows_of = G_["cfg"], G_["sidx"], G_["rows_of"]
    Lm, Lk, L3m, L3k, T, T3 = G_["Lm"], G_["Lk"], G_["L3m"], G_["L3k"], G_["T"], G_["T3"]
    sc, k, acc, m = cfg["scores"], cfg["k"], cfg["metrics"]["id_accept"], 4
    out = []
    qtags = G_["qtags"]
    for o, draws in G_["plans"].get(kb, []):
        i_id, i_ood, i_cal = rows_of[(kb, "test")], rows_of[(o, "test")], rows_of[(kb, "cal")]
        sel = np.r_[i_id, i_ood, i_cal]
        n1, n2 = len(i_id), len(i_id) + len(i_ood)
        base = {"abs": Lm[sel, sidx[kb]].astype(np.float64), "absk": Lk[sel, sidx[kb]].astype(np.float64)}
        bm = {b: metrics(v[:n1], v[n1:n2], v[n2:], acc, None) for b, v in base.items()}
        Tsel, Lsel = T[sel], Lm[sel]
        rej_abs = {b: v[:n1] < np.quantile(v[n2:], 1 - acc, method="lower") for b, v in base.items()}
        for d in draws:
            Fx = [sidx[x] for x in d["fixed"]]
            for lev, site in d["levels"].items():
                if lev == "L3":
                    col_m, col_k, Tl = L3m[sel, sidx[kb]], L3k[sel, sidx[kb]], T3[sel, sidx[kb]]
                    ov = None
                else:
                    col_m, col_k, Tl = Lm[sel, sidx[site]], Lk[sel, sidx[site]], T[sel, sidx[site]]
                    st = G_["bank_tags"][site]
                    ov = np.array([len(qtags[i] & st) > 0 for i in i_id])
                sA = np.column_stack([Lsel[:, Fx], col_m]).astype(np.float64)
                S = {}
                for t in sc["tau_grid"]:
                    S[("S_tau", str(t))] = base["abs"] - (sA.mean(1) if t == "inf" else t * (logsumexp(sA / t, axis=1) - np.log(m)))
                U = np.concatenate([Tsel[:, Fx, :].reshape(len(sel), -1), Tl], axis=1).astype(np.float64)
                top = -np.partition(-U, k - 1, axis=1)[:, :k]
                pm, pk = top.mean(1), top.min(1)
                for a in sc["alpha_grid"]:
                    S[("mix", str(a))] = base["abs"] - a * pm
                    S[("mixk", str(a))] = base["absk"] - a * pk
                for (fam, par), v in S.items():
                    b = "absk" if fam == "mixk" else "abs"
                    mt = metrics(v[:n1], v[n1:n2], v[n2:], acc, ov, rej_abs[b])
                    out.append({"kb": kb, "ood": o, "draw": d["draw"], "level": lev, "slot": site, "level_fallback": d["fallback"][lev],
                                "family": fam, "param": par, **mt, "auroc_abs": bm[b]["auroc"], "op_ood_acc_abs": bm[b]["op_ood_acc"],
                                "op_id_acc_abs": bm[b]["op_id_acc"], "mu_id_abs": bm[b]["mu_id"], "sd_id_abs": bm[b]["sd_id"],
                                "mu_ood_abs": bm[b]["mu_ood"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--smoke-kbs", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = Path(args.out) if args.out else p(cfg, "results")
    (res / "tables").mkdir(parents=True, exist_ok=True)
    log = get_logger("A4.02", res / "logs" / "02_eval.log")
    t0 = time.time()
    r, k = cfg["detector"], cfg["k"]
    a1 = p(cfg, "a1_emb", r)
    sites = load_json(a1 / "sites.json")
    sidx = {s: i for i, s in enumerate(sites)}
    qi = pd.read_parquet(a1 / "q_index.parquet")
    rows_of = {kk: g.index.to_numpy() for kk, g in qi.groupby(["site", "role"])}
    T = np.asarray(np.load(a1 / "topk.npy", mmap_mode="r"))
    T3 = np.asarray(np.load(p(cfg, "emb", r, "topk_l3.npy"), mmap_mode="r"))
    assert T.shape == T3.shape
    D = pd.read_csv(p(cfg, "a0_processed", "tag_js.csv"), index_col=0).loc[sites, sites].to_numpy()
    bands = load_json(p(cfg, "a0_processed", "bands.json"))
    q_nb, q_far = bands["q_neighbor_value"], bands["q_far_value"]
    cat = {s: v["category"] for s, v in load_json(p(cfg, "a0_processed", "sites.json")).items()}
    items = pd.read_parquet(p(cfg, "a0_processed", "items.parquet"), columns=["site", "role", "pos", "tags"])
    bank_tags = {s: set().union(*[set(json.loads(t)) for t in g.tags]) for s, g in items[items.role == "bank"].groupby("site")}
    qt = items[items.role.isin(["test", "cal"])].set_index(["site", "role", "pos"]).tags
    qtags = [set(json.loads(qt[(s, ro, j)])) for s, ro, j in zip(qi.site, qi.role, qi.pos)]

    dr = pd.read_csv(p(cfg, "a1_results", "tables", "draws.csv"))
    fx = dr.drop_duplicates(["kb", "ood", "draw"])[["kb", "ood", "draw", "fixed"]]
    kbs = sites[: args.smoke_kbs] if args.smoke_kbs else sites
    nb = lambda x, y: cat[x] == cat[y] and D[sidx[x], sidx[y]] <= q_nb
    plans, lev_rows = {}, []
    for (kb, o), g in fx.groupby(["kb", "ood"], sort=False):
        if kb not in kbs:
            continue
        dK = D[sidx[kb]]
        draws = []
        for d, fixed in zip(g.draw, g.fixed):
            fixed = fixed.split("+")
            E = [s for s in sites if s not in (kb, o) and s not in fixed and not nb(s, o)]
            same = sorted((s for s in E if cat[s] == cat[kb]), key=lambda s: (dK[sidx[s]], s))
            bands_ = {"L0": [s for s in E if cat[s] != cat[kb] and dK[sidx[s]] > q_far],
                      "L1": [s for s in same if q_nb < dK[sidx[s]] <= q_far],
                      "L2": [s for s in same if dK[sidx[s]] <= q_nb]}
            rng = np.random.default_rng(stable_seed(cfg["seed"], "A4_levels", kb, o, d))
            levels, fb = {}, {}
            for lev in ("L0", "L1", "L2"):
                cand = sorted(bands_[lev])
                fb[lev] = len(cand) == 0
                if cand:
                    levels[lev] = cand[int(rng.integers(len(cand)))]
                elif lev == "L2" and same:
                    levels[lev] = same[0]
                elif lev == "L1" and same:
                    levels[lev] = same[len(same) // 2]
                else:
                    levels[lev] = None
            levels["L3"], fb["L3"] = kb, False
            levels = {kk: v for kk, v in levels.items() if v is not None}
            draws.append({"draw": int(d), "fixed": fixed, "levels": levels, "fallback": fb})
            for lev, site in levels.items():
                lev_rows.append({"kb": kb, "ood": o, "draw": int(d), "level": lev, "slot": site, "fallback": fb[lev],
                                 "dtag_slot_kb": float(D[sidx[kb], sidx[site]]), "dtag_slot_o": float(D[sidx[o], sidx[site]])})
        plans.setdefault(kb, []).append((o, draws))
    pd.DataFrame(lev_rows).to_csv(res / "tables" / "levels.csv", index=False)
    G_.update({"cfg": cfg, "sidx": sidx, "rows_of": rows_of, "T": T, "T3": T3, "Lm": T.mean(2), "Lk": T[:, :, k - 1],
               "L3m": T3.mean(2), "L3k": T3[:, :, k - 1], "plans": plans, "bank_tags": bank_tags, "qtags": qtags})
    log.info(f"planning {time.time() - t0:.0f}s; pairs {sum(len(v) for v in plans.values())}")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for out in ex.map(eval_kb, kbs):
            rows += out
    df = pd.DataFrame(rows)
    df.to_parquet(res / "tables" / "rows.parquet", index=False)
    meta = run_metadata(cfg, "A4.02_eval")
    meta["inputs"] = fingerprint([a1 / "topk.npy", p(cfg, "emb", r, "topk_l3.npy"), a1 / "q_index.parquet", p(cfg, "a1_results", "tables", "draws.csv"),
                                  p(cfg, "a0_processed", "tag_js.csv"), p(cfg, "a0_processed", "bands.json"), p(cfg, "a0_processed", "sites.json"),
                                  p(cfg, "a0_processed", "items.parquet")])
    meta["n_rows"], meta["minutes"] = len(df), (time.time() - t0) / 60
    save_json(meta, res / "meta_02.json")
    log.info(f"rows {len(df)} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
