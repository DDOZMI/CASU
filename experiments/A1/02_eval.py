from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json, stable_seed

G_ = {}


def auroc(neg, pos):
    r = rankdata(np.concatenate([neg, pos]))
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


CLIP = 1e-4


def dprime(a):
    return np.sqrt(2) * norm.ppf(np.clip(a, CLIP, 1 - CLIP))


def metrics(S_id, S_ood, S_cal, acc):
    A = auroc(-S_id, -S_ood)
    t_or = np.sort(S_id)[int(np.floor((1 - acc) * len(S_id)))]
    t_cal = np.quantile(S_cal, 1 - acc, method="lower")
    return {"auroc": A, "fpr95": float(np.mean(S_ood >= t_or)), "op_id_acc": float(np.mean(S_id >= t_cal)),
            "op_ood_acc": float(np.mean(S_ood >= t_cal)), "mu_id": float(S_id.mean()), "sd_id": float(S_id.std()),
            "mu_ood": float(S_ood.mean())}


def plan_pair(kb, o, cfg, D, cat, sidx, q_nb, q_far):
    sites = G_["sites"]
    cc = cfg["comparison"]
    m, R = cc["m"], cc["R"]
    dO, dK = D[sidx[o]], D[sidx[kb]]
    same_cat = cfg.get("nb_same_category", True)
    nb = lambda x, y: (cat[x] == cat[y] or not same_cat) and D[sidx[x], sidx[y]] <= q_nb
    tw = G_.get("twin", {})
    U = [s for s in sites if s not in (kb, o) and s not in (tw.get(kb), tw.get(o)) and s not in tw]
    pool = sorted(s for s in U if dO[sidx[s]] > q_far and dK[sidx[s]] > q_far and cat[s] != cat[o])
    combos = list(combinations(pool, m - 1))
    if len(combos) == 0:
        return None, {"kb": kb, "ood": o, "fixed_pool": len(pool), "n_draws": 0, "dropped": True}
    rng = np.random.default_rng(stable_seed(cfg["seed"], "A1_fixed", kb, o))
    draws = []
    for di, ci in enumerate(rng.choice(len(combos), size=min(R, len(combos)), replace=False)):
        fixed = combos[ci]
        E = [s for s in U if s not in fixed]
        byO = sorted(E, key=lambda s: (dO[sidx[s]], s))
        byK = sorted(E, key=lambda s: (dK[sidx[s]], s))
        roles = {"O": o}
        Ns = [s for s in byO if not nb(s, kb)][:2]
        roles["N1"], roles["N2"] = Ns
        roles["M"] = byO[len(byO) // 2]
        zpool = sorted(s for s in E if dO[sidx[s]] > q_far and dK[sidx[s]] > q_far and s not in Ns)
        z_relaxed = 0
        if len(zpool) < 2:
            z_relaxed = 1
            zpool = sorted(s for s in E if dO[sidx[s]] > q_far and s not in Ns)
        if len(zpool) < 2:
            z_relaxed = 2
            zpool = sorted(byO[-2:])
        rz = np.random.default_rng(stable_seed(cfg["seed"], "A1_Z", kb, o, di))
        roles["Z1"], roles["Z2"] = [zpool[i] for i in rz.choice(len(zpool), size=2, replace=False)]
        rkall = {s: i for i, s in enumerate(sorted((x for x in sites if x != kb and x not in tw and x != tw.get(kb)), key=lambda x: (dK[sidx[x]], x)))}
        zm = sorted((s for s in E if dO[sidx[s]] > q_far and s not in Ns), key=lambda s: (abs(rkall[s] - rkall[Ns[0]]), s))
        if not zm:
            zm = sorted((s for s in byO[len(byO) // 2:] if s not in Ns), key=lambda s: (abs(rkall[s] - rkall[Ns[0]]), s))
        roles["Zm"] = zm[0]
        zm_relaxed = not (dO[sidx[zm[0]]] > q_far)
        roles["K"] = [s for s in byK if not nb(s, o) and s not in Ns][0]
        rO = {s: i for i, s in enumerate(byO)}
        rK = {s: i for i, s in enumerate(byK)}
        used = set(roles.values())
        nk = sorted((s for s in E if s not in used), key=lambda s: (rO[s] + rK[s], s))
        roles["NK"] = nk[0] if nk else None
        for rn, rule in G_.get("extra_roles", {}).items():
            assert rule == "twin_of_ood" and tw, rule
            roles[rn] = tw[o]
        draws.append({"draw": di, "fixed": list(fixed), "roles": roles, "z_relaxed": z_relaxed, "zm_relaxed": zm_relaxed})
    return draws, {"kb": kb, "ood": o, "fixed_pool": len(pool), "n_draws": len(draws), "dropped": False}


def eval_kb(kb):
    cfg, sites, sidx, rows_of = G_["cfg"], G_["sites"], G_["sidx"], G_["rows_of"]
    Lm, Lk, T = G_["Lm"], G_["Lk"], G_["T"]
    sc, k = cfg["scores"], cfg["k"]
    acc, ceil = cfg["metrics"]["id_accept"], cfg["metrics"]["ceiling"]
    m = cfg["comparison"]["m"]
    out = []
    for o, draws in G_["plans"][kb]:
        i_id, i_ood, i_cal = rows_of[(kb, "test")], rows_of[(o, "test")], rows_of[(kb, "cal")]
        if G_.get("id_keep") is not None:
            i_id = i_id[G_["id_keep"][i_id]]
        sel = np.r_[i_id, i_ood, i_cal]
        n1, n2 = len(i_id), len(i_id) + len(i_ood)
        base = {"abs": Lm[sel, sidx[kb]].astype(np.float64), "absk": Lk[sel, sidx[kb]].astype(np.float64)}
        bm = {b: metrics(v[:n1], v[n1:n2], v[n2:], acc) for b, v in base.items()}
        Tsel = T[sel]
        Lsel = Lm[sel].astype(np.float64)
        for d in draws:
            cache = {}
            for role, c in d["roles"].items():
                if c is None:
                    continue
                if c not in cache:
                    C = [sidx[x] for x in (*d["fixed"], c)]
                    sA = Lsel[:, C]
                    S = {}
                    for t in sc["tau_grid"]:
                        S[("S_tau", str(t))] = base["abs"] - (sA.mean(1) if t == "inf" else t * (logsumexp(sA / t, axis=1) - np.log(m)))
                    U = Tsel[:, C, :].reshape(len(sel), -1).astype(np.float64)
                    top = -np.partition(-U, k - 1, axis=1)[:, :k]
                    pm, pk = top.mean(1), top.min(1)
                    for a in sc["alpha_grid"]:
                        S[("mix", str(a))] = base["abs"] - a * pm
                        S[("mixk", str(a))] = base["absk"] - a * pk
                    cache[c] = {key: metrics(v[:n1], v[n1:n2], v[n2:], acc) for key, v in S.items()}
                for (fam, par), mt in cache[c].items():
                    b = "absk" if fam == "mixk" else "abs"
                    A0 = bm[b]["auroc"]
                    out.append({"kb": kb, "ood": o, "draw": d["draw"], "role": role, "slot": c, "family": fam, "param": par,
                                **mt, "auroc_abs": A0, "fpr95_abs": bm[b]["fpr95"], "op_ood_acc_abs": bm[b]["op_ood_acc"],
                                "op_id_acc_abs": bm[b]["op_id_acc"],
                                "ddp": float(dprime(mt["auroc"]) - dprime(A0)),
                                "G": (mt["auroc"] - A0) / (1 - A0) if A0 <= ceil else np.nan})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--smoke-kbs", type=int, default=0, help="smoke: only the first n KBs, outputs to --out")
    ap.add_argument("--out", default=None)
    ap.add_argument("--emb-smoke", action="store_true", help="read the smoke top-k (data/emb/A1/_smoke)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--retriever", default=None, help="default: config detector")
    args = ap.parse_args()
    cfg = load_config(args.config)
    global CLIP
    CLIP = cfg["metrics"]["dprime_clip"]
    r = args.retriever or cfg["detector"]
    res = Path(args.out) if args.out else (p(cfg, "results", r) if cfg.get("results_per_retriever") else p(cfg, "results"))
    (res / "tables").mkdir(parents=True, exist_ok=True)
    log = get_logger("A1.02", res / "logs" / "02_eval.log")
    t0 = time.time()
    edir = p(cfg, "emb", "_smoke" if args.emb_smoke else "", r)
    sites = load_json(edir / "sites.json")
    sidx = {s: i for i, s in enumerate(sites)}
    qi = pd.read_parquet(edir / "q_index.parquet")
    T = np.load(edir / "topk.npy", mmap_mode="r")
    T = np.asarray(T)
    k = cfg["k"]
    assert T.shape == (len(qi), len(sites), k)
    rows_of = {key: g.index.to_numpy() for key, g in qi.groupby(["site", "role"])}
    Dfull = pd.read_csv(p(cfg, "a0_processed", cfg.get("distance_file", "tag_js.csv")), index_col=0)
    D = Dfull.loc[sites, sites].to_numpy()
    bands = load_json(p(cfg, "a0_processed", "bands.json"))
    q_nb, q_far = bands["q_neighbor_value"], bands["q_far_value"]
    cat = {s: v["category"] for s, v in load_json(p(cfg, "a0_processed", "sites.json")).items()}

    G_.update({"cfg": cfg, "sites": sites, "sidx": sidx, "rows_of": rows_of, "T": T, "Lm": T.mean(2), "Lk": T[:, :, k - 1]})
    pc = cfg["pairs"]
    if pc.get("extra_roles"):
        assert pc.get("mode") == "set"
        G_["extra_roles"] = {k_: v for k_, v in pc["extra_roles"].items() if not k_.startswith("_")}
    sub_path = None
    if cfg.get("id_subset"):
        sub_path = Path(cfg["id_subset"]["path"].format(retriever=r))
        sub_path = sub_path if sub_path.is_absolute() else Path(__file__).resolve().parents[2] / sub_path
        sub = pd.read_parquet(sub_path)
        assert (sub[["site", "role", "pos"]].to_numpy() == qi[["site", "role", "pos"]].to_numpy()).all()
        G_["id_keep"] = sub[cfg["id_subset"]["column"]].to_numpy(bool)
        nk_ = {s_: int(G_["id_keep"][rows_of[(s_, "test")]].sum()) for s_ in pc["kb_set"]}
        assert min(nk_.values()) > 0, nk_
        log.info(f"id_subset {cfg['id_subset']['column']}: kept ID test queries per KB {nk_}")
    set_mode = pc.get("mode") == "set"
    if set_mode:
        G_["twin"] = {c: v["twin"] for c, v in load_json(p(cfg, "a0_processed", "sites.json")).items() if "twin" in v}
        kbs = list(pc["kb_set"])[: args.smoke_kbs or None]
    else:
        kbs = sites[: args.smoke_kbs] if args.smoke_kbs else sites
    plans, pair_rows, draw_rows = {}, [], []
    for kb in kbs:
        others = sorted((s for s in sites if s != kb), key=lambda s: (D[sidx[kb], sidx[s]], s))
        rng = np.random.default_rng(stable_seed(cfg["seed"], "A1_pairs", kb))
        plans[kb] = []
        if set_mode:
            tw = G_["twin"]
            others = sorted((s for s in sites if s not in tw and s != tw[kb]), key=lambda s: (D[sidx[kb], sidx[s]], s))
            strata = pc["rank_strata"]
            sel_pairs = []
            for o_ in pc["kb_set"]:
                if o_ == kb:
                    continue
                jj = others.index(tw[o_])
                sel_pairs.append((next(i for i, (lo, hi) in enumerate(strata) if lo <= jj + 1 <= hi), jj, o_))
        elif pc.get("mode") == "all":
            sel_pairs = [(next(i for i, (lo, hi) in enumerate(pc["tier_ranks"]) if lo <= j + 1 <= hi), j) for j in range(len(others))]
            strata = [(lo, hi) for lo, hi in pc["tier_ranks"]]
        else:
            strata = pc["rank_strata"]
            sel_pairs = []
            for si, (lo, hi) in enumerate(strata):
                cand_n = len(others[lo - 1: hi])
                sel_pairs += [(si, lo - 1 + int(j)) for j in rng.choice(cand_n, size=min(pc["per_stratum"], cand_n), replace=False)]
        for sp in sel_pairs:
            si, jj = sp[0], sp[1]
            lo = strata[si][0]
            j = jj - (lo - 1)
            if True:
                o = sp[2] if set_mode else others[jj]
                draws, info = plan_pair(kb, o, cfg, D, cat, sidx, q_nb, q_far)
                info.update({"tier": ["near", "mid", "far"][si], "rank_kb_o": lo + int(j), "dtag_kb_o": D[sidx[kb], sidx[o]],
                             "kb_cat": cat[kb], "ood_cat": cat[o]})
                pair_rows.append(info)
                if draws is None:
                    continue
                plans[kb].append((o, draws))
                for d in draws:
                    for role, c in d["roles"].items():
                        draw_rows.append({"kb": kb, "ood": o, "draw": d["draw"], "fixed": "+".join(d["fixed"]), "role": role, "slot": c,
                                          "z_relaxed": d["z_relaxed"], "zm_relaxed": d["zm_relaxed"]})
    G_["plans"] = plans
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(res / "tables" / "pairs.csv", index=False)
    pd.DataFrame(draw_rows).to_csv(res / "tables" / "draws.csv", index=False)
    log.info(f"pairs {len(pairs)}, dropped {int(pairs.dropped.sum())}; planning {time.time() - t0:.0f}s")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for out in ex.map(eval_kb, kbs):
            rows += out
    df = pd.DataFrame(rows)
    df.to_parquet(res / "tables" / "rows.parquet", index=False)
    meta = run_metadata(cfg, "A1.02_eval")
    meta["inputs"] = fingerprint([edir / "topk.npy", edir / "q_index.parquet", p(cfg, "a0_processed", cfg.get("distance_file", "tag_js.csv")),
                                  p(cfg, "a0_processed", "bands.json"), p(cfg, "a0_processed", "sites.json")]
                                 + ([sub_path] if sub_path else []))
    meta["n_rows"], meta["minutes"] = len(df), (time.time() - t0) / 60
    save_json(meta, res / "meta_02.json")
    log.info(f"rows {len(df)} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
