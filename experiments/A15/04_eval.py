from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

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


A1 = _load("a1_eval", PROJECT_ROOT / "experiments" / "A1" / "02_eval.py")
D3 = _load("a3_diag", PROJECT_ROOT / "experiments" / "A3" / "02_diagnose.py")


def unit(X):
    X = np.asarray(X, np.float64)
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def unit_t(X):
    X = torch.as_tensor(np.asarray(X, np.float32), device=DEV)
    return X / X.norm(dim=1, keepdim=True)


def centroid(X):
    return unit(unit(X).mean(0))


def bank_scores(s, absk, Tg, taus, a, k):
    A = Tg.mean(2)
    m = A.shape[1]
    out = {}
    for t in taus:
        c = A.mean(1) if t == "inf" else float(t) * (logsumexp(A / float(t), axis=1) - np.log(m))
        out[f"S_tau@{t}"] = (s - c, c)
    U = Tg.reshape(len(s), -1)
    top = -np.partition(-U, k - 1, axis=1)[:, :k]
    c = a * top.mean(1)
    out[f"mix@{a:g}"] = (s - c, c)
    out[f"flatsp@{a:g}"] = (-np.sqrt(np.clip(2 - 2 * absk, 0, None)) + a * np.sqrt(np.clip(2 - 2 * top.min(1), 0, None)), None)
    return out


def run_one(rn, run, cfg, log, smoke_kbs=0):
    ec, k = cfg["eval"], cfg["k"]
    taus, a, acc, R = ec["taus"], ec["alpha"], ec["op_accept"], ec["R"]
    r, r2 = run["retriever"], run["second"]
    edir = PROJECT_ROOT / run["emb"]
    sites = load_json(edir / "sites.json")
    sidx = {s: i for i, s in enumerate(sites)}
    qi = pd.read_parquet(edir / "q_index.parquet")
    nse = len(qi)
    rows_of = {kk: g.index.to_numpy() for kk, g in qi.groupby(["site", "role"])}
    oodq = pd.read_parquet(p(cfg, "processed", "ood_queries.parquet"))
    srcs = list(dict.fromkeys(oodq.source))
    for s_ in srcs:
        rows_of[(s_, "test")] = nse + oodq.index[oodq.source == s_].to_numpy()
    tk = p(cfg, "emb", "topk", rn)
    Tse = np.concatenate([np.asarray(np.load(edir / "topk.npy", mmap_mode="r")), np.load(tk / "se_ood.npy")]).astype(np.float64)
    Tw = np.load(tk / "wiki.npy").astype(np.float64)
    Tm = np.load(tk / "ms.npy").astype(np.float64)
    groups = load_json(tk / "groups.json")
    Ty = np.load(tk / "yahoo.npy").astype(np.float64)
    wgroups, ygroups, nms = groups["wiki_groups"], groups["yahoo_groups"], groups["ms_chunks"]
    ytopic = [g.split("#")[0] for g in ygroups]
    Lse, Lw, Ly = Tse.mean(2), Tw.mean(2), Ty.mean(2)
    items = pd.read_parquet(p(cfg, "a0_processed", "items.parquet"), columns=["site", "role", "pos", "text"])
    it = items[items.role.isin(["test", "cal"])].set_index(["site", "role", "pos"]).text
    qtext = list(it.reindex(pd.MultiIndex.from_frame(qi[["site", "role", "pos"]])).to_numpy()) + oodq.text.tolist()
    qlen = np.array([len(str(x).split()) for x in qtext], float)
    crowd_se = {(kb, j): float(Lse[rows_of[(kb, "cal")], sidx[j]].mean()) for kb in sites for j in sites if j != kb}
    thr_se = float(np.quantile(list(crowd_se.values()), ec["exclusion_quantile"]))
    excl_se = {kb: {j for j in sites if j != kb and crowd_se[(kb, j)] >= thr_se} for kb in sites}
    a14 = pd.read_csv(p(cfg, "a14_results", "tables", f"excluded_{rn}.csv"))
    assert all(set(g[g.excluded].site) == excl_se[kb] for kb, g in a14.groupby("kb")), "SE exclusion differs from A14"
    crowd_w = {(kb, g): float(Lw[rows_of[(kb, "cal")], gi].mean()) for kb in sites for gi, g in enumerate(wgroups)}
    thr_w = float(np.quantile(list(crowd_w.values()), ec["exclusion_quantile"]))
    elig_w = {kb: [gi for gi, g in enumerate(wgroups) if crowd_w[(kb, g)] < thr_w] for kb in sites}
    crowd_y = {(kb, g): float(Ly[rows_of[(kb, "cal")], gi].mean()) for kb in sites for gi, g in enumerate(ygroups)}
    thr_y = float(np.quantile(list(crowd_y.values()), ec["exclusion_quantile"]))
    elig_y = {kb: [gi for gi, g in enumerate(ygroups) if crowd_y[(kb, g)] < thr_y] for kb in sites}
    log.info(f"{rn}: Yahoo threshold {thr_y:.4f}; yahoo eligible per KB 5/50/95% " + " / ".join(f"{v:.0f}" for v in np.quantile([len(v) for v in elig_y.values()], [.05, .5, .95])))
    log.info(f"{rn}: thresholds SE {thr_se:.4f} wiki {thr_w:.4f}; wiki eligible per KB 5/50/95% "
             + " / ".join(f"{v:.0f}" for v in np.quantile([len(v) for v in elig_w.values()], [.05, .5, .95])))
    a02 = p(cfg, "a0_emb", r2)
    cen_site = {s: centroid(np.load(a02 / f"{s}__bank.npy")) for s in sites}
    W2 = np.load(p(cfg, "emb", r2, "wiki.npy"))
    cen_w = np.stack([centroid(W2[gi * 3000:(gi + 1) * 3000]) for gi in range(len(wgroups))])
    Y2 = np.load(p(cfg, "emb", r2, "yahoo.npy"))
    cen_y = np.stack([centroid(Y2[gi * 3000:(gi + 1) * 3000]) for gi in range(len(ygroups))])
    del Y2
    del W2
    O2 = np.load(p(cfg, "emb", r2, "ood.npy"))
    cen_o = {s_: centroid(O2[oodq.index[oodq.source == s_].to_numpy()]) for s_ in srcs}
    E2 = {s_: unit(O2[oodq.index[oodq.source == s_].to_numpy()]) for s_ in srcs}
    for s in sites:
        f = p(cfg, "emb", r2, f"se_test_{s}.npy")
        E2[s] = unit(np.load(f) if f.exists() else np.load(a02 / f"{s}__test.npy"))
        cen_o[s] = unit(E2[s].mean(0))
    cen_site_m = np.stack([cen_site[s] for s in sites])
    Dtag = pd.read_csv(p(cfg, "a0_processed", "tag_js.csv"), index_col=0).loc[sites, sites]
    pairs = pd.read_csv(PROJECT_ROOT / run["a1"] / "tables" / "pairs.csv")[["kb", "ood", "tier"]].assign(family="se")
    ns = pd.DataFrame([{"kb": kb, "ood": s_, "family": s_.split(":")[0], "kcos": float(cen_site[kb] @ cen_o[s_])} for kb in sites for s_ in srcs])
    ns["tier"] = pd.qcut(-ns.kcos, 3, labels=["near", "mid", "far"]).astype(str)
    cells = pd.concat([pairs, ns.drop(columns="kcos")], ignore_index=True)
    kbs = sites[:smoke_kbs] if smoke_kbs else sites
    cells = cells[cells.kb.isin(kbs)]
    Qr = unit_t(np.concatenate([np.concatenate([np.load(p(cfg, "a0_emb", r) / f"{s}__{ro}.npy") for s in sites for ro in ("test", "cal")]),
                                np.load(p(cfg, "emb", r, "ood.npy"))]))
    MS9 = unit_t(np.load(p(cfg, "p9_emb", r, "pool.npy")))
    a13 = pd.read_parquet(p(cfg, "a13_results", "tables", f"cells_{rn}.parquet"))
    a13 = a13[a13.score == "flats_ms@0.5"].set_index(["kb", "ood"]).auroc
    out, meta_cells, chk = [], [], [0.0, 0]
    t0 = time.time()
    for ci, (kb, o, tier, fam) in enumerate(zip(cells.kb, cells.ood, cells.tier, cells.family)):
        i_id, i_ood, i_cal = rows_of[(kb, "test")], rows_of[(o, "test")], rows_of[(kb, "cal")]
        sel = np.r_[i_id, i_ood, i_cal]
        n1, n2 = len(i_id), len(i_id) + len(i_ood)
        s = Lse[sel, sidx[kb]]
        absk = Tse[sel, sidx[kb], k - 1]
        A_abs = A1.metrics(s[:n1], s[n1:n2], s[n2:], acc)["auroc"]
        al = A1.auroc(-qlen[i_id], -qlen[i_ood])
        base = {"kb": kb, "ood": o, "family": fam, "tier": tier, "auroc_abs": A_abs}
        Qs = Qr[torch.as_tensor(sel, device=DEV)]
        refA = []
        for d in range(ec["R_ref"]):
            ix = np.random.default_rng(stable_seed(cfg["seed"], "A2_ms", kb, o, d)).permutation(MS9.shape[0])[:12000]
            top = (Qs @ MS9[torch.as_tensor(ix, device=DEV)].T).topk(k, dim=1).values.double().cpu().numpy()
            S = -np.sqrt(np.clip(2 - 2 * absk, 0, None)) + 0.5 * np.sqrt(np.clip(2 - 2 * top[:, k - 1], 0, None))
            refA.append(A1.metrics(S[:n1], S[n1:n2], S[n2:], acc)["auroc"])
        refA = float(np.mean(refA))
        if fam == "se" and (kb, o) in a13.index:
            chk = [max(chk[0], abs(refA - a13[(kb, o)])), chk[1] + 1]
        se_el = [j for j in sites if j != kb and j not in excl_se[kb]]
        se_el_o = [j for j in se_el if j != o]
        y_el_o = [gi for gi in elig_y[kb] if ytopic[gi] != o]
        y_min = len(elig_y[kb]) - max(sum(ytopic[gi] == tp for gi in elig_y[kb]) for tp in set(ytopic))
        n = min(len(se_el) - 1, len(elig_w[kb]), y_min, nms)
        meta_cells.append({**base, "auroc_ref": refA, "len_auroc": al, "n_matched": n, "n_se_full": len(se_el_o), "n_wiki_full": len(elig_w[kb]),
                           "n_yahoo_full": len(y_el_o)})
        banks = []
        for d in range(R):
            pick = list(np.random.default_rng(stable_seed(cfg["seed"], "A15_se", kb, d)).choice(se_el, n + 1, replace=False))
            pick = [x for x in pick if x != o][:n]
            banks.append(("se_n", d, "se", [sidx[x] for x in pick]))
            rw = np.random.default_rng(stable_seed(cfg["seed"], "A15_wiki_set", kb, d))
            banks.append(("wiki_n", d, "wiki", list(rw.choice(elig_w[kb], n, replace=False))))
            py = [elig_y[kb][i] for i in np.random.default_rng(stable_seed(cfg["seed"], "A15_yahoo_set", kb, d)).permutation(len(elig_y[kb]))]
            banks.append(("yahoo_n", d, "yahoo", [gi for gi in py if ytopic[gi] != o][:n]))
            rm = np.random.default_rng(stable_seed(cfg["seed"], "A15_ms_set", kb, d))
            banks.append(("ms_n", d, "ms", list(rm.choice(nms, n, replace=False))))
        banks += [("se_full", 0, "se", [sidx[x] for x in se_el_o]), ("wiki_full", 0, "wiki", list(elig_w[kb])), ("yahoo_full", 0, "yahoo", y_el_o),
                  ("ms_full", 0, "ms", list(range(min(len(se_el_o), nms))))]
        for bname, d, src, G in banks:
            Tg = {"se": Tse, "wiki": Tw, "yahoo": Ty, "ms": Tm}[src][np.ix_(sel, G)]
            if src == "se":
                nn = max(float(cen_o[o] @ cen_site[sites[j]]) for j in G)
                cov = float((E2[o] @ cen_site_m[G].T).max(1).mean())
                if fam == "se":
                    elig_r = sorted((x for x in sites if x not in (o, kb)), key=lambda x: (Dtag.at[o, x], x))
                    rk = {x: i + 1 for i, x in enumerate(elig_r)}
                    nnr = min(rk[sites[j]] for j in G)
                else:
                    nnr = np.nan
            elif src in ("wiki", "yahoo"):
                cg = (cen_w if src == "wiki" else cen_y)[G]
                nn = float((cg @ cen_o[o]).max())
                cov = float((E2[o] @ cg.T).max(1).mean())
                nnr = np.nan
            else:
                nn, cov, nnr = np.nan, np.nan, np.nan
            for nm, (S, c) in bank_scores(s, absk, Tg, taus, a, k).items():
                mt = A1.metrics(S[:n1], S[n1:n2], S[n2:], acc)
                rec = {**base, "bank": bname, "draw": d, "n_groups": len(G), "score": nm, "auroc": mt["auroc"], "op_id_acc": mt["op_id_acc"],
                       "op_ood_acc": mt["op_ood_acc"], "fpr95": mt["fpr95"], "nn_sim": nn, "coverage": cov, "nn_rank_tag": nnr}
                if c is not None:
                    mo = D3.moments(s[:n1], s[n1:n2], c[:n1], c[n1:n2])
                    rec.update({"pi": mo["pi"], "T_var": mo["T_var"], "T_push": mo["T_push"], "rho_id": float(np.corrcoef(s[:n1], c[:n1])[0, 1])})
                out.append(rec)
        if ci % 200 == 0:
            log.info(f"{rn}: cell {ci}/{len(cells)} ({(time.time() - t0) / 60:.1f} min)")
    log.info(f"{rn}: reference vs A13 flats_ms@0.5: max |dAUROC| = {chk[0]:.2e} over {chk[1]}")
    assert (chk[1] > 0 or smoke_kbs) and chk[0] < 1e-5, chk
    return pd.DataFrame(out), pd.DataFrame(meta_cells), {"thr_se": thr_se, "thr_wiki": thr_w, "thr_yahoo": thr_y, "ref_check": chk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--smoke-kbs", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    assert not args.smoke_kbs or args.out, "--smoke-kbs requires --out"
    cfg = load_config(args.config)
    res = Path(args.out) if args.out else p(cfg, "results")
    (res / "tables").mkdir(parents=True, exist_ok=True)
    log = get_logger("A15.04", res / "logs" / "04_eval.log")
    for rn in args.runs or list(cfg["runs"]):
        run = cfg["runs"][rn]
        t0 = time.time()
        Dd, Mc, info = run_one(rn, run, cfg, log, args.smoke_kbs)
        Dd.to_parquet(res / "tables" / f"rows_{rn}.parquet", index=False)
        Mc.to_parquet(res / "tables" / f"cells_meta_{rn}.parquet", index=False)
        meta = run_metadata(cfg, f"A15.04_eval:{rn}")
        tk = p(cfg, "emb", "topk", rn)
        meta["inputs"] = fingerprint([PROJECT_ROOT / run["emb"] / f for f in ("topk.npy", "q_index.parquet", "sites.json")] + [p(cfg, "a0_processed", "tag_js.csv")]
                                     + [tk / f for f in ("se_ood.npy", "wiki.npy", "yahoo.npy", "ms.npy", "groups.json")]
                                     + [p(cfg, "processed", "ood_queries.parquet"), p(cfg, "a14_results", "tables", f"excluded_{rn}.csv"),
                                        p(cfg, "a13_results", "tables", f"cells_{rn}.parquet")])
        meta["imported_code_sha256"] = fingerprint([PROJECT_ROOT / "experiments" / "A1" / "02_eval.py", PROJECT_ROOT / "experiments" / "A3" / "02_diagnose.py"])
        meta.update({**info, "n_rows": len(Dd), "minutes": (time.time() - t0) / 60, "smoke_kbs": args.smoke_kbs})
        save_json(meta, res / f"meta_04_{rn}.json")
        log.info(f"{rn}: {len(Dd)} rows in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
