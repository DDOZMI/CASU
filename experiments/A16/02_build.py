from __future__ import annotations

import argparse
import ast
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import fingerprint, get_logger, load_config, p, run_metadata, save_json, stable_seed

ROLES = ("test", "cal", "bank")


def norm_q(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip()


def slug(path):
    return "--".join(re.sub(r"[^a-z0-9]+", "_", x.lower()).strip("_") for x in path)


def node_of(path, depth):
    return " > ".join(path[:depth])


def rows(pth):
    with gzip.open(pth, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield ast.literal_eval(line)


def read_file(args):
    raw, cat, depth = args
    qa = [(r["asin"], r["question"], r.get("answer") or "", r.get("questionType") or "", i)
          for i, r in enumerate(rows(Path(raw) / "qa" / f"qa_{cat}.json.gz"))]
    qa_asins = {r[0] for r in qa}
    tmap, prod = {}, {}
    for m in rows(Path(raw) / "meta" / f"meta_{cat}.json.gz"):
        a = m.get("asin")
        paths = [tuple(x) for x in m.get("categories", []) if x]
        if a is None:
            continue
        if paths:
            tmap[a] = tuple(sorted({node_of(x, depth) for x in paths}))
        if a in qa_asins:
            prod[a] = {"paths": paths, "also_bought": list((m.get("related") or {}).get("also_bought", []))}
    return cat, qa, prod, tmap


def js_distance(P, Q):
    M = 0.5 * (P + Q)

    def kl(A, B):
        nz = A > 0
        return float((A[nz] * np.log2(A[nz] / B[nz])).sum())

    return float(np.sqrt(max(0.0, 0.5 * kl(P, M) + 0.5 * kl(Q, M))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = p(cfg, "processed")
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger("A16.02", p(cfg, "results", "logs", "02_build.log"))
    raw = p(cfg, "amazon_raw")
    man = json.loads((raw / "manifest.json").read_text())
    sc, sp, dc = cfg["sites"], cfg["split"], cfg["distance"]
    depth = dc["profile_depth"]

    QA, PROD, TMAP = {}, {}, defaultdict(set)
    with ProcessPoolExecutor(max_workers=9) as ex:
        for cat, qa, prod, tmap in ex.map(read_file, [(str(raw), c, depth) for c in man["categories"]]):
            QA[cat], PROD[cat] = qa, prod
            for a, ns in tmap.items():
                TMAP[a].update(ns)
            log.info(f"{cat}: {len(qa)} QA rows, {len(prod)} QA products with metadata, {len(tmap)} products in metadata")
    log.info(f"target map: {len(TMAP)} products")

    single = {}
    for cat in man["categories"]:
        for a, d in PROD[cat].items():
            if len(d["paths"]) == 1 and a not in single:
                single[a] = (d["paths"][0], cat)
    qn = Counter()
    for cat in man["categories"]:
        for a, *_ in QA[cat]:
            if a in single:
                path = single[a][0]
                for dd in (2, 3):
                    if len(path) >= dd:
                        qn[path[:dd]] += 1
    bar = sc["min_questions_single_path"]
    d2 = sorted(k for k, v in qn.items() if len(k) == 2 and v >= bar)
    d3 = sorted(k for k, v in qn.items() if len(k) == 3 and v >= bar)
    cut = []
    for n2 in d2:
        ch = [n3 for n3 in d3 if n3[:2] == n2]
        cut += ch if len(ch) >= sc["split_depth3_if_children_at_least"] else [n2]
    cut_set = set(cut)
    site_of_path = lambda path: path[:3] if path[:3] in cut_set else (path[:2] if path[:2] in cut_set else None)
    log.info(f"candidate sites {len(cut)} (depth-2 {len(d2)}, split parents {len(d2) - sum(1 for c in cut if len(c) == 2)})")

    qs = defaultdict(list)
    for cat in man["categories"]:
        for a, q, ans, qt, i in QA[cat]:
            if a not in single:
                continue
            sp_ = site_of_path(single[a][0])
            if sp_ is None or not q.strip() or not ans.strip():
                continue
            qs[sp_].append({"asin": a, "question": q.strip(), "answer": ans.strip(), "qtype": qt, "file": cat, "line": i})
    info = {}
    for s in cut:
        seen, keep = set(), []
        for r in qs[s]:
            k = norm_q(r["question"])
            if k and k not in seen:
                seen.add(k)
                r["norm"] = k
                keep.append(r)
        info[s] = {"n_rows": len(qs[s]), "n_within_dedup": len(keep)}
        qs[s] = keep
    nsite = Counter(r["norm"] for s in cut for r in qs[s])
    multi = {k for k, v in nsite.items() if v >= 2}
    for s in cut:
        kept = [r for r in qs[s] if r["norm"] not in multi]
        info[s]["n_cross_removed"] = len(qs[s]) - len(kept)
        qs[s] = kept
    log.info(f"cross-site duplicate norms {len(multi)}; questions removed {sum(info[s]['n_cross_removed'] for s in cut)}")

    nq_need = sp["n_test"] + sp["n_cal"]
    c = sp["max_q_per_query_product"]
    queries, pools, dropped = {}, {}, {}
    for s in cut:
        by_p = defaultdict(list)
        for r in qs[s]:
            by_p[r["asin"]].append(r)
        prods = sorted(by_p)
        rng = np.random.default_rng(stable_seed(cfg["seed"], "A16_qprod", slug(s)))
        sel, used = [], []
        for j in rng.permutation(len(prods)):
            a = prods[j]
            rr = by_p[a]
            sel += [rr[k] for k in rng.permutation(len(rr))[:c]]
            used.append(a)
            if len(sel) >= nq_need:
                break
        used = set(used)
        pool = [r for r in qs[s] if r["asin"] not in used]
        info[s].update({"n_products": len(prods), "n_query_products": len(used), "n_query_available": len(sel), "bank_pool": len(pool)})
        if len(sel) < nq_need:
            dropped[s] = "too few query questions"
        elif len(pool) < sp["n_bank"]:
            dropped[s] = "bank pool < n_bank"
        else:
            queries[s], pools[s] = sel[:nq_need], pool
    qprods = {s: {r["asin"] for r in qs[s]} - {r["asin"] for r in pools[s]} for s in queries}
    sites_p = [s for s in cut if s in queries]
    log.info(f"final sites {len(sites_p)}; dropped {len(dropped)}: {dropped}")

    items, names = [], {}
    for s in sites_p:
        nm = slug(s)
        names[s] = nm
        perm = np.random.default_rng(stable_seed(cfg["seed"], "A16_role", nm)).permutation(nq_need)
        qsel = [queries[s][i] for i in perm]
        bank = [pools[s][i] for i in np.sort(np.random.default_rng(stable_seed(cfg["seed"], "A16_bank", nm)).permutation(len(pools[s]))[: sp["n_bank"]])]
        for role, lst in (("test", qsel[: sp["n_test"]]), ("cal", qsel[sp["n_test"]:]), ("bank", bank)):
            for j, r in enumerate(lst):
                items.append({"site": nm, "role": role, "pos": j, "title": r["question"],
                              "text": r["question"] if role != "bank" else f"{r['question']} {r['answer']}",
                              "asin": r["asin"], "qtype": r["qtype"], "file": r["file"], "line": r["line"], "norm": r["norm"]})
    I = pd.DataFrame(items)
    assert (I.groupby(["site", "role"]).size().unstack()[list(ROLES)] == [sp["n_test"], sp["n_cal"], sp["n_bank"]]).all().all()
    for nm, g in I.groupby("site"):
        assert not set(g[g.role == "bank"].asin) & set(g[g.role != "bank"].asin), nm
    assert not I.duplicated(["norm"]).any()
    I.to_parquet(out / "items.parquet", index=False)

    S = len(sites_p)
    prof, tstat = [], {}
    for s in sites_p:
        cnt, n_t, n_un = Counter(), 0, 0
        for a in sorted({r["asin"] for r in qs[s]} - qprods[s]):
            cat = single[a][1]
            for t in PROD[cat][a]["also_bought"]:
                n_t += 1
                ns = sorted(TMAP.get(t, ()))
                if not ns:
                    n_un += 1
                    continue
                for n_ in ns:
                    cnt[n_] += 1.0 / len(ns)
        prof.append((cnt, n_un))
        tstat[names[s]] = {"targets": n_t, "unmapped": n_un, "profile_nodes": len(cnt)}
    vocab = sorted(set().union(*[c_ for c_, _ in prof]))
    vi = {t: i for i, t in enumerate(vocab)}
    P = np.zeros((S, len(vocab) + 1))
    for a, (cnt, n_un) in enumerate(prof):
        for t, v in sorted(cnt.items()):
            P[a, vi[t]] = v
        P[a, -1] = n_un
    Pu = P / P.sum(1, keepdims=True)
    P = P[:, :-1]
    assert (P.sum(1) > 0).all()
    P = P / P.sum(1, keepdims=True)
    D = np.zeros((S, S))
    Du = np.zeros((S, S))
    Tr = np.zeros((S, S), int)
    for a in range(S):
        for b in range(a + 1, S):
            D[a, b] = D[b, a] = js_distance(P[a], P[b])
            Du[a, b] = Du[b, a] = js_distance(Pu[a], Pu[b])
            x, y = sites_p[a], sites_p[b]
            Tr[a, b] = Tr[b, a] = 1 if (len(x) == 3 and len(y) == 3 and x[:2] == y[:2]) else (2 if x[0] == y[0] else 3)
    nms = [names[s] for s in sites_p]
    pd.DataFrame(D, index=nms, columns=nms).to_csv(out / "copurchase_js.csv")
    pd.DataFrame(Tr, index=nms, columns=nms).to_csv(out / "tree_dist.csv")
    pd.DataFrame(Du, index=nms, columns=nms).to_csv(out / "copurchase_js_unmapped.csv")
    d = D[np.triu_indices(S, 1)]
    q_nb, q_far = float(np.quantile(d, dc["q_neighbor"])), float(np.quantile(d, dc["q_far"]))
    save_json({"q_neighbor_level": dc["q_neighbor"], "q_far_level": dc["q_far"], "q_neighbor_value": q_nb, "q_far_value": q_far,
               "share_d_gt_0.95": float(np.mean(d > 0.95)), "d_quantiles_5_25_50_75_95": [float(x) for x in np.quantile(d, [.05, .25, .5, .75, .95])],
               "distance": "co-purchase profile JS"}, out / "bands.json")
    save_json({"groups": [], "sites": []}, out / "subset.json")
    save_json({names[s]: {"category": s[0], "path": " > ".join(s), "depth": len(s), **info[s], **tstat[names[s]]} for s in sites_p}, out / "sites.json")
    qt = I[I.role == "test"].qtype.value_counts(normalize=True).to_dict()
    rep = {"n_candidate_sites": len(cut), "n_sites": S, "dropped": {slug(k): {"reason": v, **info[k]} for k, v in dropped.items()},
           "cross_site_duplicate_norms": len(multi), "cross_site_removed_total_candidates": int(sum(info[s]["n_cross_removed"] for s in cut)),
           "n_categories": len({s[0] for s in sites_p}), "tree_dist_counts": {int(k): int(v) for k, v in zip(*np.unique(Tr[np.triu_indices(S, 1)], return_counts=True))},
           "cross_site_removed_final_sites": int(sum(info[s]["n_cross_removed"] for s in sites_p)),
           "unmapped_target_share": float(sum(v["unmapped"] for v in tstat.values()) / max(1, sum(v["targets"] for v in tstat.values()))),
           "test_qtype_share": qt, "bands": {"q_nb": q_nb, "q_far": q_far, "share_gt_0.95": float(np.mean(d > 0.95))},
           "spearman_js_tree": float(pd.Series(d).corr(pd.Series(Tr[np.triu_indices(S, 1)].astype(float)), method="spearman"))}
    save_json(rep, out / "build_report.json")
    log.info(f"build report: {json.dumps(rep)}")
    meta = run_metadata(cfg, "A16.02_build")
    meta["inputs"] = fingerprint([raw / "manifest.json"])
    meta["raw_sha256_from_manifest"] = {k: v["sha256"] for k, v in man["files"].items()}
    meta["outputs"] = fingerprint([out / f for f in ("items.parquet", "sites.json", "copurchase_js.csv", "tree_dist.csv", "copurchase_js_unmapped.csv", "bands.json")])
    save_json(meta, out / "meta_02.json")


if __name__ == "__main__":
    main()
