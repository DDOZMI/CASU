from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json, stable_seed
from oodrag.encoders import doc_text

ROLES = ("test", "cal", "bank", "l3")


def norm_title(t):
    return re.sub(r"\s+", " ", t.lower()).strip()


def build_site(args):
    path, site, n_expected, sp, tag_n, seed_split, seed_tags = args
    recs = []
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            recs.append((i, d["texts"][0], d["texts"][1], d["tags"]))
    assert len(recs) == n_expected, (site, len(recs), n_expected)
    seen, keep = set(), []
    n_empty = 0
    for r in recs:
        if not r[1].strip() or not r[2].strip():
            n_empty += 1
            continue
        k = norm_title(r[1])
        if k in seen:
            continue
        seen.add(k)
        keep.append(r)
    n_dup = len(recs) - n_empty - len(keep)
    perm = np.random.default_rng(seed_split).permutation(len(keep))
    sizes = [sp["n_test"], sp["n_cal"], sp["n_bank"], sp["n_l3"]]
    assert sum(sizes) <= len(keep), site
    out, o = [], 0
    for role, n in zip(ROLES, sizes):
        for j, k in enumerate(perm[o:o + n]):
            i, t, b, tg = keep[k]
            out.append({"site": site, "role": role, "pos": j, "dump_line": i, "title": t,
                        "text": t if role in ("test", "cal") else doc_text(t, b), "tags": json.dumps(tg)})
        o += n
    query_rows = set(perm[: sp["n_test"] + sp["n_cal"]].tolist())
    rest = np.array([k for k in range(len(keep)) if k not in query_rows])
    sel = rest[np.random.default_rng(seed_tags).permutation(len(rest))[: min(tag_n, len(rest))]]
    c = Counter()
    for k in sel:
        c.update(keep[k][3])
    return site, out, dict(c), {"n_dump": len(recs), "n_empty": n_empty, "n_dup_title": n_dup, "n_kept": len(keep),
                                "tag_sample": int(len(sel))}


def js_distance(P, Q):
    M = 0.5 * (P + Q)

    def kl(A, B):
        nz = A > 0
        return float((A[nz] * np.log2(A[nz] / B[nz])).sum())

    return float(np.sqrt(max(0.0, 0.5 * kl(P, M) + 0.5 * kl(Q, M))))


def subset_triples(sites, cat, D, quota, size):
    idx = {s: i for i, s in enumerate(sites)}
    groups = []
    for c, q in quota.items():
        pool = sorted(s for s in sites if cat[s] == c)
        used = set()
        for _ in range(q):
            best = None
            for s in pool:
                if s in used:
                    continue
                nn = sorted((t for t in pool if t != s and t not in used), key=lambda t: (D[idx[s], idx[t]], t))[: size - 1]
                if len(nn) < size - 1:
                    continue
                g = [s] + nn
                cost = np.mean([D[idx[a], idx[b]] for ai, a in enumerate(g) for b in g[ai + 1:]])
                key = (cost, tuple(sorted(g)))
                if best is None or key < best[0]:
                    best = (key, g)
            assert best is not None, c
            groups.append({"category": c, "sites": sorted(best[1]), "mean_d": float(best[0][0])})
            used.update(best[1])
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = p(cfg, "processed")
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger("A0.02", p(cfg, "results", "logs", "02_build.log"))
    raw = p(cfg, "se_raw")
    counts = load_json(raw / "site_counts.json")
    assert len(counts) == 87
    excl = set(cfg["exclude_sites"])
    sites = [s for s, _, _ in counts if s not in excl]
    cat = {s: (c if c is not None else cfg["category_alias"][s]) for s, _, c in counts}
    nq = {s: n for s, n, _ in counts}
    log.info(f"sites {len(sites)} (excluded: {sorted(excl)})")
    man = load_json(raw / "manifest.json")["files"]
    sp, tg = cfg["split"], cfg["tags"]
    jobs = [(str(raw / f"{s}.jsonl.gz"), s, nq[s], sp, tg["sample_n"], stable_seed(cfg["seed"], "A0_split", s),
             stable_seed(cfg["seed"], "A0_tags", s)) for s in sites]
    rows, tcount, info = [], {}, {}
    with ProcessPoolExecutor(max_workers=12) as ex:
        for s, o, c, inf in ex.map(build_site, jobs):
            rows += o
            tcount[s], info[s] = c, inf
            log.info(f"{s}: {inf}")
    items = pd.DataFrame(rows)
    assert (items.groupby(["site", "role"]).size().unstack()[list(ROLES)] == [sp["n_test"], sp["n_cal"], sp["n_bank"], sp["n_l3"]]).all().all()
    assert not items.duplicated(["site", "dump_line"]).any()
    items.to_parquet(out / "items.parquet", index=False)
    save_json({s: {"category": cat[s], **info[s]} for s in sites}, out / "sites.json")

    vocab = sorted(set().union(*[tcount[s] for s in sites]))
    vi = {t: i for i, t in enumerate(vocab)}
    P = np.zeros((len(sites), len(vocab)))
    for a, s in enumerate(sites):
        for t, c in tcount[s].items():
            P[a, vi[t]] = c
    P /= P.sum(1, keepdims=True)
    S = len(sites)
    D = np.zeros((S, S))
    for a in range(S):
        for b in range(a + 1, S):
            D[a, b] = D[b, a] = js_distance(P[a], P[b])
    pd.DataFrame(D, index=sites, columns=sites).to_csv(out / "tag_js.csv")
    d = D[np.triu_indices(S, 1)]
    q_nb, q_far = float(np.quantile(d, tg["q_neighbor"])), float(np.quantile(d, tg["q_far"]))
    save_json({"q_neighbor_level": tg["q_neighbor"], "q_far_level": tg["q_far"], "q_neighbor_value": q_nb,
               "q_far_value": q_far, "share_d_gt_0.95": float(np.mean(d > 0.95))}, out / "bands.json")
    groups = subset_triples(sites, cat, D, cfg["subset"]["group_quota"], cfg["subset"]["group_size"])
    save_json({"groups": groups, "sites": sorted(s for g in groups for s in g["sites"])}, out / "subset.json")
    log.info(f"bands: neighbour <= {q_nb:.4f}, far > {q_far:.4f}; subset {sum(len(g['sites']) for g in groups)} sites")
    for g in groups:
        log.info(f"  {g}")
    meta = run_metadata(cfg, "A0.02_build")
    meta["inputs"] = fingerprint([raw / "site_counts.json", raw / "manifest.json"])
    meta["dump_files_sha256_from_manifest"] = {f"{s}.jsonl.gz": man[f"{s}.jsonl.gz"]["sha256"] for s in sites}
    meta["outputs"] = fingerprint([out / "items.parquet", out / "tag_js.csv", out / "bands.json", out / "subset.json"])
    save_json(meta, out / "meta_02.json")


if __name__ == "__main__":
    main()
