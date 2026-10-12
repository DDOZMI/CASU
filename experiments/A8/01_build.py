from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import PROJECT_ROOT, fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json, set_determinism, stable_seed

set_determinism()
import torch


def unit(X):
    X = np.asarray(X, np.float32)
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--retrievers", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("A8.01", p(cfg, "results", "logs", "01_build.log"))
    k = cfg["k"]
    smap = {f: s for f, s in cfg["cqa_site_map"].items() if not f.startswith("_")}
    F = list(smap)
    a0p = p(cfg, "a0_processed_src")
    a0_sites = load_json(PROJECT_ROOT / "data" / "emb" / "A1" / "bge-m3" / "sites.json")
    cqa_cols = [f"cqa:{smap[f]}" for f in F]
    sites = a0_sites + cqa_cols
    twin = {c: c[4:] for c in cqa_cols}

    out_p = p(cfg, "a0_processed")
    out_p.mkdir(parents=True, exist_ok=True)
    D = pd.read_csv(a0p / "tag_js.csv", index_col=0)
    full = [twin.get(s, s) for s in sites]
    DX = pd.DataFrame(D.loc[full, full].to_numpy(), index=sites, columns=sites)
    DX.to_csv(out_p / "tag_js.csv")
    shutil.copy(a0p / "bands.json", out_p / "bands.json")
    sinfo = load_json(a0p / "sites.json")
    save_json({**sinfo, **{c: {**sinfo[twin[c]], "twin": twin[c]} for c in cqa_cols}}, out_p / "sites.json")

    Q = pd.read_parquet(p(cfg, "p4_processed", "queries.parquet"))
    cqa = (Q["group"] == "cqa").to_numpy()
    Qc = Q[cqa].reset_index(drop=True).assign(qrow=np.flatnonzero(cqa))
    Qc["qid"] = Qc.qid.astype(str)
    parts = []
    for f in F:
        g = Qc[Qc.source == f].sort_values("qid").reset_index(drop=True)
        perm = np.random.default_rng(stable_seed(cfg["seed"], "A8_split", f)).permutation(len(g))
        n_cal = int(np.floor(len(g) * cfg["split"]["cal_frac"]))
        g = g.iloc[perm].reset_index(drop=True)
        g["role"] = np.where(np.arange(len(g)) < n_cal, "cal", "test")
        for ro in ("test", "cal"):
            h = g[g.role == ro].reset_index(drop=True)
            h["pos"] = np.arange(len(h))
            h["site"] = f"cqa:{smap[f]}"
            parts.append(h)
    QI = pd.concat(parts, ignore_index=True)
    log.info("queries per forum (test / cal): " + ", ".join(f"{f} {int(((QI.source == f) & (QI.role == 'test')).sum())}/{int(((QI.source == f) & (QI.role == 'cal')).sum())}" for f in F))

    Bk = pd.read_parquet(p(cfg, "p6_processed", "banks.parquet"))
    PS = pd.read_parquet(p(cfg, "p6_processed", "positives.parquet"))
    masks, mrows = {}, []
    for f in F:
        ids = Bk[Bk.source == f].sort_values("rank")["doc_id"].astype(str).to_numpy()
        assert len(ids) == 3000
        rk = {d: i for i, d in enumerate(ids)}
        pmap = PS[PS.source == f].assign(qid=lambda x: x.qid.astype(str), doc_id=lambda x: x.doc_id.astype(str)).groupby("qid")["doc_id"].apply(list).to_dict()
        rs = np.flatnonzero(QI.source.to_numpy() == f)
        qids = QI.qid.to_numpy()
        pr = {(i, rk[qids[i]]) for i in rs if qids[i] in rk} | {(i, rk[d]) for i in rs for d in pmap.get(qids[i], []) if d in rk}
        pr = sorted(pr)
        masks[f] = (np.array([a for a, _ in pr], np.int64), np.array([b for _, b in pr], np.int64))
        mrows.append({"forum": f, "masked_pairs": len(pr), "queries_affected": len({a for a, _ in pr}), "queries": len(rs)})
    pd.DataFrame(mrows).to_csv(out_p / "mask_counts.csv", index=False)
    log.info("masked: " + ", ".join(f"{r['forum']} {r['queries_affected']}/{r['queries']}" for r in mrows))

    for r in args.retrievers or cfg["retrievers"]:
        t0 = __import__("time").time()
        E = torch.from_numpy(unit(np.load(p(cfg, "p4_emb", r, "q.npy"))[QI.qrow.to_numpy()])).cuda()
        T = np.empty((len(QI), len(sites), k), np.float32)
        Tf = np.empty_like(T)
        for j, s in enumerate(sites):
            if s.startswith("cqa:"):
                f = next(ff for ff in F if smap[ff] == twin[s])
                B = unit(np.load(p(cfg, "p6_emb", r, f"bank_{f}.npy")))
            else:
                B = unit(np.load(p(cfg, "a0_emb", r, f"{s}__bank.npy")))
            assert B.shape[0] == 3000
            Bg = torch.from_numpy(B).cuda()
            S = E @ Bg.T
            Tf[:, j] = S.topk(k, dim=1).values.cpu().numpy()
            if s.startswith("cqa:") and len(masks[f][0]):
                S[torch.from_numpy(masks[f][0]).cuda(), torch.from_numpy(masks[f][1]).cuda()] = float("-inf")
            T[:, j] = S.topk(k, dim=1).values.cpu().numpy()
            del Bg, S
        assert np.isfinite(T).all()
        for root, X in (("emb", T), ("emb_full", Tf)):
            out = p(cfg, root, r)
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "topk.npy", X)
            QI[["site", "role", "pos", "source", "qid", "qrow"]].to_parquet(out / "q_index.parquet", index=False)
            save_json(sites, out / "sites.json")
            for rr in cfg["retrievers"]:
                files = [p(cfg, "a0_emb", rr, f"{twin.get(s, s)}__bank.npy") for s in sites]
                if not all(f_.exists() for f_ in files):
                    log.info(f"centroids {rr}: banks missing, skipped")
                    continue
                np.save(out / f"centroids_{rr}.npy", np.stack([unit(unit(np.load(f_)).mean(0)) for f_ in files]))
            meta = run_metadata(cfg, f"A8.01_build:{r}:{root}")
            meta["inputs"] = fingerprint([p(cfg, "p4_emb", r, "q.npy"), p(cfg, "p4_processed", "queries.parquet"), p(cfg, "p6_processed", "banks.parquet"),
                                          p(cfg, "p6_processed", "positives.parquet"), a0p / "tag_js.csv"]
                                         + [p(cfg, "p6_emb", r, f"bank_{f}.npy") for f in F]
                                         + [p(cfg, "a0_emb", rr, f"{s}__bank.npy") for rr in cfg["retrievers"] for s in a0_sites]
                                         + [PROJECT_ROOT / "data" / "emb" / "A1" / "bge-m3" / "sites.json", a0p / "sites.json", a0p / "bands.json"])
            meta["shape"], meta["masked"] = list(X.shape), root == "emb"
            save_json(meta, out / "meta_01.json")
        log.info(f"{r}: topk {T.shape} in {(__import__('time').time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
