from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import PROJECT_ROOT, fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json, set_determinism

set_determinism()
import torch


def norm_title(t):
    return re.sub(r"\s+", " ", t.lower()).strip()


def unit(X):
    X = np.asarray(X, np.float32)
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--retrievers", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("A8.04", p(cfg, "results", "logs", "04_extras_build.log"))
    k = cfg["k"]
    smap = {f: s for f, s in cfg["cqa_site_map"].items() if not f.startswith("_")}
    F = list(smap)
    out_p = p(cfg, "a0_processed")
    emb_ip = PROJECT_ROOT / "data" / "emb" / "A8_ip"

    Bk = pd.read_parquet(p(cfg, "p6_processed", "banks.parquet"))
    PS = pd.read_parquet(p(cfg, "p6_processed", "positives.parquet")).astype({"qid": str, "doc_id": str})
    pmap = {f: g.groupby("qid")["doc_id"].apply(list).to_dict() for f, g in PS.groupby("source")}
    bank_ids = {f: set(Bk[Bk.source == f]["doc_id"].astype(str)) for f in F}

    items = pd.read_parquet(p(cfg, "a0_processed_src", "items.parquet"), columns=["site", "role", "pos", "title"])
    items = items[items.role == "bank"]
    twin_idx = {}
    for f in F:
        t = items[items.site == smap[f]].sort_values("pos")
        assert (t.pos.to_numpy() == np.arange(3000)).all()
        d = {}
        for i, x in enumerate(t.title):
            d.setdefault(norm_title(x), []).append(i)
        twin_idx[f] = d
    corpus_title = {}
    for f in F:
        ct = {}
        with open(PROJECT_ROOT / "data" / "raw" / "cqadupstack" / f / "corpus.jsonl") as fh:
            for line in fh:
                r_ = json.loads(line)
                ct[str(r_["_id"])] = r_.get("title", "")
        corpus_title[f] = ct
    Q = pd.read_parquet(p(cfg, "p4_processed", "queries.parquet"))

    for r in args.retrievers or cfg["retrievers"]:
        e1, e0 = p(cfg, "emb", r), p(cfg, "emb_full", r)
        qi = pd.read_parquet(e1 / "q_index.parquet")
        assert qi.equals(pd.read_parquet(e0 / "q_index.parquet"))
        sites = load_json(e1 / "sites.json")
        sidx = {s: i for i, s in enumerate(sites)}
        T1 = np.load(e1 / "topk.npy")
        T0 = np.load(e0 / "topk.npy", mmap_mode="r")
        own = qi.site.map(sidx).to_numpy()
        ar = np.arange(len(qi))
        changed = (T1[ar, own] != np.asarray(T0[ar, own])).any(1)
        qid = qi.qid.astype(str).to_numpy()
        src = qi.source.to_numpy()
        has_mask = np.array([qid[i] in bank_ids[src[i]] or any(d in bank_ids[src[i]] for d in pmap.get(src[i], {}).get(qid[i], [])) for i in ar])
        mc = pd.read_csv(out_p / "mask_counts.csv").set_index("forum")
        for f in F:
            assert int(has_mask[src == f].sum()) == int(mc.loc[f, "queries_affected"]), f
        assert not (changed & ~has_mask).any()
        R = qi[["site", "role", "pos", "source", "qid"]].assign(changed=changed, has_mask=has_mask)
        R.to_parquet(out_p / f"reach_{r}.parquet", index=False)
        log.info(f"{r}: reach changed (test) " + ", ".join(f"{f} {int(((src == f) & changed & (qi.role.to_numpy() == 'test')).sum())}" for f in F)
                 + f"; total test {int((changed & (qi.role.to_numpy() == 'test')).sum())}, has_mask test {int((has_mask & (qi.role.to_numpy() == 'test')).sum())}")

        qtext = dict(zip(zip(Q.source, Q.qid.astype(str)), Q.text))
        E = torch.from_numpy(unit(np.load(p(cfg, "p4_emb", r, "q.npy"))[qi.qrow.to_numpy()])).cuda()
        Tip = T1.copy()
        mrows = []
        for f in F:
            j = sidx[smap[f]]
            rs = np.flatnonzero(src == f)
            pr = []
            for i in rs:
                hit = set(twin_idx[f].get(norm_title(qtext[(f, qid[i])]), []))
                for d in pmap.get(f, {}).get(qid[i], []):
                    hit |= set(twin_idx[f].get(norm_title(corpus_title[f].get(d, "")), []))
                pr += [(i, b) for b in sorted(hit)]
            Bg = torch.from_numpy(unit(np.load(p(cfg, "a0_emb", r, f"{smap[f]}__bank.npy")))).cuda()
            S = E[torch.from_numpy(rs).cuda()] @ Bg.T
            top_un = S.topk(k, dim=1).values.cpu().numpy()
            assert np.abs(top_un - T1[rs, j]).max() < 1e-5, (f, np.abs(top_un - T1[rs, j]).max())
            if pr:
                loc = {g: n for n, g in enumerate(rs)}
                S[torch.tensor([loc[a] for a, _ in pr]).cuda(), torch.tensor([b for _, b in pr]).cuda()] = float("-inf")
            Tip[rs, j] = S.topk(k, dim=1).values.cpu().numpy()
            ch_ = (Tip[rs, j] != T1[rs, j]).any(1)
            mrows.append({"retriever": r, "forum": f, "masked_pairs": len(pr), "queries_masked": len({a for a, _ in pr}), "queries": len(rs),
                          "queries_test_top10_changed": int((ch_ & (qi.role.to_numpy()[rs] == "test")).sum())})
            del Bg, S
        assert np.isfinite(Tip).all()
        od = emb_ip / r
        od.mkdir(parents=True, exist_ok=True)
        np.save(od / "topk.npy", Tip)
        qi.to_parquet(od / "q_index.parquet", index=False)
        save_json(sites, od / "sites.json")
        for c in sorted(e1.glob("centroids_*.npy")):
            shutil.copy(c, od / c.name)
        meta = run_metadata(cfg, f"A8.04_extras_build:{r}")
        meta["inputs"] = fingerprint([e1 / "topk.npy", e0 / "topk.npy", e1 / "q_index.parquet", p(cfg, "p4_emb", r, "q.npy"),
                                      p(cfg, "p4_processed", "queries.parquet"), p(cfg, "p6_processed", "banks.parquet"),
                                      p(cfg, "p6_processed", "positives.parquet"), p(cfg, "a0_processed_src", "items.parquet")]
                                     + [p(cfg, "a0_emb", r, f"{smap[f]}__bank.npy") for f in F]
                                     + [PROJECT_ROOT / "data" / "raw" / "cqadupstack" / f / "corpus.jsonl" for f in F])
        meta["outputs"] = fingerprint([od / "topk.npy", out_p / f"reach_{r}.parquet"])
        save_json(meta, od / "meta_04.json")
        M = pd.DataFrame(mrows)
        fp_ = out_p / "ip_mask_counts.csv"
        old = pd.read_csv(fp_) if fp_.exists() else M.iloc[:0]
        pd.concat([old[old.retriever != r], M], ignore_index=True).to_csv(fp_, index=False)
        log.info(f"{r}: (i') title mask: queries masked {int(M.queries_masked.sum())}, pairs {int(M.masked_pairs.sum())}, test queries whose twin top-10 changed {int(M.queries_test_top10_changed.sum())}")


if __name__ == "__main__":
    main()
