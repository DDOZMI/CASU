from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("A2.01", p(cfg, "results", "logs", "01_tagvec.log"))
    out = p(cfg, "emb")
    out.mkdir(parents=True, exist_ok=True)
    sites = load_json(p(cfg, "a1_emb", cfg["detector"], "sites.json"))
    ipath = p(cfg, "a0_processed", "items.parquet")
    it = pd.read_parquet(ipath, columns=["site", "role", "pos", "tags"])
    it = it[it.role == "bank"]
    docs = []
    for s in sites:
        b = it[it.site == s].sort_values("pos")
        assert (b.pos.to_numpy() == np.arange(3000)).all()
        docs += [sorted(set(json.loads(t))) for t in b.tags]
    assert all(len(d) > 0 for d in docs)
    vocab = sorted({t for d in docs for t in d})
    vi = {t: i for i, t in enumerate(vocab)}
    ind = np.array([vi[t] for d in docs for t in d], np.int32)
    ptr = np.r_[0, np.cumsum([len(d) for d in docs])].astype(np.int64)
    M = sp.csr_matrix((np.ones(len(ind), np.float32), ind, ptr), shape=(len(docs), len(vocab)))
    sp.save_npz(out / "tag_incidence.npz", M)
    save_json(vocab, out / "tag_vocab.json")
    meta = run_metadata(cfg, "A2.01_tagvec")
    meta["inputs"] = fingerprint([ipath, p(cfg, "a1_emb", cfg["detector"], "sites.json")])
    meta.update({"n_docs": int(M.shape[0]), "n_tags": int(M.shape[1]), "mean_tags_per_doc": float(M.sum() / M.shape[0])})
    save_json(meta, out / "meta_01.json")
    log.info(f"tag incidence {M.shape}, nnz {M.nnz}")


if __name__ == "__main__":
    main()
