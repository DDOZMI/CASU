from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import (fingerprint, get_logger, load_config, load_json, model_fingerprint, p, run_metadata,
                           save_json, set_determinism, stable_seed)

set_determinism()
import torch

from oodrag.encoders import DenseEncoder, length_batches


def encode(enc, tx, budget):
    L = enc.lengths(tx)
    E = np.empty((len(tx), enc.dim), np.float32)
    for bb in length_batches(L, budget):
        E[bb] = enc.encode_batch([tx[i] for i in bb])
    assert np.isfinite(E).all()
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--smoke", type=int, default=0, help="smoke: only the first n texts of each, outputs under emb/_smoke")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("A12.00", p(cfg, "results", "logs", "00_encode.log"))
    ec = cfg["encode"]
    cfg4 = load_config(p(cfg, "p4_config"))
    proc = p(cfg, "processed")
    proc.mkdir(parents=True, exist_ok=True)
    wsrc = p(cfg, "a6_processed", "corpus_wikipedia.parquet")
    W = pd.read_parquet(wsrc)
    perm = np.random.default_rng(stable_seed(cfg["seed"], "A12_wiki_pool")).permutation(len(W))[: ec["wiki_pool_n"]]
    wp = pd.DataFrame({"pool_rank": np.arange(len(perm)), "doc_id": W.doc_id.to_numpy()[perm], "text": W.text.to_numpy()[perm]})
    wpath = proc / "wiki_pool.parquet"
    if not args.smoke:
        wp.to_parquet(wpath, index=False)
    bpath = p(cfg, "p2_processed", "bank.parquet")
    bank = pd.read_parquet(bpath)
    msq_t, wiki_t = bank.text.tolist(), wp.text.tolist()
    if args.smoke:
        msq_t, wiki_t = msq_t[: args.smoke], wiki_t[: args.smoke]
    emb_root = p(cfg, "emb") / "_smoke" if args.smoke else p(cfg, "emb")
    for r in ec["retrievers"]:
        rc = cfg4["retrievers"][r]
        d = emb_root / r
        d.mkdir(parents=True, exist_ok=True)
        mdir = p(cfg, "models", rc["path"])
        fp = fingerprint([bpath, wsrc])
        fp["wiki_pool_idx_sha256"] = hashlib.sha256(perm.tobytes()).hexdigest()
        fp["retriever_config"] = hashlib.sha256(json.dumps(rc, sort_keys=True).encode()).hexdigest()
        fp["model_files"] = model_fingerprint(mdir)
        fp["smoke"] = args.smoke
        mpath = d / "meta.json"
        if mpath.exists() and load_json(mpath).get("upstream_fingerprint") == fp and (d / "msq.npy").exists() and (d / "wiki.npy").exists():
            log.info(f"{r}: up to date, skip")
            continue
        t0 = time.time()
        enc = DenseEncoder(mdir, rc["pooling"], rc["normalize"], rc["max_len"], dtype=getattr(torch, rc["dtype"]))
        Eq = encode(enc, [rc["query_prefix"] + t for t in msq_t], rc["max_tokens_per_batch"])
        np.save(d / "msq.npy", Eq)
        Ew = encode(enc, wiki_t, rc["max_tokens_per_batch"])
        np.save(d / "wiki.npy", Ew)
        p2 = p(cfg, "p2_emb", r, "bank.npy")
        agree = None
        if p2.exists():
            P2 = np.load(p2)[: len(Eq)]
            u = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
            cs = (u(P2) * u(Eq)).sum(1)
            agree = {"min_cos": float(cs.min()), "median_cos": float(np.median(cs))}
            log.info(f"{r}: msq vs P2 bank embeddings: cosine min {cs.min():.6f}, median {np.median(cs):.6f}")
        meta = run_metadata(cfg, f"A12.00_encode:{r}")
        meta.update({"upstream_fingerprint": fp, "dtype": rc["dtype"], "n_msq": len(Eq), "n_wiki": len(Ew),
                     "p2_bank_agreement": agree, "minutes": (time.time() - t0) / 60})
        save_json(meta, mpath)
        del enc
        torch.cuda.empty_cache()
        log.info(f"{r}: msq {Eq.shape}, wiki {Ew.shape} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
