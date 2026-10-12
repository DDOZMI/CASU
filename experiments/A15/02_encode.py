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
from oodrag.common import fingerprint, get_logger, load_config, load_json, model_fingerprint, p, run_metadata, save_json, set_determinism

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
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("A15.02", p(cfg, "results", "logs", "02_encode.log"))
    ec = cfg["encode"]
    cfg4 = load_config(p(cfg, "p4_config"))
    proc = p(cfg, "processed")
    W = pd.read_parquet(proc / "wiki_groups.parquet", columns=["text"]).text.tolist()
    M = pd.read_parquet(proc / "ms_pool.parquet", columns=["text"]).text.tolist()
    Q = pd.read_parquet(proc / "ood_queries.parquet", columns=["text"]).text.tolist()
    items = pd.read_parquet(p(cfg, "a0_processed", "items.parquet"), columns=["site", "role", "pos", "text"])
    root = p(cfg, "emb") / ("_smoke" if args.smoke else "")
    cut = (lambda x: x[: args.smoke]) if args.smoke else (lambda x: x)
    for r in ec["wiki"]:
        rc = cfg4["retrievers"][r]
        d = root / r
        d.mkdir(parents=True, exist_ok=True)
        mdir = p(cfg, "models", rc["path"])
        jobs = {"wiki": (cut(W), False), "ms": (cut(M[: ec["ms"][r]]), False), "ood": (cut(Q), True)}
        if r in ec["se_test_second"]:
            sub = items[items.role == "test"].sort_values(["site", "pos"])
            a1_sites = load_json(Path("data/emb/A1/bge-m3/sites.json"))
            for s in a1_sites:
                jobs[f"se_test_{s}"] = (cut(sub[sub.site == s].text.tolist()), True)
        fp = fingerprint([proc / "wiki_groups.parquet", proc / "ms_pool.parquet", proc / "ood_queries.parquet", p(cfg, "a0_processed", "items.parquet")])
        fp["retriever_config"] = hashlib.sha256(json.dumps(rc, sort_keys=True).encode()).hexdigest()
        fp["model_files"] = model_fingerprint(mdir)
        fp["jobs"] = sorted(jobs)
        fp["smoke"] = args.smoke
        mpath = d / "meta.json"
        if mpath.exists() and load_json(mpath).get("upstream_fingerprint") == fp and all((d / f"{k}.npy").exists() for k in jobs):
            log.info(f"{r}: up to date, skip")
            continue
        t0 = time.time()
        enc = DenseEncoder(mdir, rc["pooling"], rc["normalize"], rc["max_len"], dtype=getattr(torch, rc["dtype"]))
        counts = {}
        for k, (tx, is_q) in jobs.items():
            E = encode(enc, [rc["query_prefix"] + t for t in tx] if is_q else tx, rc["max_tokens_per_batch"])
            np.save(d / f"{k}.npy", E)
            counts[k] = len(E)
            if not k.startswith("se_test_"):
                log.info(f"{r}: {k} {E.shape} ({(time.time() - t0) / 60:.1f} min)")
        meta = run_metadata(cfg, f"A15.02_encode:{r}")
        meta.update({"upstream_fingerprint": fp, "dtype": rc["dtype"], "counts": counts, "minutes": (time.time() - t0) / 60})
        save_json(meta, mpath)
        del enc
        torch.cuda.empty_cache()
        log.info(f"{r}: done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
