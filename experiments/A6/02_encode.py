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


def encode(enc, texts, budget):
    L = enc.lengths(texts)
    E = np.empty((len(texts), enc.dim), np.float32)
    for bb in length_batches(L, budget):
        E[bb] = enc.encode_batch([texts[i] for i in bb])
    assert np.isfinite(E).all()
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--smoke", type=int, default=0, help="encode only the first n texts of each set into emb/_smoke")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("A6.02", p(cfg, "results", "logs", "02_encode.log"))
    proc = p(cfg, "processed")
    names = [cfg["corpora"]["kb"]] + cfg["corpora"]["comparison"]
    Q = pd.read_parquet(proc / "queries.parquet")
    for r, rc in cfg["retrievers"].items():
        if r.startswith("_"):
            continue
        d = p(cfg, "emb", "_smoke" if args.smoke else "", r)
        d.mkdir(parents=True, exist_ok=True)
        mdir = p(cfg, "models", rc["path"])
        fp = fingerprint([proc / "queries.parquet"] + [proc / f"corpus_{n}.parquet" for n in names])
        fp["retriever_config"] = hashlib.sha256(json.dumps(rc, sort_keys=True).encode()).hexdigest()
        fp["model_files"] = model_fingerprint(mdir)
        if not args.smoke and (d / "meta.json").exists() and load_json(d / "meta.json").get("upstream_fingerprint") == fp:
            log.info(f"{r}: up to date, skip")
            continue
        t0 = time.time()
        enc = DenseEncoder(mdir, rc["pooling"], rc["normalize"], rc["max_len"], dtype=getattr(torch, rc["dtype"]))
        for n in names:
            tx = pd.read_parquet(proc / f"corpus_{n}.parquet").text.tolist()
            tx = tx[: args.smoke] if args.smoke else tx
            np.save(d / f"corpus_{n}.npy", encode(enc, tx, rc["max_tokens_per_batch"]))
            log.info(f"{r} corpus {n}: {len(tx)} in {(time.time() - t0) / 60:.1f} min")
        for col in ("question", "question_options"):
            tx = [rc["query_prefix"] + t for t in Q[col].tolist()]
            tx = tx[: args.smoke] if args.smoke else tx
            np.save(d / f"queries_{col}.npy", encode(enc, tx, rc["max_tokens_per_batch"]))
        meta = run_metadata(cfg, f"A6.02_encode:{r}")
        meta.update({"upstream_fingerprint": fp, "minutes": (time.time() - t0) / 60})
        save_json(meta, d / "meta.json")
        del enc
        torch.cuda.empty_cache()
        log.info(f"{r}: done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
