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
from oodrag.common import (fingerprint, get_logger, load_config, load_json, model_fingerprint, p,
                           run_metadata, save_json, set_determinism)

set_determinism()
import torch

from oodrag.encoders import DenseEncoder, length_batches


def encode(enc: DenseEncoder, texts: list[str], max_tokens: int) -> np.ndarray:
    L = enc.lengths(texts)
    out = np.empty((len(texts), enc.dim), np.float32)
    for b in length_batches(L, max_tokens):
        out[b] = enc.encode_batch([texts[i] for i in b])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("P2.02", p(cfg, "results", "logs", "02_encode.log"))
    proc = p(cfg, "processed")
    inputs = [proc / f"{ds}_{k}.parquet" for ds in cfg["datasets"] for k in ("docs", "queries")] + [proc / "bank.parquet"]
    for r, rc in cfg["retrievers"].items():
        d = p(cfg, "emb", r)
        d.mkdir(parents=True, exist_ok=True)
        mdir = p(cfg, "models", rc["path"])
        fp = fingerprint(inputs)
        fp["retriever_config"] = hashlib.sha256(json.dumps(rc, sort_keys=True).encode()).hexdigest()
        fp["model_files"] = model_fingerprint(mdir)
        if (d / "meta.json").exists() and load_json(d / "meta.json").get("upstream_fingerprint") == fp:
            log.info(f"{r}: up to date, skip")
            continue
        t0 = time.time()
        enc = DenseEncoder(mdir, rc["pooling"], rc["normalize"], rc["max_len"], dtype=getattr(torch, rc["dtype"]))
        timing = {}
        for ds in cfg["datasets"]:
            t1 = time.time()
            docs = pd.read_parquet(proc / f"{ds}_docs.parquet")["text"].tolist()
            qs = [rc["query_prefix"] + q for q in pd.read_parquet(proc / f"{ds}_queries.parquet")["text"].tolist()]
            np.save(d / f"{ds}_docs.npy", encode(enc, docs, rc["max_tokens_per_batch"]))
            np.save(d / f"{ds}_q.npy", encode(enc, qs, rc["max_tokens_per_batch"]))
            timing[ds] = (time.time() - t1) / 60
            log.info(f"{r} {ds}: {len(docs)} docs, {len(qs)} queries in {timing[ds]:.1f} min")
        bank = [rc["query_prefix"] + q for q in pd.read_parquet(proc / "bank.parquet")["text"].tolist()]
        np.save(d / "bank.npy", encode(enc, bank, rc["max_tokens_per_batch"]))
        meta = run_metadata(cfg, f"02_encode:{r}")
        meta.update({"upstream_fingerprint": fp, "dtype": rc["dtype"], "minutes_per_dataset": timing,
                     "minutes_total": (time.time() - t0) / 60})
        save_json(meta, d / "meta.json")
        log.info(f"{r}: done in {(time.time() - t0) / 60:.1f} min")
        del enc
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
