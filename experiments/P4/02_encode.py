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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("P4.02", p(cfg, "results", "logs", "02_encode.log"))
    qpath = p(cfg, "processed", "queries.parquet")
    texts = pd.read_parquet(qpath)["text"].tolist()
    for r, rc in cfg["retrievers"].items():
        d = p(cfg, "emb", r)
        d.mkdir(parents=True, exist_ok=True)
        mdir = p(cfg, "models", rc["path"])
        fp = fingerprint([qpath])
        fp["retriever_config"] = hashlib.sha256(json.dumps(rc, sort_keys=True).encode()).hexdigest()
        fp["model_files"] = model_fingerprint(mdir)
        if (d / "meta.json").exists() and load_json(d / "meta.json").get("upstream_fingerprint") == fp:
            log.info(f"{r}: up to date, skip")
            continue
        t0 = time.time()
        enc = DenseEncoder(mdir, rc["pooling"], rc["normalize"], rc["max_len"], dtype=getattr(torch, rc["dtype"]))
        tx = [rc["query_prefix"] + t for t in texts]
        L = enc.lengths(tx)
        E = np.empty((len(tx), enc.dim), np.float32)
        for b in length_batches(L, rc["max_tokens_per_batch"]):
            E[b] = enc.encode_batch([tx[i] for i in b])
        assert np.isfinite(E).all()
        np.save(d / "q.npy", E)
        np.save(d / "lengths.npy", L)
        meta = run_metadata(cfg, f"02_encode:{r}")
        meta.update({"upstream_fingerprint": fp, "dtype": rc["dtype"], "n": len(tx), "minutes": (time.time() - t0) / 60})
        save_json(meta, d / "meta.json")
        del enc
        torch.cuda.empty_cache()
        log.info(f"{r}: {len(tx)} queries in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
