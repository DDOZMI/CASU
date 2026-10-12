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
    log = get_logger("P6.02", p(cfg, "results", "logs", "02_encode.log"))
    cfg4 = load_config(p(cfg, "p4_config"))
    cfg2 = load_config(p(cfg, "p4_config").parents[1] / "P2" / "config.json")
    assert cfg2["retrievers"] == cfg4["retrievers"], "P2 and P4 encoder settings differ"
    bpath = p(cfg, "processed", "banks.parquet")
    B = pd.read_parquet(bpath)
    sources = list(dict.fromkeys(B["source"]))
    for r, rc in cfg4["retrievers"].items():
        d = p(cfg, "emb", r)
        d.mkdir(parents=True, exist_ok=True)
        mdir = p(cfg, "models", rc["path"])
        fp = fingerprint([bpath] + [p(cfg, "p2_emb", r, "meta.json")])
        fp["retriever_config"] = hashlib.sha256(json.dumps(rc, sort_keys=True).encode()).hexdigest()
        fp["model_files"] = model_fingerprint(mdir)
        if (d / "meta.json").exists() and load_json(d / "meta.json").get("upstream_fingerprint") == fp:
            log.info(f"{r}: up to date, skip")
            continue
        t0 = time.time()
        enc, timing = None, {}
        for s in sources:
            t1 = time.time()
            b = B[B.source == s].sort_values("rank")
            if s in cfg["beir_sources"]:
                docs = pd.read_parquet(p(cfg, "p2_processed", f"{s}_docs.parquet"), columns=["doc_id"])["doc_id"].astype(str).to_numpy()
                idx = b["corpus_idx"].to_numpy()
                assert (docs[idx] == b["doc_id"].to_numpy()).all(), s
                X = np.load(p(cfg, "p2_emb", r, f"{s}_docs.npy"), mmap_mode="r")
                assert len(X) == len(docs)
                E = np.asarray(X[idx], np.float32)
            else:
                if enc is None:
                    enc = DenseEncoder(mdir, rc["pooling"], rc["normalize"], rc["max_len"], dtype=getattr(torch, rc["dtype"]))
                tx = b["text"].tolist()
                L = enc.lengths(tx)
                E = np.empty((len(tx), enc.dim), np.float32)
                for bb in length_batches(L, rc["max_tokens_per_batch"]):
                    E[bb] = enc.encode_batch([tx[i] for i in bb])
            assert np.isfinite(E).all() and len(E) == len(b)
            np.save(d / f"bank_{s}.npy", E)
            timing[s] = (time.time() - t1) / 60
            log.info(f"{r} {s}: {len(E)} docs in {timing[s]:.2f} min")
        meta = run_metadata(cfg, f"02_encode:{r}")
        meta.update({"upstream_fingerprint": fp, "dtype": rc["dtype"], "minutes_per_source": timing,
                     "minutes_total": (time.time() - t0) / 60})
        save_json(meta, d / "meta.json")
        del enc
        torch.cuda.empty_cache()
        log.info(f"{r}: done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
