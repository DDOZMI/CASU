from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

from oodrag.encoders import DenseEncoder

spec = importlib.util.spec_from_file_location("a15_02", Path(__file__).with_name("02_encode.py"))
E2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--retrievers", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("A15.02b", p(cfg, "results", "logs", "02b_encode_yahoo.log"))
    cfg4 = load_config(p(cfg, "p4_config"))
    src = p(cfg, "processed", "yahoo_groups.parquet")
    tx = pd.read_parquet(src, columns=["text"]).text.tolist()
    if args.smoke:
        tx = tx[: args.smoke]
    root = p(cfg, "emb") / ("_smoke" if args.smoke else "")
    for r in [x for x in cfg["encode"]["wiki"] if not args.retrievers or x in args.retrievers]:
        rc = cfg4["retrievers"][r]
        d = root / r
        d.mkdir(parents=True, exist_ok=True)
        mdir = p(cfg, "models", rc["path"])
        fp = fingerprint([src])
        fp["retriever_config"] = hashlib.sha256(json.dumps(rc, sort_keys=True).encode()).hexdigest()
        fp["model_files"] = model_fingerprint(mdir)
        fp["smoke"] = args.smoke
        mpath = d / "meta_yahoo.json"
        if mpath.exists() and load_json(mpath).get("upstream_fingerprint") == fp and (d / "yahoo.npy").exists():
            log.info(f"{r}: up to date, skip")
            continue
        t0 = time.time()
        enc = DenseEncoder(mdir, rc["pooling"], rc["normalize"], rc["max_len"], dtype=getattr(torch, rc["dtype"]))
        E = E2.encode(enc, tx, rc["max_tokens_per_batch"])
        np.save(d / "yahoo.npy", E)
        meta = run_metadata(cfg, f"A15.02b_encode_yahoo:{r}")
        meta.update({"upstream_fingerprint": fp, "n": len(E), "minutes": (time.time() - t0) / 60})
        save_json(meta, mpath)
        del enc
        torch.cuda.empty_cache()
        log.info(f"{r}: yahoo {E.shape} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
