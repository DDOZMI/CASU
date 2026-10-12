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
    ap.add_argument("--part", choices=["main", "subset"], required=True)
    ap.add_argument("--smoke-sites", type=int, default=0, help="smoke: only the first n sites, outputs under emb/_smoke")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger(f"A0.03.{args.part}", p(cfg, "results", "logs", f"03_encode_{args.part}.log"))
    cfg4 = load_config(p(cfg, "p4_config"))
    ipath = p(cfg, "processed", "items.parquet")
    items = pd.read_parquet(ipath, columns=["site", "role", "pos", "text"])
    all_sites = list(dict.fromkeys(items.site))
    subset = load_json(p(cfg, "processed", "subset.json"))["sites"]
    sites = all_sites if args.part == "main" else subset
    if args.smoke_sites:
        sites = sites[: args.smoke_sites]
    emb_root = p(cfg, "emb") / "_smoke" if args.smoke_sites else p(cfg, "emb")
    for r, roles in cfg["encode"][args.part].items():
        rc = cfg4["retrievers"][r]
        d = emb_root / r
        d.mkdir(parents=True, exist_ok=True)
        mdir = p(cfg, "models", rc["path"])
        fp = fingerprint([ipath, p(cfg, "processed", "subset.json")])
        fp["retriever_config"] = hashlib.sha256(json.dumps(rc, sort_keys=True).encode()).hexdigest()
        fp["model_files"] = model_fingerprint(mdir)
        fp["roles"], fp["sites"] = roles, sites
        files = [d / f"{s}__{ro}.npy" for s in sites for ro in roles]
        mpath = d / f"meta_{args.part}.json"
        if mpath.exists() and load_json(mpath).get("upstream_fingerprint") == fp and all(f.exists() for f in files):
            log.info(f"{r}: up to date, skip")
            continue
        t0 = time.time()
        enc = DenseEncoder(mdir, rc["pooling"], rc["normalize"], rc["max_len"], dtype=getattr(torch, rc["dtype"]))
        n_done = 0
        for s in sites:
            for ro in roles:
                sub = items[(items.site == s) & (items.role == ro)].sort_values("pos")
                assert (sub.pos.to_numpy() == np.arange(len(sub))).all()
                tx = sub.text.tolist()
                if ro in ("test", "cal"):
                    tx = [rc["query_prefix"] + t for t in tx]
                L = enc.lengths(tx)
                E = np.empty((len(tx), enc.dim), np.float32)
                for bb in length_batches(L, rc["max_tokens_per_batch"]):
                    E[bb] = enc.encode_batch([tx[i] for i in bb])
                assert np.isfinite(E).all()
                np.save(d / f"{s}__{ro}.npy", E)
                n_done += len(E)
            log.info(f"{r} {s}: done ({n_done} texts, {(time.time() - t0) / 60:.1f} min)")
        meta = run_metadata(cfg, f"A0.03_encode:{args.part}:{r}")
        meta.update({"upstream_fingerprint": fp, "dtype": rc["dtype"], "n_texts": n_done, "minutes": (time.time() - t0) / 60})
        save_json(meta, mpath)
        del enc
        torch.cuda.empty_cache()
        log.info(f"{r}: {n_done} texts in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
