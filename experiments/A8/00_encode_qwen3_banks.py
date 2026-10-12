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

R = "qwen3-emb-0.6b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0-config", default="experiments/A0/config.json")
    ap.add_argument("--verify-site", default=None, help="re-encode an existing A0 Qwen3 bank into --out and compare (settings check)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.a0_config)
    log = get_logger("A8.00", Path("experiments/A8/results/logs/00_encode.log"))
    rc = load_config(p(cfg, "p4_config"))["retrievers"][R]
    ipath = p(cfg, "processed", "items.parquet")
    items = pd.read_parquet(ipath, columns=["site", "role", "pos", "text"])
    sites = list(dict.fromkeys(items.site))
    d = p(cfg, "emb") / R
    missing = [s for s in sites if not (d / f"{s}__bank.npy").exists()]
    ref = None
    if args.verify_site:
        ref = np.load(d / f"{args.verify_site}__bank.npy")
        missing, d = [args.verify_site], Path(args.out)
        d.mkdir(parents=True, exist_ok=True)
    log.info(f"{len(missing)} sites to encode")
    mdir = p(cfg, "models", rc["path"])
    enc = DenseEncoder(mdir, rc["pooling"], rc["normalize"], rc["max_len"], dtype=getattr(torch, rc["dtype"]))
    t0, n_done = time.time(), 0
    for s in missing:
        sub = items[(items.site == s) & (items.role == "bank")].sort_values("pos")
        assert (sub.pos.to_numpy() == np.arange(len(sub))).all()
        tx = sub.text.tolist()
        L = enc.lengths(tx)
        E = np.empty((len(tx), enc.dim), np.float32)
        for bb in length_batches(L, rc["max_tokens_per_batch"]):
            E[bb] = enc.encode_batch([tx[i] for i in bb])
        assert np.isfinite(E).all()
        np.save(d / f"{s}__bank.npy", E)
        n_done += len(E)
        log.info(f"{s}: done ({n_done} texts, {(time.time() - t0) / 60:.1f} min)")
    if ref is not None:
        log.info(f"verify {args.verify_site}: max |diff| vs A0 = {np.abs(E - ref).max():.2e}")
        return
    meta = run_metadata(cfg, "A8.00_encode_qwen3_banks")
    meta.update({"sites": missing, "retriever_config_sha256": hashlib.sha256(json.dumps(rc, sort_keys=True).encode()).hexdigest(),
                 "model_files": model_fingerprint(mdir), "upstream": fingerprint([ipath]), "n_texts": n_done, "minutes": (time.time() - t0) / 60})
    save_json(meta, d / "meta_a8_banks.json")


if __name__ == "__main__":
    main()
