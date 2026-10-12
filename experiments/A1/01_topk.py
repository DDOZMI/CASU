from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json, set_determinism

set_determinism()
import torch


def unit(X):
    X = np.asarray(X, np.float32)
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--retriever", default=None, help="default: config detector")
    ap.add_argument("--smoke-sites", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    r = args.retriever or cfg["detector"]
    log = get_logger("A1.01", p(cfg, "results", "logs", f"01_topk_{r}.log"))
    k = cfg["k"]
    items = pd.read_parquet(p(cfg, "a0_processed", "items.parquet"), columns=["site", "role", "pos"])
    sites = list(dict.fromkeys(items.site))
    if cfg.get("site_set", "all") == "subset":
        sub = set(load_json(p(cfg, "a0_processed", "subset.json"))["sites"])
        sites = [s for s in sites if s in sub]
    if args.smoke_sites:
        sites = sites[: args.smoke_sites]
    emb_in = p(cfg, "a0_emb") / ("_smoke" if args.smoke_sites else "")
    out = p(cfg, "emb", "_smoke" if args.smoke_sites else "", r)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    qi = [(s, ro, j) for s in sites for ro in ("test", "cal") for j in range(int(((items.site == s) & (items.role == ro)).sum()))]
    Q = np.concatenate([unit(np.load(emb_in / r / f"{s}__{ro}.npy")) for s in sites for ro in ("test", "cal")])
    assert len(Q) == len(qi)
    Qg = torch.from_numpy(Q).cuda()
    T = np.empty((len(Q), len(sites), k), np.float32)
    for j, s in enumerate(sites):
        Bg = torch.from_numpy(unit(np.load(emb_in / r / f"{s}__bank.npy"))).cuda()
        assert Bg.shape[0] == 3000
        for i in range(0, len(Q), 16384):
            T[i:i + 16384, j] = (Qg[i:i + 16384] @ Bg.T).topk(k, dim=1).values.cpu().numpy()
        del Bg
    np.save(out / "topk.npy", T)
    pd.DataFrame(qi, columns=["site", "role", "pos"]).to_parquet(out / "q_index.parquet", index=False)
    save_json(sites, out / "sites.json")
    for rr in sorted({r, cfg["second_distance_encoder"][r]}):
        files = [emb_in / rr / f"{s}__bank.npy" for s in sites]
        if all(f.exists() for f in files):
            C = np.stack([unit(unit(np.load(f)).mean(0)) for f in files])
            np.save(out / f"centroids_{rr}.npy", C)
        else:
            log.info(f"centroids {rr}: banks missing, skipped")
    meta = run_metadata(cfg, f"A1.01_topk:{r}")
    metas = [emb_in / r / f"meta_{part}.json" for part in ("main", "subset") if (emb_in / r / f"meta_{part}.json").exists()]
    meta["inputs"] = {"a0_emb_meta": fingerprint(metas), "items": fingerprint([p(cfg, "a0_processed", "items.parquet")])}
    meta["shape"] = list(T.shape)
    meta["minutes"] = (time.time() - t0) / 60
    save_json(meta, out / "meta_01.json")
    log.info(f"{r}: topk {T.shape} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
