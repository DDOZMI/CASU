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
    args = ap.parse_args()
    cfg = load_config(args.config)
    r, k = cfg["detector"], cfg["k"]
    log = get_logger("A4.01", p(cfg, "results", "logs", "01_l3topk.log"))
    t0 = time.time()
    a1 = p(cfg, "a1_emb", r)
    sites = load_json(a1 / "sites.json")
    qi = pd.read_parquet(a1 / "q_index.parquet")
    a0 = p(cfg, "a0_emb", r)
    Q = np.concatenate([unit(np.load(a0 / f"{s}__{ro}.npy")) for s in sites for ro in ("test", "cal")])
    assert len(Q) == len(qi)
    assert (qi.site.to_numpy() == np.repeat(sites, 1500)).all()
    Qg = torch.from_numpy(Q).cuda()
    T = np.empty((len(Q), len(sites), k), np.float32)
    for j, s in enumerate(sites):
        Bg = torch.from_numpy(unit(np.load(a0 / f"{s}__l3.npy"))).cuda()
        assert Bg.shape[0] == 3000
        for i in range(0, len(Q), 16384):
            T[i:i + 16384, j] = (Qg[i:i + 16384] @ Bg.T).topk(k, dim=1).values.cpu().numpy()
    out = p(cfg, "emb", r)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "topk_l3.npy", T)
    meta = run_metadata(cfg, "A4.01_l3topk")
    meta["inputs"] = fingerprint([a1 / "sites.json", a1 / "q_index.parquet", a0 / "meta_main.json"])
    meta["shape"], meta["minutes"] = list(T.shape), (time.time() - t0) / 60
    save_json(meta, out / "meta_01.json")
    log.info(f"topk_l3 {T.shape} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
