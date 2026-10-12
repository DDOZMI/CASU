from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import PROJECT_ROOT, fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json, set_determinism

set_determinism()
import torch

DEV = "cuda"


def unit_t(X):
    X = torch.as_tensor(np.asarray(X, np.float32), device=DEV)
    return X / X.norm(dim=1, keepdim=True)


def topk_groups(Q, docs, gsize, k, chunk=8192):
    G = docs.shape[0] // gsize
    out = np.empty((Q.shape[0], G, k), np.float32)
    for g in range(G):
        D = docs[g * gsize:(g + 1) * gsize]
        for i in range(0, Q.shape[0], chunk):
            out[i:i + chunk, g] = (Q[i:i + chunk] @ D.T).topk(k, dim=1).values.cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--runs", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("A15.03", p(cfg, "results", "logs", "03_topk.log"))
    k, gs = cfg["k"], cfg["prepare"]["group_size"]
    W = pd.read_parquet(p(cfg, "processed", "wiki_groups.parquet"), columns=["group"])
    wgroups = list(dict.fromkeys(W.group))
    assert (W.group.to_numpy() == np.repeat(wgroups, gs)).all()
    for rn in args.runs or list(cfg["runs"]):
        run = cfg["runs"][rn]
        r = run["retriever"]
        t0 = time.time()
        edir = PROJECT_ROOT / run["emb"]
        sites = load_json(edir / "sites.json")
        qi = pd.read_parquet(edir / "q_index.parquet")
        a0 = p(cfg, "a0_emb", r)
        Qse = unit_t(np.concatenate([np.load(a0 / f"{s}__{ro}.npy") for s in sites for ro in ("test", "cal")]))
        assert Qse.shape[0] == len(qi)
        Qood = unit_t(np.load(p(cfg, "emb", r, "ood.npy")))
        Qall = torch.cat([Qse, Qood])
        out = p(cfg, "emb", "topk", rn)
        out.mkdir(parents=True, exist_ok=True)
        Tse = np.empty((Qood.shape[0], len(sites), k), np.float32)
        Tst = np.asarray(np.load(edir / "topk.npy", mmap_mode="r"))
        maxerr = 0.0
        for j, s in enumerate(sites):
            B = unit_t(np.load(a0 / f"{s}__bank.npy"))
            Tse[:, j] = (Qood @ B.T).topk(k, dim=1).values.cpu().numpy()
            if j < 3:
                chk = (Qse[:2000] @ B.T).topk(k, dim=1).values.cpu().numpy()
                maxerr = max(maxerr, float(np.abs(chk - Tst[:2000, j]).max()))
        assert maxerr < 1e-5, maxerr
        np.save(out / "se_ood.npy", Tse)
        Wd = unit_t(np.load(p(cfg, "emb", r, "wiki.npy")))
        np.save(out / "wiki.npy", topk_groups(Qall, Wd, gs, k))
        del Wd
        Yg = pd.read_parquet(p(cfg, "processed", "yahoo_groups.parquet"), columns=["group"])
        ygroups = list(dict.fromkeys(Yg.group))
        assert (Yg.group.to_numpy() == np.repeat(ygroups, gs)).all()
        Yd = unit_t(np.load(p(cfg, "emb", r, "yahoo.npy")))
        np.save(out / "yahoo.npy", topk_groups(Qall, Yd, gs, k))
        del Yd
        nms = run["ms_chunks"] * gs
        Md = unit_t(np.load(p(cfg, "emb", r, "ms.npy"))[:nms])
        assert Md.shape[0] == nms
        np.save(out / "ms.npy", topk_groups(Qall, Md, gs, k))
        del Md
        torch.cuda.empty_cache()
        save_json({"wiki_groups": wgroups, "yahoo_groups": ygroups, "ms_chunks": run["ms_chunks"], "n_se_queries": len(qi), "n_ood_queries": int(Qood.shape[0])}, out / "groups.json")
        meta = run_metadata(cfg, f"A15.03_topk:{rn}")
        meta["inputs"] = fingerprint([edir / f for f in ("topk.npy", "q_index.parquet", "sites.json")]
                                     + [p(cfg, "emb", r, f) for f in ("wiki.npy", "ms.npy", "ood.npy", "meta.json", "yahoo.npy", "meta_yahoo.json")]
                                     + [p(cfg, "processed", "yahoo_groups.parquet")]
                                     + [a0 / f"{s}__{ro}.npy" for s in sites for ro in ("test", "cal", "bank")])
        meta["se_topk_check"], meta["minutes"] = maxerr, (time.time() - t0) / 60
        save_json(meta, out / "meta_03.json")
        log.info(f"{rn}: done in {(time.time() - t0) / 60:.1f} min (SE topk check {maxerr:.1e})")


if __name__ == "__main__":
    main()
