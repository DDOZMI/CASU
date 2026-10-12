from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import get_logger, load_config, p, run_metadata, save_json, sha256_file, stable_seed
from oodrag.encoders import doc_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    cfg = load_config(ap.parse_args().config)
    log = get_logger("P9.01", p(cfg, "results", "logs", "01_prepare.log"))
    src, dst = p(cfg, "rag01_msmarco_corpus"), p(cfg, "raw", cfg["msmarco"]["local"])
    assert src.stat().st_size == cfg["msmarco"]["expected_size"]
    h = sha256_file(src)
    assert h == cfg["msmarco"]["expected_sha256"], h
    if not (dst.exists() and sha256_file(dst) == h):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        assert sha256_file(dst) == h
    t = pq.read_table(dst, columns=["_id", "title", "text"])
    N = t.num_rows
    idx = np.sort(np.random.default_rng(stable_seed(cfg["seed"], "P9_pool")).choice(N, cfg["pool_size"], replace=False))
    sub = t.take(idx)
    P = pd.DataFrame({"pool_rank": np.arange(len(idx)), "corpus_idx": idx, "doc_id": [str(x) for x in sub.column("_id").to_pylist()],
                      "text": [doc_text(a, b) for a, b in zip(sub.column("title").to_pylist(), sub.column("text").to_pylist())]})
    out = p(cfg, "processed")
    out.mkdir(parents=True, exist_ok=True)
    P.to_parquet(out / "pool.parquet", index=False)
    meta = run_metadata(cfg, "01_prepare")
    meta.update({"msmarco_sha256": h, "msmarco_rows": N, "pool": len(P), "empty_texts": int((P.text.str.len() == 0).sum())})
    save_json(meta, out / "meta_01.json")
    log.info(f"msmarco {N:,} rows sha256 {h[:12]}; pool {len(P):,}")


if __name__ == "__main__":
    main()
