from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import (check_fingerprint, fingerprint, get_logger, load_config, load_json,
                           model_fingerprint, p, run_metadata, save_json, set_determinism, stable_seed)

set_determinism()
import torch

from oodrag.encoders import DenseEncoder, doc_text, length_batches


def load_corpus_texts(cfg) -> list[str]:
    t = pq.read_table(p(cfg, "raw", "beir/nq/corpus/corpus-00000-of-00001.parquet"), columns=["title", "text"])
    return [doc_text(a, b) for a, b in zip(t.column("title").to_pylist(), t.column("text").to_pylist())]


def make_encoder(cfg, r) -> DenseEncoder:
    rc = cfg["retrievers"][r]
    return DenseEncoder(p(cfg, "models", rc["path"]), rc["pooling"], rc["normalize"], rc["max_len"])


def encode_texts(enc: DenseEncoder, texts: list[str], max_tokens: int) -> np.ndarray:
    L = enc.lengths(texts)
    out = np.empty((len(texts), enc.dim), np.float32)
    for b in length_batches(L, max_tokens):
        out[b] = enc.encode_batch([texts[i] for i in b])
    return out


def get_lengths(enc, texts, path: Path, key: dict, log) -> np.ndarray:
    side = path.with_suffix(".json")
    if path.exists() and side.exists() and load_json(side) == key:
        return np.load(path)
    t0 = time.time()
    L = enc.lengths(texts)
    np.save(path, L)
    save_json(key, side)
    log.info(f"token lengths computed in {time.time() - t0:.0f}s (mean {L.mean():.1f}, max {L.max()})")
    return L


def benchmark(cfg, log):
    texts = load_corpus_texts(cfg)
    res = {"meta": run_metadata(cfg, "02_encode:benchmark"), "retrievers": {}}
    for r, rc in cfg["retrievers"].items():
        enc = make_encoder(cfg, r)
        d = p(cfg, "emb", r)
        d.mkdir(parents=True, exist_ok=True)
        L = get_lengths(enc, texts, d / "lengths.npy", corpus_fp(cfg, r), log)
        rng = np.random.default_rng(stable_seed(cfg["seed"], "benchmark"))
        sample = np.sort(rng.choice(len(texts), size=cfg["benchmark_sample"], replace=False))
        st = [texts[i] for i in sample]
        Ls = L[sample]
        batches = length_batches(Ls, rc["max_tokens_per_batch"])
        enc.encode_batch([st[i] for i in batches[len(batches) // 2]])
        torch.cuda.synchronize()
        t0 = time.time()
        for b in batches:
            enc.encode_batch([st[i] for i in b])
        torch.cuda.synchronize()
        dt = time.time() - t0
        est = dt * float(L.sum()) / float(Ls.sum())
        res["retrievers"][r] = {"sample_passages": len(sample), "sample_tokens": int(Ls.sum()),
                                "seconds": dt, "passages_per_s": len(sample) / dt,
                                "corpus_tokens": int(L.sum()), "est_full_corpus_hours": est / 3600}
        log.info(f"{r}: {len(sample) / dt:.0f} passages/s, est. full corpus {est / 3600:.2f} h")
        del enc
        torch.cuda.empty_cache()
    res["est_total_hours"] = sum(v["est_full_corpus_hours"] for v in res["retrievers"].values())
    save_json(res, p(cfg, "results", "encode_benchmark.json"))


_MODEL_FP: dict = {}


def corpus_fp(cfg, r) -> dict:
    rc = cfg["retrievers"][r]
    fp = fingerprint([p(cfg, "raw", "beir/nq/corpus/corpus-00000-of-00001.parquet")])
    fp["retriever_config"] = hashlib.sha256(json.dumps(rc, sort_keys=True).encode()).hexdigest()
    if r not in _MODEL_FP:
        _MODEL_FP[r] = model_fingerprint(p(cfg, "models", rc["path"]))
    fp["model_files"] = _MODEL_FP[r]
    return fp


def query_fp(cfg, r) -> dict:
    fp = corpus_fp(cfg, r)
    fp.update(fingerprint([p(cfg, "processed", "queries_nq.parquet"), p(cfg, "processed", "bank.parquet"),
                           p(cfg, "processed", "external.parquet")]))
    return fp


def encode_queries(cfg, r, enc, log):
    d = p(cfg, "emb", r)
    fp = query_fp(cfg, r)
    qm = d / "q_meta.json"
    if qm.exists() and load_json(qm).get("upstream_fingerprint") == fp:
        log.info(f"{r}: query embeddings up to date, skip")
        return
    rc = cfg["retrievers"][r]
    if enc is None:
        enc = make_encoder(cfg, r)
    for name, path in [("q_nq", p(cfg, "processed", "queries_nq.parquet")),
                       ("q_bank", p(cfg, "processed", "bank.parquet")),
                       ("q_ext", p(cfg, "processed", "external.parquet"))]:
        qtexts = pd.read_parquet(path)["text"].astype(str).tolist()
        np.save(d / f"{name}.npy", encode_texts(enc, qtexts, rc["max_tokens_per_batch"]))
    meta = run_metadata(cfg, f"02_encode:queries:{r}")
    meta["upstream_fingerprint"] = fp
    save_json(meta, qm)
    log.info(f"{r}: query embeddings written")


def encode_full(cfg, r, texts, log):
    rc = cfg["retrievers"][r]
    d = p(cfg, "emb", r)
    d.mkdir(parents=True, exist_ok=True)
    fp = corpus_fp(cfg, r)
    if (d / "meta.json").exists():
        check_fingerprint(d / "meta.json", fp, f"emb/{r}")
        log.info(f"{r}: corpus embeddings up to date, skip")
        encode_queries(cfg, r, None, log)
        return
    meta = run_metadata(cfg, f"02_encode:{r}")
    enc = make_encoder(cfg, r)
    L = get_lengths(enc, texts, d / "lengths.npy", fp, log)
    batches = length_batches(L, rc["max_tokens_per_batch"])
    plan_sha = hashlib.sha256(b"".join(b.tobytes() for b in batches)).hexdigest()
    prog_path = d / "progress.json"
    cpath = d / "corpus.f16.npy"
    start = 0
    if prog_path.exists() and cpath.exists():
        prog = load_json(prog_path)
        if prog.get("plan_sha") == plan_sha and prog.get("upstream") == fp:
            start = prog["done_batches"]
            C = np.load(cpath, mmap_mode="r+")
            log.info(f"{r}: resuming at batch {start}/{len(batches)}")
        else:
            raise RuntimeError(f"{r}: progress file does not match current plan/inputs; delete {d} to restart")
    else:
        C = np.lib.format.open_memmap(cpath, mode="w+", dtype=np.float16, shape=(len(texts), enc.dim))
    t0 = time.time()
    for k in range(start, len(batches)):
        b = batches[k]
        C[b] = enc.encode_batch([texts[i] for i in b]).astype(np.float16)
        if (k + 1) % 200 == 0 or k + 1 == len(batches):
            C.flush()
            torch.cuda.empty_cache()
            save_json({"plan_sha": plan_sha, "upstream": fp, "done_batches": k + 1, "n_batches": len(batches)},
                      prog_path)
            el = time.time() - t0
            rate = (k + 1 - start) / el
            log.info(f"{r}: batch {k + 1}/{len(batches)}  {el / 60:.1f} min, ETA {(len(batches) - k - 1) / rate / 60:.1f} min")
    del C
    Cr = np.load(cpath, mmap_mode="r")
    n_nonfinite = int(sum((~np.isfinite(Cr[i:i + 500000].astype(np.float32))).sum()
                          for i in range(0, len(Cr), 500000)))
    meta.update({"upstream_fingerprint": fp, "n_batches": len(batches), "batch_plan_sha": plan_sha,
                 "dim": enc.dim, "corpus_nonfinite_values": n_nonfinite,
                 "encode_minutes_this_run": (time.time() - t0) / 60})
    if n_nonfinite:
        raise RuntimeError(f"{r}: {n_nonfinite} non-finite values in fp16 corpus embeddings")
    save_json(meta, d / "meta.json")
    log.info(f"{r}: corpus done")
    encode_queries(cfg, r, enc, log)
    del enc
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--only", default=None, help="encode a single retriever")
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("P0.02", p(cfg, "results", "logs", "02_encode.log"))
    if args.benchmark:
        benchmark(cfg, log)
        return
    texts = load_corpus_texts(cfg)
    for r in cfg["retrievers"]:
        if args.only and r != args.only:
            continue
        encode_full(cfg, r, texts, log)


if __name__ == "__main__":
    main()
