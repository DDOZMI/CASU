from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import get_logger, load_config, p, run_metadata, save_json, sha256_file, stable_seed
from oodrag.encoders import doc_text


def main():
    from huggingface_hub import hf_hub_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("P2.01", p(cfg, "results", "logs", "01_prepare.log"))
    out = p(cfg, "processed")
    out.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(cfg, "01_prepare")
    meta["inputs_sha256"], meta["counts"] = {}, {}

    for ds, e in cfg["datasets"].items():
        name = e["repo"].split("/")[1]
        files = {}
        for repo, rev, fn in [(e["repo"], e["revision"], "corpus/corpus-00000-of-00001.parquet"),
                              (e["repo"], e["revision"], "queries/queries-00000-of-00001.parquet"),
                              (e["qrels_repo"], e["qrels_revision"], "test.tsv")]:
            local = p(cfg, "raw", "beir", repo.split("/")[1])
            f = Path(hf_hub_download(repo, fn, repo_type="dataset", revision=rev, local_dir=str(local)))
            files[fn] = f
            meta["inputs_sha256"][str(f.relative_to(p(cfg, "raw")))] = sha256_file(f)
        C = pd.read_parquet(files["corpus/corpus-00000-of-00001.parquet"])
        docs = pd.DataFrame({"doc_id": C["_id"].astype(str),
                             "text": [doc_text(a, b) for a, b in zip(C["title"].fillna(""), C["text"].fillna(""))]})
        assert docs["doc_id"].is_unique
        Qall = pd.read_parquet(files["queries/queries-00000-of-00001.parquet"])
        qtext = dict(zip(Qall["_id"].astype(str), Qall["text"].astype(str)))
        qr = pd.read_csv(files["test.tsv"], sep="\t", dtype={0: str, 1: str})
        qr.columns = ["qid", "doc_id", "score"]
        qr["score"] = qr["score"].astype(int)
        n_qr = len(qr)
        n_missing_doc = int((~qr["doc_id"].isin(set(docs["doc_id"]))).sum())
        pos_q = sorted(qr.loc[qr["score"] > 0, "qid"].unique())
        missing_text = [q for q in pos_q if q not in qtext]
        assert not missing_text, f"{ds}: {len(missing_text)} test qids without text"
        qr = qr[qr["qid"].isin(set(pos_q))]
        docs.to_parquet(out / f"{ds}_docs.parquet", index=False)
        pd.DataFrame({"qid": pos_q, "text": [qtext[q] for q in pos_q]}).to_parquet(out / f"{ds}_queries.parquet", index=False)
        qr.to_parquet(out / f"{ds}_qrels.parquet", index=False)
        meta["counts"][ds] = {"docs": len(docs), "test_queries_kept": len(pos_q),
                              "qrels_rows_raw": n_qr, "qrels_rows_kept": len(qr),
                              "qrels_rows_with_missing_doc_kept": n_missing_doc}
        log.info(f"{ds} ({name}): {meta['counts'][ds]}")

    for rel in cfg["msmarco_copy_from_rag01"]:
        src, dst = p(cfg, "rag01_raw", rel), p(cfg, "raw", rel)
        h = sha256_file(src)
        if not (dst.exists() and sha256_file(dst) == h):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            assert sha256_file(dst) == h
        meta["inputs_sha256"][rel] = h
    mq = pd.read_parquet(p(cfg, "raw", "beir/msmarco/queries/queries-00000-of-00001.parquet"))
    mtext = dict(zip(mq["_id"].astype(str), mq["text"].astype(str)))
    tr = pd.read_csv(p(cfg, "raw", "beir/msmarco-qrels/train.tsv"), sep="\t", dtype=str)
    train_ids = sorted(set(tr.iloc[:, 0]) & set(mtext))
    rng = np.random.default_rng(stable_seed(cfg["seed"], "P2_bank"))
    sel = np.sort(rng.choice(len(train_ids), size=cfg["bank_size"], replace=False))
    bank = pd.DataFrame({"qid": [train_ids[i] for i in sel], "text": [mtext[train_ids[i]] for i in sel]})
    bank.to_parquet(out / "bank.parquet", index=False)
    meta["counts"]["msmarco_train_queries_with_text"] = len(train_ids)
    meta["counts"]["bank"] = len(bank)
    save_json(meta, out / "meta.json")
    log.info(f"bank {len(bank)} from {len(train_ids)} MS MARCO train queries")


if __name__ == "__main__":
    main()
