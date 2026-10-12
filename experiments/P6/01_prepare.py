from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import fingerprint, get_logger, load_config, p, run_metadata, save_json, sha256_file, stable_seed
from oodrag.encoders import doc_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("P6.01", p(cfg, "results", "logs", "01_prepare.log"))
    out = p(cfg, "processed")
    out.mkdir(parents=True, exist_ok=True)
    cfg4 = load_config(p(cfg, "p4_config"))
    revs = cfg4["cqadupstack"]["revisions"]
    assert cfg4["beir_sources"] == cfg["beir_sources"] and cfg4["nq_source"]
    from huggingface_hub import hf_hub_download

    qpath = p(cfg, "p4_processed", "queries.parquet")
    Q = pd.read_parquet(qpath)
    sources = list(dict.fromkeys(Q["source"]))
    assert set(sources) == set(revs) | set(cfg["beir_sources"]) | {"nq"}
    n, ns = cfg["bank"]["n"], cfg["bank"]["n_sensitivity"]
    files, banks, pos, info = [qpath], [], [], {}

    def take(s, N, get_rows):
        perm = np.random.default_rng(stable_seed(cfg["seed"], "P6_bank", s)).permutation(N)[:n]
        ids, texts = get_rows(perm)
        banks.append(pd.DataFrame({"source": s, "rank": np.arange(n), "corpus_idx": perm.astype(np.int64),
                                   "doc_id": [str(x) for x in ids], "text": texts, "in_sens": np.arange(n) < ns}))
        return set(str(x) for x in ids)

    for s in sources:
        qids = set(Q.loc[Q.source == s, "qid"])
        if s in revs:
            dst = p(cfg, "raw", "cqadupstack", s, cfg["cqadupstack"]["file"])
            got = hf_hub_download(cfg["cqadupstack"]["repo_pattern"].format(forum=s), cfg["cqadupstack"]["file"],
                                  repo_type="dataset", revision=revs[s])
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(got, dst)
            files.append(dst)
            ids, tx = [], []
            with open(dst) as f:
                for line in f:
                    o = json.loads(line)
                    ids.append(str(o["_id"]))
                    tx.append(doc_text(o.get("title", ""), o.get("text", "")))
            ids_a = np.array(ids, dtype=object)
            bank_ids = take(s, len(ids), lambda pm: (ids_a[pm], [tx[i] for i in pm]))
            all_ids = set(ids)
            qr = pd.read_csv(p(cfg, "raw", "cqadupstack", s, "qrels", "test.tsv"), sep="\t", dtype=str)
            qr = qr[qr["score"].astype(int) > 0]
            pos.append(pd.DataFrame({"source": s, "qid": qr["query-id"].astype(str), "doc_id": qr["corpus-id"].astype(str)}))
            N = len(ids)
            info[s] = {"corpus_sha256": sha256_file(dst), "revision": revs[s]}
        elif s in cfg["beir_sources"]:
            f = p(cfg, "p2_processed", f"{s}_docs.parquet")
            files.append(f)
            D = pd.read_parquet(f)
            ids_a, tx_a = D["doc_id"].astype(str).to_numpy(dtype=object), D["text"].astype(str).to_numpy(dtype=object)
            bank_ids = take(s, len(D), lambda pm: (ids_a[pm], list(tx_a[pm])))
            all_ids = set(ids_a)
            fq = p(cfg, "p2_processed", f"{s}_qrels.parquet")
            files.append(fq)
            qr = pd.read_parquet(fq)
            qr = qr[qr["score"] > 0]
            pos.append(pd.DataFrame({"source": s, "qid": qr["qid"].astype(str), "doc_id": qr["doc_id"].astype(str)}))
            N = len(D)
            info[s] = {}
        else:
            f = p(cfg, "raw", cfg["nq_corpus"])
            files.append(f)
            t = pq.read_table(f, columns=["_id", "title", "text"])
            N = t.num_rows

            def rows(pm):
                sub = t.take(pm)
                return (sub.column("_id").to_pylist(),
                        [doc_text(a, b) for a, b in zip(sub.column("title").to_pylist(), sub.column("text").to_pylist())])
            bank_ids = take(s, N, rows)
            all_ids = set(t.column("_id").to_pylist())
            del t
            fq = p(cfg, "raw", cfg["nq_qrels"])
            files.append(fq)
            qr = pd.read_csv(fq, sep="\t", dtype=str)
            qr = qr[qr["score"].astype(int) > 0]
            pos.append(pd.DataFrame({"source": s, "qid": qr["query-id"].astype(str), "doc_id": qr["corpus-id"].astype(str)}))
            info[s] = {}
        P_s = pos[-1]
        P_s = P_s[P_s.qid.isin(qids)]
        pos[-1] = P_s
        ev_q = set(Q.loc[(Q.source == s) & (Q.split == "eval"), "qid"])
        hit = P_s[P_s.doc_id.isin(bank_ids)].groupby("qid").size()
        btx = banks[-1]["text"].str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
        info[s].update({"corpus_size": int(N), "bank_fraction_of_corpus": n / N,
                        "p4_qids_also_corpus_ids": int(len(qids & all_ids)), "p4_qids_also_bank_ids": int(len(qids & bank_ids)),
                        "p4_queries_without_positive": int(len(qids - set(P_s.qid))),
                        "eval_queries_with_positive_in_bank": float(np.mean([q in hit.index for q in ev_q])),
                        "bank_docs_with_exact_duplicate_text": int(btx.duplicated(keep=False).sum())})
        assert info[s]["p4_queries_without_positive"] == 0, s
        log.info(f"{s}: corpus {N:,}, {info[s]}")

    B = pd.concat(banks, ignore_index=True)
    B.to_parquet(out / "banks.parquet", index=False)
    PS = pd.concat(pos, ignore_index=True).drop_duplicates()
    PS.to_parquet(out / "positives.parquet", index=False)
    meta = run_metadata(cfg, "01_prepare")
    meta.update({"per_source": info, "bank_n": n, "bank_n_sensitivity": ns, "upstream_fingerprint": fingerprint(files)})
    save_json(meta, out / "meta_01.json")
    log.info(f"banks {len(B):,} rows over {B.source.nunique()} sources; positives {len(PS):,}")


if __name__ == "__main__":
    main()
