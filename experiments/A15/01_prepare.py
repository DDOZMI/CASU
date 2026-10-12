from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import fingerprint, get_logger, load_config, p, run_metadata, save_json, stable_seed
from oodrag.encoders import doc_text

YAHOO = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports",
         "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government"]


def norm(s):
    return " ".join(str(s).lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    cfg = load_config(ap.parse_args().config)
    log = get_logger("A15.01", p(cfg, "results", "logs", "01_prepare.log"))
    pc = cfg["prepare"]
    out = p(cfg, "processed")
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    raw = p(cfg, "raw", "dbpedia_classes")
    D = pd.concat([pd.read_csv(raw / f"DBPEDIA_{s}.csv") for s in ("train", "val", "test")], ignore_index=True)
    D["key"] = D.text.map(norm)
    D = D.drop_duplicates("key").reset_index(drop=True)
    rows = []
    for l2, g in D.groupby("l2", sort=True):
        if len(g) < pc["group_size"]:
            continue
        g = g.assign(r=np.random.default_rng(stable_seed(seed, "A15_wiki", l2)).random(len(g))).sort_values(["l3", "r"])
        for j in range(len(g) // pc["group_size"]):
            c = g.iloc[j * pc["group_size"]:(j + 1) * pc["group_size"]]
            rows.append(pd.DataFrame({"group": f"wiki:{l2}#{j}", "l1": c.l1.to_numpy(), "l2": l2, "l3": c.l3.to_numpy(), "text": c.text.to_numpy()}))
    W = pd.concat(rows, ignore_index=True)
    W.to_parquet(out / "wiki_groups.parquet", index=False)
    log.info(f"wiki: {W.group.nunique()} groups, {len(W)} docs, l2 classes {W.l2.nunique()}")
    t = pq.read_table(p(cfg, "msmarco_corpus"), columns=["_id", "title", "text"])
    p9 = set(pd.read_parquet(p(cfg, "p9_processed", "pool.parquet")).corpus_idx.tolist())
    rng = np.random.default_rng(stable_seed(seed, "A15_ms"))
    cand = rng.permutation(t.num_rows)
    cand = cand[~np.isin(cand, list(p9))][: pc["ms_pool"]]
    sub = t.take(cand)
    M = pd.DataFrame({"pool_rank": np.arange(len(cand)), "corpus_idx": cand, "doc_id": [str(x) for x in sub.column("_id").to_pylist()],
                      "text": [doc_text(a, b) for a, b in zip(sub.column("title").to_pylist(), sub.column("text").to_pylist())]})
    M.to_parquet(out / "ms_pool.parquet", index=False)
    log.info(f"ms pool: {len(M)} docs")
    Q = []
    Y = pd.read_parquet(p(cfg, "raw", "yahoo_answers_topics", "test-00000-of-00001.parquet"))
    Y = Y[Y.question_title.fillna("").str.strip().str.len() > 0].assign(key=lambda x: x.question_title.map(norm)).drop_duplicates("key")
    for k, name in enumerate(YAHOO):
        g = Y[Y.topic == k]
        g = g.iloc[np.sort(np.random.default_rng(stable_seed(seed, "A15_yahoo", k)).choice(len(g), pc["n_ood"], replace=False))]
        Q.append(pd.DataFrame({"source": f"yahoo:{name}", "qid": g.id.astype(str).to_numpy(), "text": g.question_title.str.strip().to_numpy()}))
    NQ = pd.read_parquet(p(cfg, "p0_processed", "queries_nq.parquet"))
    NQ = NQ.iloc[np.sort(np.random.default_rng(stable_seed(seed, "A15_nq")).choice(len(NQ), pc["n_ood"], replace=False))]
    Q.append(pd.DataFrame({"source": "nq", "qid": NQ.qid.astype(str).to_numpy(), "text": NQ.text.to_numpy()}))
    MQ = pd.read_parquet(p(cfg, "p2_processed", "bank.parquet"))
    MQ = MQ.iloc[np.sort(np.random.default_rng(stable_seed(seed, "A15_msq")).choice(len(MQ), pc["n_ood"], replace=False))]
    Q.append(pd.DataFrame({"source": "msq", "qid": MQ.qid.astype(str).to_numpy(), "text": MQ.text.to_numpy()}))
    Q = pd.concat(Q, ignore_index=True)
    Q["pos"] = Q.groupby("source").cumcount()
    Q.to_parquet(out / "ood_queries.parquet", index=False)
    log.info(f"ood queries: {Q.source.value_counts().to_dict()}")
    meta = run_metadata(cfg, "A15.01_prepare")
    meta["inputs"] = fingerprint([raw / f"DBPEDIA_{s}.csv" for s in ("train", "val", "test")] + [p(cfg, "msmarco_corpus"),
                                 p(cfg, "raw", "yahoo_answers_topics", "test-00000-of-00001.parquet"), p(cfg, "p0_processed", "queries_nq.parquet"),
                                 p(cfg, "p2_processed", "bank.parquet")])
    meta.update({"wiki_groups": int(W.group.nunique()), "wiki_docs": len(W), "ms_pool": len(M), "ood": Q.source.value_counts().to_dict()})
    save_json(meta, out / "meta_01.json")


if __name__ == "__main__":
    main()
