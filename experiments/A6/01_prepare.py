from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import fingerprint, get_logger, load_config, p, run_metadata, save_json, stable_seed
from oodrag.encoders import doc_text


def read_chunks(pattern):
    rows = []
    for f in sorted(glob.glob(str(pattern))):
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                rows.append({"doc_id": d["id"], "text": d["contents"], "title": d.get("title", ""), "pmid": d.get("PMID")})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = p(cfg, "processed")
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger("A6.01", p(cfg, "results", "logs", "01_prepare.log"))
    raw = p(cfg, "li_raw")
    n = cfg["corpora"]["size"]
    tb = read_chunks(raw / "MedRAG__textbooks" / "chunk" / "*.jsonl")
    pm = read_chunks(raw / "MedRAG__pubmed" / "chunk" / "*.jsonl")
    wk = read_chunks(raw / "MedRAG__wikipedia" / "chunk" / "*.jsonl")
    assert len(tb) == 125847 and len(pm) == 90692 and len(wk) == 111734, (len(tb), len(pm), len(wk))
    wk = wk.iloc[np.sort(np.random.default_rng(stable_seed(cfg["seed"], "A6_wiki")).permutation(len(wk))[:n])].reset_index(drop=True)
    mpath = Path(cfg["paths"]["msmarco_corpus"])
    nms = pq.ParquetFile(mpath).metadata.num_rows
    sel = np.sort(np.random.default_rng(stable_seed(cfg["seed"], "A6_msmarco")).permutation(nms)[:n])
    ms = pq.read_table(mpath).to_pandas()
    ms = ms.iloc[sel].reset_index(drop=True)
    tcol = "text" if "text" in ms else "contents"
    ms = pd.DataFrame({"doc_id": ms["_id"].astype(str), "text": [doc_text(t, x) for t, x in zip(ms.get("title", [""] * len(ms)), ms[tcol])], "pmid": None})
    for name, df in (("textbooks", tb), ("pubmed", pm), ("wikipedia", wk), ("msmarco", ms)):
        df[["doc_id", "text"]].to_parquet(out / f"corpus_{name}.parquet", index=False)
        log.info(f"corpus {name}: {len(df)}")

    b = json.load(open(raw / "MIRAGE_benchmark.json"))
    pm_ids = {int(x) for x in pm.pmid.dropna()}
    wk_titles = {t.strip().lower() for t in wk.title}
    q = []
    for ds in ("medqa", "mmlu", "medmcqa", "pubmedqa", "bioasq"):
        for qid, v in b[ds].items():
            opts = " ".join(f"{k}. {t}" for k, t in v["options"].items())
            src = {int(x) for x in v.get("PMID", [])}
            q.append({"dataset": ds, "qid": f"{ds}:{qid}", "question": v["question"], "question_options": f"{v['question']} {opts}",
                      "source_in_comparison": bool(src & pm_ids)})
    tq = pd.read_parquet(raw / "truthfulqa__truthful_qa" / "generation" / "validation-00000-of-00001.parquet")
    tq = tq[tq.category != "Health"]
    def wiki_title(url):
        if "wikipedia.org/wiki/" not in str(url):
            return None
        from urllib.parse import unquote
        return unquote(str(url).split("/wiki/")[1].split("#")[0]).replace("_", " ").strip().lower()
    q += [{"dataset": "truthfulqa", "qid": f"truthfulqa:{i}", "question": t, "question_options": t,
           "source_in_comparison": wiki_title(src) in wk_titles if wiki_title(src) else False}
          for i, t, src in zip(tq.index, tq.question, tq.source)]
    wq = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(raw / "microsoft__wiki_qa" / "data" / "*.parquet")))])
    wq = wq.drop_duplicates("question_id").sort_values("question_id")
    q += [{"dataset": "wikiqa", "qid": f"wikiqa:{i}", "question": t, "question_options": t,
           "source_in_comparison": str(dt).strip().lower() in wk_titles}
          for i, t, dt in zip(wq.question_id, wq.question, wq.document_title)]
    cq = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(raw / "tau__commonsense_qa" / "data" / "*.parquet")))])
    cq = cq.drop_duplicates("id").sort_values("id")
    q += [{"dataset": "commonsenseqa", "qid": f"commonsenseqa:{i}", "question": t,
           "question_options": t + " " + " ".join(f"{l}. {x}" for l, x in zip(c["label"], c["text"])), "source_in_comparison": False}
          for i, t, c in zip(cq.id, cq.question, cq.choices)]
    Q = pd.DataFrame(q)
    Q["split"] = "all"
    med = np.flatnonzero(Q.dataset == "medqa")
    perm = np.random.default_rng(stable_seed(cfg["seed"], "A6_medqa_split")).permutation(len(med))
    Q.loc[med[perm[: len(med) // 2]], "split"] = "cal"
    Q.loc[med[perm[len(med) // 2:]], "split"] = "test"
    Q.to_parquet(out / "queries.parquet", index=False)
    counts = Q.groupby("dataset").size().to_dict()
    srcov = Q.groupby("dataset").source_in_comparison.sum().astype(int).to_dict()
    log.info(f"queries {counts}; source document in a comparison corpus (excluded from the relative analysis): {srcov}")
    meta = run_metadata(cfg, "A6.01_prepare")
    meta["inputs"] = fingerprint([raw / "MIRAGE_benchmark.json", mpath] + sorted(Path(x) for x in glob.glob(str(raw / "MedRAG__*" / "chunk" / "*.jsonl"))))
    meta["counts"] = counts
    meta["source_in_comparison"] = srcov
    meta["outputs"] = fingerprint([out / "queries.parquet"] + [out / f"corpus_{n_}.parquet" for n_ in ("textbooks", "pubmed", "wikipedia", "msmarco")])
    save_json(meta, out / "meta_01.json")


if __name__ == "__main__":
    main()
