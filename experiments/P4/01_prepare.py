from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import fingerprint, get_logger, load_config, p, run_metadata, save_json, stable_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("P4.01", p(cfg, "results", "logs", "01_prepare.log"))
    out = p(cfg, "processed")
    out.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    cq = cfg["cqadupstack"]
    rows, files = [], []
    for forum, rev in cq["revisions"].items():
        dst = p(cfg, "raw", "cqadupstack", forum)
        for fn in cq["files"]:
            local = dst / fn
            got = hf_hub_download(cq["repo_pattern"].format(forum=forum), fn, repo_type="dataset", revision=rev)
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(got, local)
            files.append(local)
        qr = pd.read_csv(dst / "qrels" / "test.tsv", sep="\t", dtype=str)
        pos = set(qr.loc[qr["score"].astype(int) > 0, "query-id"])
        with open(dst / "queries.jsonl") as f:
            for line in f:
                o = json.loads(line)
                if str(o["_id"]) in pos:
                    rows.append({"source": forum, "group": "cqa", "qid": str(o["_id"]), "text": str(o["text"])})
    for ds in cfg["beir_sources"]:
        f = p(cfg, "p2_processed", f"{ds}_queries.parquet")
        files.append(f)
        Q = pd.read_parquet(f)
        rows += [{"source": ds, "group": "beir", "qid": str(q), "text": str(t)} for q, t in zip(Q["qid"], Q["text"])]
    if cfg["nq_source"]:
        f = p(cfg, "p0_processed", "queries_nq.parquet")
        files.append(f)
        Q = pd.read_parquet(f, columns=["qid", "text"])
        rows += [{"source": "nq", "group": "beir", "qid": str(q), "text": str(t)} for q, t in zip(Q["qid"], Q["text"])]
    D = pd.DataFrame(rows)
    n0 = D.groupby("source").size().to_dict()
    D["key"] = D["text"].str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    D = D.drop_duplicates(["source", "key"])
    n_dedup = D.groupby("source").size().to_dict()
    nsrc = D.groupby("key")["source"].nunique()
    amb = set(nsrc[nsrc > 1].index)
    D = D[~D["key"].isin(amb)].drop(columns="key")
    removed = {s_: {"within_source_duplicates": int(n0[s_] - n_dedup[s_]),
                    "cross_source_ambiguous": int(n_dedup[s_] - int((D["source"] == s_).sum()))} for s_ in n0}
    parts = []
    for s, g in D.groupby("source", sort=False):
        g = g.sort_values("qid").reset_index(drop=True)
        perm = np.random.default_rng(stable_seed(cfg["seed"], "P4_split", s)).permutation(len(g))
        nfit = int(round(cfg["fit_fraction"] * len(g)))
        sp = np.empty(len(g), dtype=object)
        sp[perm[:nfit]], sp[perm[nfit:]] = "fit", "eval"
        parts.append(g.assign(split=sp))
    D = pd.concat(parts, ignore_index=True)
    D.to_parquet(out / "queries.parquet", index=False)
    counts = D.groupby(["source", "split"]).size().unstack().to_dict(orient="index")
    meta = run_metadata(cfg, "01_prepare")
    meta.update({"n_raw_per_source": n0, "n_ambiguous_texts_dropped": len(amb), "rows_removed_per_source": removed, "counts": counts,
                 "upstream_fingerprint": fingerprint(files)})
    save_json(meta, out / "meta_01.json")
    log.info(f"sources {D['source'].nunique()}, queries {len(D)}, ambiguous texts dropped {len(amb)}; counts {counts}")


if __name__ == "__main__":
    main()
