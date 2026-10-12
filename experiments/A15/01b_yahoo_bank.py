from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import fingerprint, get_logger, load_config, p, run_metadata, save_json, stable_seed

spec = importlib.util.spec_from_file_location("a15_01", Path(__file__).with_name("01_prepare.py"))
P1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    cfg = load_config(ap.parse_args().config)
    log = get_logger("A15.01b", p(cfg, "results", "logs", "01b_yahoo_bank.log"))
    yc = cfg["yahoo_bank"]
    raw = p(cfg, "raw", "yahoo_answers_topics")
    files = [raw / f"train-0000{i}-of-00002.parquet" for i in (0, 1)]
    Y = pd.concat([pd.read_parquet(f, columns=["topic", "question_title", "question_content"]) for f in files], ignore_index=True)
    Y = Y[Y.question_title.fillna("").str.strip().str.len() > 0].copy()
    Y["key"] = Y.question_title.map(P1.norm)
    ood_keys = set(pd.read_parquet(p(cfg, "processed", "ood_queries.parquet")).text.map(P1.norm))
    Y = Y.drop_duplicates("key")
    n_ood_match = int(Y.key.isin(ood_keys).sum())
    Y = Y[~Y.key.isin(ood_keys)]
    need = yc["chunks_per_topic"] * yc["group_size"]
    rows = []
    for k, name in enumerate(P1.YAHOO):
        g = Y[Y.topic == k]
        g = g.iloc[np.random.default_rng(stable_seed(cfg["seed"], "A15_yahoo_bank", k)).permutation(len(g))[:need]]
        assert len(g) == need, (name, len(g))
        txt = (g.question_title.str.strip() + " " + g.question_content.fillna("").str.strip()).str.strip().to_numpy()
        for j in range(yc["chunks_per_topic"]):
            rows.append(pd.DataFrame({"group": f"yahoo:{name}#{j}", "topic": f"yahoo:{name}", "text": txt[j * yc["group_size"]:(j + 1) * yc["group_size"]]}))
    G = pd.concat(rows, ignore_index=True)
    G.to_parquet(p(cfg, "processed", "yahoo_groups.parquet"), index=False)
    log.info(f"yahoo bank: {G.group.nunique()} groups, {len(G)} docs; train titles equal to an out-of-scope test title removed: {n_ood_match}")
    meta = run_metadata(cfg, "A15.01b_yahoo_bank")
    meta["inputs"] = fingerprint(files + [p(cfg, "processed", "ood_queries.parquet")])
    meta.update({"groups": int(G.group.nunique()), "docs": len(G), "removed_title_matches": n_ood_match})
    save_json(meta, p(cfg, "processed", "meta_01b.json"))


if __name__ == "__main__":
    main()
