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
from oodrag.text import answer_keys, answer_type, has_answer_key, is_yesno, normalize_question, token_key


def copy_inputs(cfg, log) -> dict:
    hashes = {}
    for rel in cfg["copy_from_rag01"]:
        src, dst = p(cfg, "rag01_raw", rel), p(cfg, "raw", rel)
        h_src = sha256_file(src)
        if not (dst.exists() and sha256_file(dst) == h_src):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if sha256_file(dst) != h_src:
                raise RuntimeError(f"copy hash mismatch: {rel}")
            log.info(f"copied {rel}")
        hashes[rel] = h_src
    return hashes


def download_external(cfg, log) -> dict:
    from huggingface_hub import hf_hub_download

    hashes = {}
    for name, e in cfg["external"].items():
        for repo, rev, fn in [(e["repo"], e["revision"], "queries/queries-00000-of-00001.parquet"),
                              (e["qrels_repo"], e["qrels_revision"], "test.tsv")]:
            local = p(cfg, "raw", "beir", repo.split("/")[1])
            f = Path(hf_hub_download(repo, fn, repo_type="dataset", revision=rev, local_dir=str(local)))
            rel = str(f.relative_to(p(cfg, "raw")))
            hashes[rel] = sha256_file(f)
            log.info(f"external {repo}@{rev[:7]} {fn} -> {rel}")
    return hashes


def read_qrels(path: Path) -> pd.DataFrame:
    q = pd.read_csv(path, sep="\t", dtype=str)
    q.columns = ["qid", "doc_id", "score"]
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = get_logger("P0.01", p(cfg, "results", "logs", "01_prepare.log"))
    seed = cfg["seed"]
    out = p(cfg, "processed")
    out.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(cfg, "01_prepare")
    meta["inputs_sha256"] = {**copy_inputs(cfg, log), **download_external(cfg, log)}

    t = pq.read_table(p(cfg, "raw", "beir/nq/corpus/corpus-00000-of-00001.parquet"), columns=["_id", "title", "text"])
    doc_id = np.array(t.column("_id").to_pylist(), dtype=object)
    titles = t.column("title").to_pylist()
    texts = t.column("text").to_pylist()
    title_id, uniq_titles = pd.factorize(pd.Series(titles), sort=False)
    title_id = title_id.astype(np.int32)
    n_pass = np.bincount(title_id, minlength=len(uniq_titles))
    np.savez(out / "corpus_index.npz", doc_id=doc_id.astype(str), title_id=title_id)
    pd.DataFrame({"title_id": np.arange(len(uniq_titles), dtype=np.int32), "title": list(uniq_titles),
                  "n_passages": n_pass}).to_parquet(out / "titles.parquet", index=False)
    id2idx = {d: i for i, d in enumerate(doc_id)}
    log.info(f"corpus {len(doc_id):,} passages, {len(uniq_titles):,} titles")

    qt = pd.read_parquet(p(cfg, "raw", "beir/nq/queries/queries-00000-of-00001.parquet"))
    qtext = dict(zip(qt["_id"].astype(str), qt["text"].astype(str)))
    qrels = read_qrels(p(cfg, "raw", "beir/nq-qrels/test.tsv"))
    qids = sorted(qrels["qid"].unique())
    gold = qrels.groupby("qid")["doc_id"].apply(list).to_dict()

    def answer_map(fn):
        df = pd.read_parquet(p(cfg, "raw", "answers/nq_open/nq_open", fn))
        m: dict[str, list[str]] = {}
        for q, a in zip(df["question"], df["answer"]):
            lst = m.setdefault(normalize_question(q), [])
            for x in list(a):
                if x not in lst:
                    lst.append(x)
        return m, df

    ans_val, _ = answer_map("validation-00000-of-00001.parquet")
    ans_train, train_df = answer_map("train-00000-of-00001.parquet")

    rows = []
    for qid in qids:
        text = qtext[qid]
        nq = normalize_question(text)
        answers = ans_val.get(nq, [])
        g_idx = [id2idx[d] for d in gold[qid]]
        g_tids = sorted({int(title_id[i]) for i in g_idx})
        keys = answer_keys(answers)
        g_has = [bool(keys) and has_answer_key(token_key(f"{titles[i]} {texts[i]}"), keys) for i in g_idx]
        title_ans = bool(keys) and any(has_answer_key(token_key(str(uniq_titles[t_])), keys) for t_ in g_tids)
        yesno = is_yesno(answers)
        rows.append({
            "qid": qid, "text": text, "answers": answers, "matched_val": bool(answers),
            "matched_train": nq in ans_train, "yesno": yesno,
            "answer_type": answer_type(answers) if answers else None,
            "gold_doc_idx": g_idx, "gold_title_ids": g_tids, "gold_has_answer": g_has,
            "answer_in_gold_title": title_ans,
            "gold_title_npass": int(sum(n_pass[t_] for t_ in g_tids)),
            "candidate": bool(answers) and not yesno and any(g_has),
        })
    Q = pd.DataFrame(rows)
    Q.to_parquet(out / "queries_nq.parquet", index=False)

    test_norm = {normalize_question(qtext[q]) for q in qids}
    seen, bank_txt = set(), []
    for q in train_df["question"].astype(str):
        n = normalize_question(q)
        if n in seen or n in test_norm:
            continue
        seen.add(n)
        bank_txt.append(q)
    rng_b = np.random.default_rng(stable_seed(seed, "bank"))
    sel = np.sort(rng_b.choice(len(bank_txt), size=cfg["bank_size"], replace=False))
    pd.DataFrame({"bank_id": sel.astype(np.int64), "text": [bank_txt[i] for i in sel]}).to_parquet(
        out / "bank.parquet", index=False)

    ext = []
    for name, e in cfg["external"].items():
        d = e["repo"].split("/")[1]
        qq = pd.read_parquet(p(cfg, "raw", "beir", d, "queries/queries-00000-of-00001.parquet"))
        test_ids = sorted(read_qrels(p(cfg, "raw", "beir", f"{d}-qrels", "test.tsv"))["qid"].unique())
        m = dict(zip(qq["_id"].astype(str), qq["text"].astype(str)))
        missing = [q for q in test_ids if q not in m]
        if missing:
            raise RuntimeError(f"{name}: {len(missing)} test qids without text")
        ext += [{"qid": f"{name}:{q}", "text": m[q], "source": name} for q in test_ids]
    E = pd.DataFrame(ext)
    E.to_parquet(out / "external.parquet", index=False)

    meta["counts"] = {
        "nq_test_queries": len(Q), "matched_nqopen_val": int(Q["matched_val"].sum()),
        "matched_nqopen_train_only": int((~Q["matched_val"] & Q["matched_train"]).sum()),
        "yesno": int(Q["yesno"].sum()),
        "gold_has_answer_any": int(Q["gold_has_answer"].apply(any).sum()),
        "candidates": int(Q["candidate"].sum()),
        "candidates_answer_in_gold_title": int((Q["candidate"] & Q["answer_in_gold_title"]).sum()),
        "candidate_answer_type": Q.loc[Q["candidate"], "answer_type"].value_counts().to_dict(),
        "bank_pool_after_dedup_exclusion": len(bank_txt), "bank": int(cfg["bank_size"]),
        "external": E["source"].value_counts().to_dict(),
    }
    save_json(meta, out / "meta.json")
    log.info(f"counts: {meta['counts']}")


if __name__ == "__main__":
    main()
