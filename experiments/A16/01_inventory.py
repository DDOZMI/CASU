import ast, gzip, json
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "amazon_qa"
PROC = ROOT / "data" / "processed" / "A16"
RES = ROOT / "experiments" / "A16" / "results"
PROC.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)
man = json.loads((RAW / "manifest.json").read_text())
LEVELS = [4500, 3000, 2000, 1000]


def rows(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield ast.literal_eval(line)


prod = []
for cat in man["categories"]:
    qa = defaultdict(lambda: {"n_q": 0, "n_yesno": 0, "qlen": 0})
    for r in rows(RAW / "qa" / f"qa_{cat}.json.gz"):
        d = qa[r["asin"]]
        d["n_q"] += 1
        d["n_yesno"] += r.get("questionType") == "yes/no"
        d["qlen"] += len(r["question"].split())
    found = set()
    for m in rows(RAW / "meta" / f"meta_{cat}.json.gz"):
        a = m.get("asin")
        if a not in qa:
            continue
        found.add(a)
        paths = [tuple(p) for p in m.get("categories", []) if p]
        rel = m.get("related", {}) or {}
        prod.append({"file": cat, "asin": a, **qa[a], "n_paths": len(paths), "paths": json.dumps(paths),
                     "has_title": bool(m.get("title")), "also_bought": len(rel.get("also_bought", [])),
                     "also_viewed": len(rel.get("also_viewed", [])), "bought_together": len(rel.get("bought_together", []))})
    for a in set(qa) - found:
        prod.append({"file": cat, "asin": a, **qa[a], "n_paths": -1, "paths": "[]", "has_title": False,
                     "also_bought": 0, "also_viewed": 0, "bought_together": 0})
    print(cat, len(qa), "products,", sum(v["n_q"] for v in qa.values()), "questions,", len(found), "with metadata", flush=True)

P = pd.DataFrame(prod)
P.to_parquet(PROC / "inventory_products.parquet")
M = P[P.n_paths >= 0]

node = defaultdict(lambda: Counter())
for r in M.itertuples():
    paths = [tuple(p) for p in json.loads(r.paths)]
    single = len(paths) == 1
    seen = set()
    for p in paths:
        for d in range(2, len(p) + 1):
            seen.add(("d%d" % d, p[:d]))
        seen.add(("leaf", p))
    for key in seen:
        node[key]["prod_all"] += 1
        node[key]["q_all"] += r.n_q
        if single:
            node[key]["prod_single"] += 1
            node[key]["q_single"] += r.n_q
N = pd.DataFrame([{"level": k[0], "path": " > ".join(k[1]), "depth": len(k[1]), **v} for k, v in node.items()]).fillna(0)
N.to_csv(PROC / "inventory_nodes.csv", index=False)

out = {"files": {}, "overall": {}, "nodes_reaching": {}}
for cat, g in P.groupby("file"):
    out["files"][cat] = {"products": int(len(g)), "questions": int(g.n_q.sum()), "meta_coverage": float((g.n_paths >= 0).mean()),
                         "multi_path_rate": float((g.n_paths > 1).sum() / max((g.n_paths >= 0).sum(), 1))}
out["overall"] = {"products": int(len(P)), "questions": int(P.n_q.sum()), "meta_coverage_products": float((P.n_paths >= 0).mean()),
                  "meta_coverage_questions": float(M.n_q.sum() / P.n_q.sum()), "multi_path_rate": float((M.n_paths > 1).mean()),
                  "q_in_single_path_products": float(M[M.n_paths == 1].n_q.sum() / M.n_q.sum()),
                  "yesno_share": float(P.n_yesno.sum() / P.n_q.sum()), "mean_q_words": float(P.qlen.sum() / P.n_q.sum()),
                  "q_per_product_median": float(P.n_q.median()),
                  "related_present": {k: float((M[k] > 0).mean()) for k in ["also_bought", "also_viewed", "bought_together"]}}
for lev in ["d2", "d3", "d4", "d5", "leaf"]:
    g = N[N.level == lev]
    out["nodes_reaching"][lev] = {f"q_single>={L}": int((g.q_single >= L).sum()) for L in LEVELS} | \
                                 {f"q_all>={L}": int((g.q_all >= L).sum()) for L in LEVELS} | {"n_nodes": int(len(g))}
(RES / "inventory.json").write_text(json.dumps(out, indent=1))

md = ["# A16-01 inventory (Amazon QA x 2014 metadata; descriptive)", "",
      "Per-site need in the SE design (A0): 1,000 test + 500 cal + 3,000 bank = 4,500 questions; other levels are reference only.",
      "q_single counts only products with exactly one category path; q_all counts a product under every node on any path.", "",
      "## Overall", "", "```", json.dumps(out["overall"], indent=1), "```", "",
      "## Per file", "", "| file | products | questions | metadata coverage | multi-path rate |", "|---|---|---|---|---|"]
md += [f"| {c} | {v['products']} | {v['questions']} | {v['meta_coverage']:.3f} | {v['multi_path_rate']:.3f} |" for c, v in out["files"].items()]
md += ["", "## Nodes reaching question levels", "", "| level | nodes | " + " | ".join(f"single≥{L}" for L in LEVELS) + " | " + " | ".join(f"all≥{L}" for L in LEVELS) + " |",
       "|---|---|" + "---|" * (2 * len(LEVELS))]
for lev, v in out["nodes_reaching"].items():
    md.append(f"| {lev} | {v['n_nodes']} | " + " | ".join(str(v[f'q_single>={L}']) for L in LEVELS) + " | " + " | ".join(str(v[f'q_all>={L}']) for L in LEVELS) + " |")
for lev in ["d3", "d4"]:
    g = N[(N.level == lev) & (N.q_single >= 2000)].sort_values("q_single", ascending=False)
    md += ["", f"## {lev} nodes with q_single ≥ 2,000 ({len(g)})", "", "| path | products single | questions single | questions all |", "|---|---|---|---|"]
    md += [f"| {r.path} | {int(r.prod_single)} | {int(r.q_single)} | {int(r.q_all)} |" for r in g.itertuples()]
(RES / "inventory.md").write_text("\n".join(md) + "\n")
print("done", json.dumps(out["nodes_reaching"]))
