import hashlib, json, random, urllib.request
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "li_bridge"
OUT.mkdir(parents=True, exist_ok=True)
SEED, N_PUBMED, N_WIKI = 20261006, 6, 3
REV = {"MedRAG/textbooks": "9c72838920a1323ffa867467d3f7aa7b36b0f994", "MedRAG/pubmed": "33da3593d5756bc04c8909f170003c0b14197957",
       "MedRAG/wikipedia": "d76b9ad82135e352235d17e75921c49b68fd07b2", "truthfulqa/truthful_qa": "741b8276f2d1982aa3d5b832d3ee81ed3b896490",
       "microsoft/wiki_qa": "3f104672b5de699878fe7907afc486f0de325eb5", "tau/commonsense_qa": "94630fe30dad47192a8546eb75f094926d47e155"}
MIRAGE = ("gzxiong/MIRAGE", "392943af99cd94cafd50a0de2e7fca24bbf65494")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


api = HfApi()
man = {"seed": SEED, "revisions": REV, "mirage": {"repo": MIRAGE[0], "commit": MIRAGE[1]}, "files": {}}
plan = {"MedRAG/textbooks": sorted(f for f in api.list_repo_files("MedRAG/textbooks", repo_type="dataset", revision=REV["MedRAG/textbooks"]) if f.startswith("chunk/"))}
rng = random.Random(SEED)
for repo, n in (("MedRAG/pubmed", N_PUBMED), ("MedRAG/wikipedia", N_WIKI)):
    fs = sorted(f for f in api.list_repo_files(repo, repo_type="dataset", revision=REV[repo]) if f.startswith("chunk/") and f.endswith(".jsonl"))
    plan[repo] = sorted(rng.sample(fs, n))
man["selected_shards"] = {k: v for k, v in plan.items() if k != "MedRAG/textbooks"}
for repo, files in plan.items():
    for f in files:
        p = Path(hf_hub_download(repo, f, repo_type="dataset", revision=REV[repo], local_dir=OUT / repo.replace("/", "__")))
        man["files"][f"{repo}/{f}"] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
        print(repo, f, f"{p.stat().st_size / 1e6:.1f} MB", flush=True)
for repo in ("truthfulqa/truthful_qa", "microsoft/wiki_qa", "tau/commonsense_qa"):
    d = Path(snapshot_download(repo, repo_type="dataset", revision=REV[repo], local_dir=OUT / repo.replace("/", "__")))
    for p in sorted(d.rglob("*")):
        if p.is_file() and ".cache" not in p.parts:
            man["files"][f"{repo}/{p.relative_to(d)}"] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    print(repo, "done", flush=True)
url = f"https://raw.githubusercontent.com/{MIRAGE[0]}/{MIRAGE[1]}/benchmark.json"
p = OUT / "MIRAGE_benchmark.json"
p.write_bytes(urllib.request.urlopen(url, timeout=120).read())
man["files"]["MIRAGE/benchmark.json"] = {"bytes": p.stat().st_size, "sha256": sha256(p), "url": url}
man["total_bytes"] = sum(v["bytes"] for v in man["files"].values())
(OUT / "manifest.json").write_text(json.dumps(man, indent=1))
print(f"done: {len(man['files'])} files, {man['total_bytes'] / 1e9:.2f} GB")
