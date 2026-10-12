import hashlib, json, re, time, urllib.request
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO, REV = "flax-sentence-embeddings/stackexchange_title_body_jsonl", "a3d99bf21570ed043e19e41af46f3f19bf4e4bb6"
OUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "stackexchange"
OUT.mkdir(parents=True, exist_ok=True)
EXCLUDE = {"stackoverflow.com-Posts.jsonl.gz", "math.stackexchange.com.jsonl.gz", "small_stackexchanges.jsonl.gz"}

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset", revision=REV)
               if f.endswith(".jsonl.gz") and f not in EXCLUDE and not f.startswith("meta."))
man = {"repo": REPO, "revision": REV, "excluded": sorted(EXCLUDE) + ["meta.*"], "files": {}}
for i, f in enumerate(files):
    p = Path(hf_hub_download(REPO, f, repo_type="dataset", revision=REV, local_dir=OUT))
    man["files"][f] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    print(f"[{i + 1}/{len(files)}] {f} {p.stat().st_size / 1e6:.1f} MB", flush=True)

req = urllib.request.Request("https://stackexchange.com/sites?view=list", headers={"User-Agent": "Mozilla/5.0"})
html = urllib.request.urlopen(req, timeout=60).read()
(OUT / "site_categories.html").write_bytes(html)
pairs = re.findall(r'class="[^"]*(category-(?:technology|culturerecreation|lifearts|science|professional|business|uncategorized))[^"]*".{0,800}?href="(https://[a-z0-9.\-]+)"',
                   html.decode("utf-8", "ignore"), flags=re.S)
cat = {}
for c, u in pairs:
    cat.setdefault(u.replace("https://", ""), c.replace("category-", ""))
(OUT / "site_categories.csv").write_text("site,category\n" + "".join(f"{s},{c}\n" for s, c in sorted(cat.items())))
man["site_categories"] = {"url": "https://stackexchange.com/sites?view=list", "fetched": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                          "sha256_html": hashlib.sha256(html).hexdigest(), "n_sites": len(cat)}
man["total_bytes"] = sum(v["bytes"] for v in man["files"].values())
(OUT / "manifest.json").write_text(json.dumps(man, indent=1))
print(f"done: {len(files)} files, {man['total_bytes'] / 1e9:.2f} GB, {len(cat)} site categories")
