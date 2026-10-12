import hashlib, json, time, urllib.request
from pathlib import Path

CATS = ["Automotive", "Baby", "Beauty", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Electronics",
        "Grocery_and_Gourmet_Food", "Health_and_Personal_Care", "Home_and_Kitchen", "Musical_Instruments",
        "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies", "Sports_and_Outdoors",
        "Tools_and_Home_Improvement", "Toys_and_Games", "Video_Games"]
QA_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/qa/qa_{}.json.gz"
META_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_{}.json.gz"
OUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "amazon_qa"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        hdr = {k: r.headers.get(k) for k in ("Content-Length", "Last-Modified", "ETag")}
        n = int(hdr["Content-Length"]) if hdr["Content-Length"] else None
        if not (dst.exists() and n is not None and dst.stat().st_size == n):
            tmp = dst.with_suffix(dst.suffix + ".part")
            with open(tmp, "wb") as f:
                for b in iter(lambda: r.read(1 << 20), b""):
                    f.write(b)
            tmp.rename(dst)
    assert n is None or dst.stat().st_size == n, (url, n, dst.stat().st_size)
    return {"url": url, "bytes": dst.stat().st_size, "sha256": sha256(dst), "last_modified": hdr["Last-Modified"],
            "etag": hdr["ETag"], "fetched": time.strftime("%Y-%m-%d %H:%M:%S %Z")}


def fetch_retry(url, dst, tries=5):
    for k in range(tries):
        try:
            return fetch(url, dst)
        except Exception as e:
            print(f"  retry {k + 1}/{tries} {url}: {type(e).__name__} {e}", flush=True)
            for t in dst.parent.glob(dst.name + ".part"):
                t.unlink()
            time.sleep(10)
    raise RuntimeError(f"failed after {tries} tries: {url}")


man = {"scope": "17 QA categories with per-category metadata (user 2026-10-08)", "categories": CATS, "files": {}}
jobs = [(f"qa/qa_{c}.json.gz", QA_URL.format(c)) for c in CATS] + [(f"meta/meta_{c}.json.gz", META_URL.format(c)) for c in CATS]
for i, (rel, url) in enumerate(jobs):
    man["files"][rel] = fetch_retry(url, OUT / rel)
    print(f"[{i + 1}/{len(jobs)}] {rel} {man['files'][rel]['bytes'] / 1e6:.1f} MB", flush=True)
man["total_bytes"] = sum(v["bytes"] for v in man["files"].values())
(OUT / "manifest.json").write_text(json.dumps(man, indent=1))
print(f"done: {len(jobs)} files, {man['total_bytes'] / 1e9:.2f} GB")
