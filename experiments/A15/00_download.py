from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import get_logger, load_config, p, save_json, sha256_file
from huggingface_hub import hf_hub_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    cfg = load_config(ap.parse_args().config)
    log = get_logger("A15.00", p(cfg, "results", "logs", "00_download.log"))
    for name, d in cfg["download"].items():
        if name.startswith("_"):
            continue
        out = p(cfg, "raw", d["local"])
        out.mkdir(parents=True, exist_ok=True)
        man = {"repo": d["repo"], "revision": d["revision"], "license": d["license"], "files": {}}
        for f in d["files"]:
            src = hf_hub_download(d["repo"], f, repo_type="dataset", revision=d["revision"])
            dst = out / Path(f).name
            shutil.copyfile(src, dst)
            man["files"][Path(f).name] = {"sha256": sha256_file(dst), "bytes": dst.stat().st_size}
            log.info(f"{name}: {f} -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
        save_json(man, out / "manifest.json")


if __name__ == "__main__":
    main()
