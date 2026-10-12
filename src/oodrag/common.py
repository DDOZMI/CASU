from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
import zlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    cfg["_config_path"] = str(Path(path).resolve())
    cfg["_config_sha256"] = sha256_file(path)
    return cfg


def p(cfg: dict, key: str, *parts: str) -> Path:
    base = Path(cfg["paths"][key])
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    return base.joinpath(*parts)


def stable_seed(base: int, *keys) -> int:
    h = zlib.crc32(("|".join(str(k) for k in keys)).encode())
    return (base * 1_000_003 + h) % (2**31 - 1)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def set_determinism():

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_MEM_FRACTION", "0.5")))


def _git(*args) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def run_metadata(cfg: dict, step: str) -> dict:
    meta = {
        "step": step,
        "time_start": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": cfg.get("_config_path"),
        "config_sha256": cfg.get("_config_sha256"),
        "argv": sys.argv,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty_tracked": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }
    lock = PROJECT_ROOT / "requirements.lock.txt"
    if lock.exists():
        meta["requirements_lock_sha256"] = sha256_file(lock)
    meta["code_sha256"] = {}
    script = Path(sys.argv[0]).resolve()
    if script.is_file():
        meta["code_sha256"][str(script.relative_to(PROJECT_ROOT))] = sha256_file(script)
    for f in sorted((PROJECT_ROOT / "src" / "oodrag").glob("*.py")):
        meta["code_sha256"][str(f.relative_to(PROJECT_ROOT))] = sha256_file(f)
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["cuda"] = torch.version.cuda
        meta["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        meta["float32_matmul_precision"] = torch.get_float32_matmul_precision()
        meta["allow_tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
        meta["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
        meta["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        meta["pytorch_cuda_alloc_conf"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        meta["gpu_mem_fraction"] = os.environ.get("GPU_MEM_FRACTION", "0.5")
    except Exception as e:
        meta["torch_error"] = repr(e)
    return meta


def model_fingerprint(model_dir: str | Path) -> dict:
    model_dir = Path(model_dir)
    return {str(f.relative_to(model_dir)): sha256_file(f) for f in sorted(model_dir.rglob("*"))
            if f.is_file() and ".cache" not in f.relative_to(model_dir).parts}


def fingerprint(paths) -> dict:
    out = {}
    for x in paths:
        x = Path(x).resolve()
        key = str(x.relative_to(PROJECT_ROOT)) if x.is_relative_to(PROJECT_ROOT) else str(x)
        out[key] = sha256_file(x)
    return out


def check_fingerprint(meta_path: Path, expected: dict, what: str):
    if not Path(meta_path).exists():
        raise RuntimeError(f"{what}: output exists but its meta {meta_path} is missing; delete the output to rebuild")
    got = load_json(meta_path).get("upstream_fingerprint")
    if got != expected:
        raise RuntimeError(f"{what}: upstream inputs changed since it was built "
                           f"(stored {got} vs now {expected}); delete the output to rebuild")


def get_logger(name: str, logfile: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
