from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import PROJECT_ROOT, fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json, set_determinism

set_determinism()
import torch

DEV = "cuda"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A2 = _load("a2_eval", PROJECT_ROOT / "experiments" / "A2" / "02_eval.py")
B1 = _load("a13_01", PROJECT_ROOT / "experiments" / "A13" / "01_baselines.py")


def run_one(rn, run, cfg, log, smoke_pairs=0):
    k, cap, mt = cfg["k"], cfg["cap"], cfg["metrics"]
    r = run["retriever"]
    edir = PROJECT_ROOT / run["emb"]
    sites = load_json(edir / "sites.json")
    sidx = {s: i for i, s in enumerate(sites)}
    qi = pd.read_parquet(edir / "q_index.parquet")
    rows_of = {kk: g.index.to_numpy() for kk, g in qi.groupby(["site", "role"])}
    a0 = p(cfg, "a0_emb", r)
    Q = B1.unit_t(np.concatenate([np.load(a0 / f"{s}__{ro}.npy") for s in sites for ro in ("test", "cal")]))
    pairs = pd.read_csv(PROJECT_ROOT / run["a1"] / "tables" / "pairs.csv")[["kb", "ood", "tier"]]
    if smoke_pairs:
        pairs = pairs.iloc[:: max(1, len(pairs) // smoke_pairs)].head(smoke_pairs)
    base_abs = pd.read_parquet(p(cfg, "results", "tables", f"cells_{rn}.parquet"))
    base_abs = base_abs[base_abs.score == "abs"].set_index(["kb", "ood"]).auroc
    out, cache, maxerr = [], {}, 0.0
    t0 = time.time()
    for pi_, (kb, o, tier) in enumerate(zip(pairs.kb, pairs.ood, pairs.tier)):
        if kb not in cache:
            cache.clear()
            Bk = B1.unit_t(np.load(a0 / f"{kb}__bank.npy"))
            mu = Bk.mean(0)
            _, sv, Vt = torch.linalg.svd((Bk - mu).double(), full_matrices=False)
            ev = (sv ** 2) / (sv ** 2).sum()
            m90 = int(torch.searchsorted(torch.cumsum(ev, 0), torch.tensor(0.9, dtype=ev.dtype, device=ev.device)).item()) + 1
            cache[kb] = (Bk, mu, Vt[: cfg["kbpca"]["m"]].float(), Vt[:m90].float(), m90, float(ev[: cfg["kbpca"]["m"]].sum()))
        Bk, mu, V15, V90, m90, evr15 = cache[kb]
        i_id, i_ood, i_cal = rows_of[(kb, "test")], rows_of[(o, "test")], rows_of[(kb, "cal")]
        sel = np.r_[i_id, i_ood, i_cal]
        n_id, n_ood = len(i_id), len(i_ood)
        Qs = Q[torch.as_tensor(sel, device=DEV)]
        Skb = (Qs @ Bk.T).double()
        s = Skb.topk(k, dim=1).values.mean(1)
        Xc = Qs - mu
        sc = {"kbpca_rec15": -(Xc - (Xc @ V15.T) @ V15).norm(dim=1).double(),
              "kbpca_rec90": -(Xc - (Xc @ V90.T) @ V90).norm(dim=1).double(),
              "kb_mean": Skb.mean(1), "kb_sd": Skb.std(1), "kb_peak": s - Skb.mean(1), "kb_scaled": s / Skb.std(1), "abs_chk": s}
        names = list(sc)
        Sc = torch.stack([sc[x] for x in names])
        auc, ida, ooda = A2.gpu_metrics(Sc, n_id, n_ood, mt["op_accept"])
        fp = B1.fpr95(Sc, n_id, n_ood)
        rec = {nm: (a_, ia, oa, f_) for nm, a_, ia, oa, f_ in zip(names, auc, ida, ooda, fp)}
        if (kb, o) in base_abs.index:
            maxerr = max(maxerr, abs(rec["abs_chk"][0] - base_abs[(kb, o)]))
        for nm in names:
            if nm == "abs_chk":
                continue
            a_, ia, oa, f_ = rec[nm]
            out.append({"run": rn, "kb": kb, "ood": o, "tier": tier, "score": nm, "auroc_abs": rec["abs_chk"][0], "auroc": a_,
                        "op_id_acc": ia, "op_ood_acc": oa, "fpr95": f_, "m90": m90, "evr15": evr15})
        if pi_ % 200 == 0:
            log.info(f"{rn}: pair {pi_}/{len(pairs)} ({(time.time() - t0) / 60:.1f} min)")
    log.info(f"{rn}: abs vs 01 cells: max |dAUROC| = {maxerr:.2e}")
    assert maxerr < 1e-9, maxerr
    return pd.DataFrame(out), maxerr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--smoke-pairs", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    assert not args.smoke_pairs or args.out, "--smoke-pairs requires --out"
    cfg = load_config(args.config)
    res = Path(args.out) if args.out else p(cfg, "results")
    (res / "tables").mkdir(parents=True, exist_ok=True)
    log = get_logger("A13.03", res / "logs" / "03_extra.log")
    for rn in args.runs or list(cfg["runs"]):
        run = cfg["runs"][rn]
        t0 = time.time()
        D, e = run_one(rn, run, cfg, log, args.smoke_pairs)
        D.to_parquet(res / "tables" / f"cells_extra_{rn}.parquet", index=False)
        meta = run_metadata(cfg, f"A13.03_extra:{rn}")
        meta["inputs"] = fingerprint([PROJECT_ROOT / run["emb"] / f for f in ("q_index.parquet", "sites.json")] + [PROJECT_ROOT / run["a1"] / "tables" / "pairs.csv",
                                     p(cfg, "results", "tables", f"cells_{rn}.parquet")]
                                     + [p(cfg, "a0_emb", run["retriever"], f"{s_}__{ro}.npy") for s_ in load_json(PROJECT_ROOT / run["emb"] / "sites.json") for ro in ("test", "cal", "bank")])
        meta["imported_code_sha256"] = fingerprint([PROJECT_ROOT / "experiments" / "A2" / "02_eval.py", PROJECT_ROOT / "experiments" / "A13" / "01_baselines.py"])
        meta["abs_check"], meta["minutes"] = e, (time.time() - t0) / 60
        save_json(meta, res / f"meta_03_{rn}.json")
        log.info(f"{rn}: {len(D)} rows in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
