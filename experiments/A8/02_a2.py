from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import PROJECT_ROOT, fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json, set_determinism, stable_seed

set_determinism()
import torch

_spec = importlib.util.spec_from_file_location("a2_eval", PROJECT_ROOT / "experiments" / "A2" / "02_eval.py")
A2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A2)
DEV = "cuda"


def unit_t(X):
    X = torch.as_tensor(np.asarray(X, np.float32), device=DEV)
    return X / X.norm(dim=1, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--retriever", required=True)
    ap.add_argument("--a1-results", default="experiments/A8/results_a1")
    ap.add_argument("--out", default="experiments/A8/results_a2")
    ap.add_argument("--smoke-pairs", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    a2c = cfg["a2"]
    r = args.retriever
    res = PROJECT_ROOT / args.out / r
    (res / "tables").mkdir(parents=True, exist_ok=True)
    log = get_logger("A8.02", res / "logs" / "02_a2.log")
    t0 = time.time()
    k, m, cap, R, acc = cfg["k"], a2c["m"], a2c["cap"], a2c["R"], a2c["id_accept"]
    taus, alphas = a2c["tau_grid"], a2c["alpha_grid"]
    edir = p(cfg, "emb", r)
    sites = load_json(edir / "sites.json")
    sidx = {s: i for i, s in enumerate(sites)}
    twin = {c: v["twin"] for c, v in load_json(p(cfg, "a0_processed", "sites.json")).items() if "twin" in v}
    a0_sites = [s for s in sites if s not in twin]
    T = np.load(edir / "topk.npy", mmap_mode="r")
    qi = pd.read_parquet(edir / "q_index.parquet")
    rows_of = {key: g.index.to_numpy() for key, g in qi.groupby(["site", "role"])}
    Q = unit_t(np.load(p(cfg, "p4_emb", r, "q.npy"))[qi.qrow.to_numpy()])
    Bd = unit_t(np.concatenate([np.load(p(cfg, "a0_emb", r, f"{s}__bank.npy")) for s in a0_sites]))
    aidx = {s: i for i, s in enumerate(a0_sites)}
    assert Bd.shape[0] == len(a0_sites) * cap
    MS = unit_t(np.load(p(cfg, "p9_emb", r, "pool.npy")))
    pairs = pd.read_csv(PROJECT_ROOT / args.a1_results / r / "tables" / "pairs.csv")[["kb", "ood", "tier"]]
    if args.smoke_pairs:
        pairs = pairs.head(args.smoke_pairs)
    rows, sets_rows, maxerr = [], [], 0.0
    for pi_, (kb, o, tier) in enumerate(zip(pairs.kb, pairs.ood, pairs.tier)):
        i_id, i_ood, i_cal = rows_of[(kb, "test")], rows_of[(o, "test")], rows_of[(kb, "cal")]
        sel = np.r_[i_id, i_ood, i_cal]
        n_id, n_ood = len(i_id), len(i_ood)
        Tsel = torch.as_tensor(np.asarray(T[sel]), device=DEV).double()
        s_kb, absk = Tsel[:, sidx[kb]].mean(1), Tsel[:, sidx[kb], k - 1]
        a_auc, a_id, a_ood = A2.gpu_metrics(torch.stack([s_kb, absk]), n_id, n_ood, acc)
        abs_m = {"abs": (a_auc[0], a_id[0], a_ood[0]), "absk": (a_auc[1], a_id[1], a_ood[1])}
        Qp = Q[torch.as_tensor(sel, device=DEV)]
        elig = [s for s in a0_sites if s not in (twin[kb], twin[o])]
        for d in range(R):
            order = [elig[i] for i in np.random.default_rng(stable_seed(cfg["seed"], "A2_sets", kb, o, d)).permutation(len(elig))[:m]]
            sets_rows.append({"kb": kb, "ood": o, "draw": d, "order": "+".join(order)})
            docs = torch.cat([torch.arange(aidx[s] * cap, (aidx[s] + 1) * cap, device=DEV) for s in order])
            S = (Qp @ Bd[docs].T).double()
            n = m * cap
            ms_idx = torch.as_tensor(np.random.default_rng(stable_seed(cfg["seed"], "A2_ms", kb, o, d)).permutation(MS.shape[0])[:n], device=DEV)
            Sm = (Qp @ MS[ms_idx].T).double()
            site_groups = [torch.arange(j * cap, (j + 1) * cap, device=DEV) for j in range(m)]
            pr = torch.as_tensor(np.random.default_rng(stable_seed(cfg["seed"], "A2_rand", kb, o, d, m)).permutation(n), device=DEV)
            rand_groups = [pr[j * cap:(j + 1) * cap] for j in range(m)]
            Gs = A2.group_logits(S, site_groups, k)
            Gt = Tsel[:, [sidx[s] for s in order], :].mean(2)
            maxerr = max(maxerr, float((Gs - Gt).abs().max()))
            names, vecs = [], []
            for part, groups in (("site", site_groups), ("random", rand_groups)):
                G = A2.group_logits(S, groups, k)
                for t in taus:
                    names.append(("se", part, "S_tau", str(t)))
                    vecs.append(A2.s_tau(s_kb, G, t))
            for src, SS in (("se", S), ("ms", Sm)):
                top = SS.topk(k, dim=1).values
                for a in alphas:
                    names.append((src, "pooled", "mix", str(a)))
                    vecs.append(s_kb - a * top.mean(1))
                    names.append((src, "pooled", "mixk", str(a)))
                    vecs.append(absk - a * top[:, k - 1])
            auc, ida, ooda = A2.gpu_metrics(torch.stack(vecs), n_id, n_ood, acc)
            for (src, part, fam, par), A, ia, oa in zip(names, auc, ida, ooda):
                b = "absk" if fam == "mixk" else "abs"
                rows.append({"kb": kb, "ood": o, "tier": tier, "draw": d, "m": m, "corpus": src, "partition": part, "family": fam,
                             "param": par, "auroc": float(A), "op_id_acc": float(ia), "op_ood_acc": float(oa),
                             "auroc_abs": float(abs_m[b][0]), "op_ood_acc_abs": float(abs_m[b][2]), "op_id_acc_abs": float(abs_m[b][1])})
        if pi_ % 20 == 0:
            log.info(f"pair {pi_}/{len(pairs)} ({(time.time() - t0) / 60:.1f} min)")
    log.info(f"site logits: document-level vs top-k max |diff| = {maxerr:.2e}")
    assert maxerr < 1e-5, maxerr
    pd.DataFrame(rows).to_parquet(res / "tables" / "rows.parquet", index=False)
    pd.DataFrame(sets_rows).to_csv(res / "tables" / "sets.csv", index=False)
    meta = run_metadata(cfg, f"A8.02_a2:{r}")
    meta["inputs"] = fingerprint([edir / "topk.npy", edir / "q_index.parquet", p(cfg, "p4_emb", r, "q.npy"), p(cfg, "p9_emb", r, "pool.npy"),
                                  PROJECT_ROOT / args.a1_results / r / "tables" / "pairs.csv"] + [p(cfg, "a0_emb", r, f"{s}__bank.npy") for s in a0_sites])
    meta["minutes"] = (time.time() - t0) / 60
    save_json(meta, res / "meta_02.json")
    log.info(f"rows {len(rows)} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
