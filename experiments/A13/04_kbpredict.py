from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import PROJECT_ROOT, fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json, set_determinism, stable_seed

set_determinism()
import torch

DEV = "cuda"
spec = importlib.util.spec_from_file_location("a12_01", PROJECT_ROOT / "experiments" / "A12" / "01_dose.py")
A12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A12)
PRED = ["rho_sg", "pi_ms", "pi_flats", "rho_sc", "crowd", "crowd_ms"]


def kb_predictors(rn, run, cfg):
    k, n, R = cfg["k"], cfg["m"] * cfg["cap"], cfg["R"]
    r = run["retriever"]
    edir = PROJECT_ROOT / run["emb"]
    sites = load_json(edir / "sites.json")
    sidx = {s: i for i, s in enumerate(sites)}
    Tk = np.asarray(np.load(edir / "topk.npy", mmap_mode="r"), dtype=np.float64)
    Lm, T10 = Tk.mean(2), Tk[:, :, k - 1]
    qi = pd.read_parquet(edir / "q_index.parquet")
    rows_of = {kk: g.index.to_numpy() for kk, g in qi.groupby(["site", "role"])}
    MS = A12.unit_t(np.load(p(cfg, "p9_emb", r, "pool.npy")))
    idx = [np.random.default_rng(stable_seed(cfg["seed"], "A13_bg", "ms", d)).permutation(MS.shape[0])[:n] for d in range(R)]
    a0 = p(cfg, "a0_emb", r)
    out = []
    for kb in sites:
        ic = rows_of[(kb, "cal")]
        Lc = Lm[ic]
        s = Lc[:, sidx[kb]]
        others = [j for j in range(len(sites)) if j != sidx[kb]]
        g, _, wk = A12.factor(Lc, 0, others)
        Qc = A12.unit_t(np.load(a0 / f"{kb}__cal.npy"))
        tops = [(Qc @ MS[torch.as_tensor(ix, device=DEV)].T).topk(k, dim=1).values.double().cpu().numpy() for ix in idx]
        cs = [t_.mean(1) for t_ in tops]
        pis = [s.std() / (s - 0.5 * c).std() - 1 for c in cs]
        sf = -np.sqrt(np.clip(2 - 2 * T10[ic, sidx[kb]], 0, None))
        pif = [sf.std() / (sf + 0.5 * np.sqrt(np.clip(2 - 2 * t_[:, k - 1], 0, None))).std() - 1 for t_ in tops]
        out.append({"kb": kb, "rho_sg": float(np.corrcoef(s, g)[0, 1]), "pi_ms": float(np.mean(pis)), "pi_flats": float(np.mean(pif)),
                    "rho_sc": float(np.mean([np.corrcoef(s, c)[0, 1] for c in cs])), "crowd": float(Lc[:, others].max(1).mean()),
                    "crowd_ms": float(np.mean([c.mean() for c in cs])), "g_wk": wk})
    return pd.DataFrame(out)


def loko_spearman(X, y, groups=None):
    X = np.atleast_2d(X.T).T if X.ndim == 1 else X
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    pred = np.empty(len(y))
    for i in range(len(y)):
        m = groups != groups[i]
        A = np.column_stack([np.ones(m.sum()), X[m]])
        beta = np.linalg.lstsq(A, y[m], rcond=None)[0]
        pred[i] = X[i] @ beta[1:]
    return spearmanr(pred, y).statistic


def boot_sp(x, y, B, seed):
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        ix = rng.integers(0, len(x), len(x))
        bs.append(spearmanr(x[ix], y[ix]).statistic)
    return np.nanquantile(bs, [0.025, 0.975])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--res", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = Path(args.res) if args.res else p(cfg, "results")
    log = get_logger("A13.04", res / "logs" / "04_kbpredict.log")
    B = cfg["metrics"]["bootstrap_B"]
    runs = [r for r in cfg["runs"] if not args.runs or r in args.runs]
    L = ["# A13 — predicting each KB's FLatS gain from in-scope data (calibration queries and other corpora only)", "",
         "- Outcomes and predictors: see the script docstring (primary: KB mean G of flats_ms@0.5 on cells with A_abs ≤ 0.98, KBs with ≥ 3 such cells; predictor ρ_sg = ρ(s_KB, g) on calibration queries). No pass criterion.",
         "- Spearman [KB bootstrap 95%]; LOKO = leave-one-KB-out prediction Spearman (slope part only; bootstrap leaves out all copies of a KB). Forecasts: JOURNAL 2026-10-07 20:05.",
         "- Our choices (not in the 20:14 decisions): KBs need ≥ 3 G cells; LOKO slope-only ranking (A7 rule).",
         "- In-scope data: the KB's calibration queries plus other corpora (MS MARCO; the other StackExchange sites' document banks for g and crowd). The other SE banks include the sites that serve as O in the outcome cells; no out-of-scope query, label or site identity is used (D4).", ""]
    pooled = []
    for rn in runs:
        P = kb_predictors(rn, cfg["runs"][rn], cfg)
        C = pd.read_parquet(res / "tables" / f"cellsG_{rn}.parquet")
        Y = {}
        for nm, sc, col in (("G_flats", "flats_ms@0.5", "G"), ("dA_flats", "flats_ms@0.5", "dA"), ("G_mix", "mix_ms@0.5", "G")):
            d = C[C.score == sc]
            agg = d.groupby("kb")[col].agg(["mean", "count"])
            Y[nm] = agg["mean"].where(agg["count"] >= 3)
        K = P.set_index("kb").join(pd.DataFrame(Y))
        K.to_csv(res / "tables" / f"kbpredict_{rn}.csv")
        both = K[["G_flats", "G_mix"]].dropna()
        L_agree = f"KB-level Spearman(G_flats, G_mix) = {spearmanr(both.G_flats, both.G_mix).statistic:+.2f} (how interchangeable the flats and mix forms are; π_ms is mix-form, π_flats flats-form)"
        L += [f"## {rn} ({K.G_flats.notna().sum()} KBs with ≥ 3 G cells of {len(K)})", "",
              "Predictor distribution (KB 5 / 50 / 95%): " + "; ".join(f"{x} " + " / ".join(f"{v:+.3f}" for v in K[x].quantile([.05, .5, .95])) for x in PRED), "", L_agree, "",
              "| outcome | predictor | Spearman [95%] | LOKO Spearman [95%] |", "|---|---|---|---|"]
        for yn in ("G_flats", "dA_flats", "G_mix"):
            Kd = K[K[yn].notna()]
            y = Kd[yn].to_numpy()
            for x in PRED + ["rho_sg+crowd"]:
                if x == "rho_sg+crowd":
                    X = Kd[["rho_sg", "crowd"]].to_numpy()
                    sp, ci = np.nan, (np.nan, np.nan)
                else:
                    X = Kd[x].to_numpy()
                    sp = spearmanr(X, y).statistic
                    ci = boot_sp(X, y, B, stable_seed(cfg["seed"], "A13_kbp", rn, yn, x))
                lo = loko_spearman(X, y)
                rng = np.random.default_rng(stable_seed(cfg["seed"], "A13_kbp_loko", rn, yn, x))
                bl = [loko_spearman(X[ix], y[ix], ix) for ix in (rng.integers(0, len(y), len(y)) for _ in range(200))]
                L.append(f"| {yn} | {x} | {sp:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] | {lo:+.2f} [{np.nanquantile(bl, .025):+.2f}, {np.nanquantile(bl, .975):+.2f}] |")
        L.append("")
        if rn.startswith("a5_"):
            pooled.append(K.assign(model=rn[3:]))
        log.info(f"{rn} done")
    if pooled:
        A = pd.concat(pooled)
        A = A[A.G_flats.notna()]
        L += [f"## Pooled A5 (model × KB units: {len(A)})", "", "| predictor | Spearman raw [95%] | within-model Spearman (ranks within model) |", "|---|---|---|"]
        kb_ids = sorted(A.index.unique())
        rows_by_kb = {kb: np.flatnonzero(A.index == kb) for kb in kb_ids}
        rng = np.random.default_rng(stable_seed(cfg["seed"], "A13_kbp_pool"))
        reps = [np.concatenate([rows_by_kb[kb_ids[j]] for j in rng.integers(0, len(kb_ids), len(kb_ids))]) for _ in range(B)]
        L[-2] = "| predictor | Spearman raw [KB-clustered 95%] | within-model Spearman [KB-clustered 95%] |"
        for x in PRED:
            sp = spearmanr(A[x], A.G_flats).statistic
            rx = A.groupby("model")[x].rank().to_numpy()
            ry = A.groupby("model").G_flats.rank().to_numpy()
            bs = [spearmanr(A[x].to_numpy()[ix], A.G_flats.to_numpy()[ix]).statistic for ix in reps]
            bw = [spearmanr(rx[ix], ry[ix]).statistic for ix in reps]
            L.append(f"| {x} | {sp:+.2f} [{np.nanquantile(bs, .025):+.2f}, {np.nanquantile(bs, .975):+.2f}] | {spearmanr(rx, ry).statistic:+.2f} "
                     f"[{np.nanquantile(bw, .025):+.2f}, {np.nanquantile(bw, .975):+.2f}] |")
        L.append("")
    (res / "A13_kbpredict.md").write_text("\n".join(L) + "\n")
    meta = run_metadata(cfg, "A13.04_kbpredict")
    meta["inputs"] = fingerprint([res / "tables" / f"cellsG_{rn}.parquet" for rn in runs] + [PROJECT_ROOT / cfg["runs"][rn]["emb"] / "topk.npy" for rn in runs]
                                 + [p(cfg, "p9_emb", r_, "pool.npy") for r_ in sorted({cfg["runs"][rn]["retriever"] for rn in runs})]
                                 + [p(cfg, "a0_emb", cfg["runs"][rn]["retriever"], f"{s_}__cal.npy") for rn in runs for s_ in load_json(PROJECT_ROOT / cfg["runs"][rn]["emb"] / "sites.json")])
    meta["imported_code_sha256"] = fingerprint([PROJECT_ROOT / "experiments" / "A12" / "01_dose.py"])
    save_json(meta, res / "meta_04.json")


if __name__ == "__main__":
    main()
