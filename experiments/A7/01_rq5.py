from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm, pearsonr, rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import PROJECT_ROOT, fingerprint, get_logger, load_config, load_json, p, run_metadata, save_json, set_determinism, stable_seed

set_determinism()
import torch


def agg(A, t):
    return A.mean(1) if t == "inf" else float(t) * (logsumexp(A / float(t), axis=1) - np.log(A.shape[1]))


def auroc(neg, pos):
    r = rankdata(np.concatenate([neg, pos]))
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def unit(X):
    X = torch.as_tensor(np.asarray(X, np.float32), device="cuda")
    return X / X.norm(dim=1, keepdim=True)


def kb_table(rn, run, cfg, log):
    root = PROJECT_ROOT
    edir = root / run["emb"]
    sites = load_json(edir / "sites.json")
    sidx = {s: i for i, s in enumerate(sites)}
    Lm = np.asarray(np.load(edir / "topk.npy", mmap_mode="r"), dtype=np.float64).mean(2)
    qi = pd.read_parquet(edir / "q_index.parquet")
    rows_of = {k: g.index.to_numpy() for k, g in qi.groupby(["site", "role"])}
    sinfo = load_json(p(cfg, "a0_processed", "sites.json"))
    sets = pd.read_csv(root / run["a2"] / "tables" / "sets.csv")
    R2 = pd.read_parquet(root / run["a2"] / "tables" / "rows.parquet",
                         columns=["kb", "ood", "draw", "m", "corpus", "partition", "family", "param", "auroc", "auroc_abs"])
    R2 = R2[(R2.m == cfg["m"]) & (R2.corpus == "se") & (R2.partition == "site") & (R2.family == "S_tau")].copy()
    R2["param"] = R2.param.astype(str)
    clip = cfg["dprime_clip"]
    dp = lambda a: np.sqrt(2) * norm.ppf(np.clip(a, clip, 1 - clip))
    R2["ddp"] = dp(R2.auroc.to_numpy()) - dp(R2.auroc_abs.to_numpy())
    R2["dA"] = R2.auroc - R2.auroc_abs
    a2auc = R2.set_index(["kb", "ood", "draw", "param"]).auroc
    MS = unit(np.load(p(cfg, "p9_emb", run["retriever"], "pool.npy")))
    rows, maxerr, nchk = [], 0.0, 0
    for kb in sites:
        i_cal = rows_of[(kb, "cal")]
        s = Lm[i_cal, sidx[kb]]
        g = sets[sets.kb == kb]
        o_set = set(g.ood)
        non_o = [sidx[x] for x in sites if x != kb and x not in o_set]
        Q = unit(np.load(p(cfg, "a0_emb", run["retriever"], f"{kb}__cal.npy")))
        assert Q.shape[0] == len(i_cal)
        gen = float((Q @ MS.T).topk(cfg["k"], dim=1).values.mean(1).mean().item())
        if len(non_o) >= cfg["other_banks"]["min_strict"]:
            groups, mode = [non_o], "strict"
        else:
            groups, mode = [[sidx[x] for x in sites if x not in (kb, o)] for o in sorted(o_set)], "per_cell"
        crowd = np.mean([Lm[np.ix_(i_cal, G)].max(1).mean() for G in groups])
        d_hat = np.mean([((s - Lm[np.ix_(i_cal, G)].mean(1)) / s.std()).mean() for G in groups])
        all_oth = [sidx[x] for x in sites if x != kb]
        percell = [[sidx[x] for x in sites if x not in (kb, o)] for o in sorted(o_set)]
        rec = {"kb": kb, "log_n": float(np.log(sinfo[kb]["n_kept"])), "sd_kb": float(s.std()), "mu_kb": float(s.mean()), "gen": gen,
               "crowd": float(crowd), "d_hat": float(d_hat), "other_banks": mode, "n_other": int(np.mean([len(G) for G in groups])),
               "crowd_all": float(Lm[np.ix_(i_cal, all_oth)].max(1).mean()),
               "crowd_percell": float(np.mean([Lm[np.ix_(i_cal, G)].max(1).mean() for G in percell]))}
        for t in cfg["taus"]:
            pis, rhos = [], []
            for o, d, order in zip(g.ood, g.draw, g.order):
                C = [sidx[x] for x in order.split("+")[: cfg["m"]]]
                c = agg(Lm[np.ix_(i_cal, C)], t)
                r_, rho = c.std() / s.std(), np.corrcoef(s, c)[0, 1]
                pis.append(1 / np.sqrt(1 + r_ ** 2 - 2 * rho * r_) - 1)
                rhos.append(rho)
                if d == 0:
                    i_id, i_ood = rows_of[(kb, "test")], rows_of[(o, "test")]
                    S_id = Lm[i_id, sidx[kb]] - agg(Lm[np.ix_(i_id, C)], t)
                    S_ood = Lm[i_ood, sidx[kb]] - agg(Lm[np.ix_(i_ood, C)], t)
                    maxerr = max(maxerr, abs(auroc(-S_id, -S_ood) - a2auc.loc[(kb, o, d, t)]))
                    nchk += 1
            rec[f"pi_var@{t}"], rec[f"rho@{t}"] = float(np.mean(pis)), float(np.mean(rhos))
            rec[f"theory@{t}"] = rec[f"pi_var@{t}"] * rec["d_hat"]
            q = R2[(R2.kb == kb) & (R2.param == t)].groupby("ood")[["ddp", "dA"]].mean()
            rec[f"y_ddp@{t}"], rec[f"y_dA@{t}"] = float(q.ddp.mean()), float(q.dA.mean())
            rec["n_cells"] = int(len(q))
        rows.append(rec)
    log.info(f"{rn}: max |AUROC(recomputed) - A2| = {maxerr:.2e} over {nchk} checks")
    assert nchk > 0 and maxerr < 1e-5, f"{rn}: comparison sets do not reproduce A2 ({maxerr})"
    return pd.DataFrame(rows), maxerr


def loko(X, y):
    n = len(y)
    part, full = np.empty(n), np.empty(n)
    for i in range(n):
        tr = np.arange(n) != i
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1
        A = np.column_stack([np.ones(tr.sum()), (X[tr] - mu) / sd])
        beta = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        z = (X[i] - mu) / sd
        part[i], full[i] = z @ beta[1:], beta[0] + z @ beta[1:]
    return part, full


def q2(full, y):
    n = len(y)
    mean_only = np.array([np.delete(y, i).mean() for i in range(n)])
    return 1 - ((y - full) ** 2).sum() / ((y - mean_only) ** 2).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = p(cfg, "results")
    (res / "tables").mkdir(parents=True, exist_ok=True)
    log = get_logger("A7.01", res / "logs" / "01_rq5.log")
    pc = cfg["predictors"]
    srows = []
    L = ["# A7 — RQ5 (exploratory): ID statistics predicting the per-KB gain of relative judgment", "",
         "- Predictors from the KB's calibration queries only (no OOD query); target = KB mean of the A2 site (m = 4) gain on test queries. The theory form is post hoc informed (A5 05_offdiag); see config `_source`.",
         f"- Judgment (fixed before the run): {cfg['judgment']}",
         f"- Primary (fixed before the run): {cfg['primary']['model']} on the main target, each τ read separately ({cfg['primary']['_source']}).",
         "- LOKO predictions are ranked by their slope part (held-out intercept dropped); q² uses the full prediction against a leave-one-out mean-only model. The bootstrap is conditional on the fixed LOKO predictions; KB targets are not independent (a KB's test queries are OOD in other KBs' cells).", ""]
    for rn, run in cfg["runs"].items():
        K, maxerr = kb_table(rn, run, cfg, log)
        K.to_csv(res / "tables" / f"kb_{rn}.csv", index=False)
        n = len(K)
        L += [f"## {rn} ({n} KBs; reproduction check max |ΔAUROC| = {maxerr:.1e}; other banks for crowd / d_hat: {K.other_banks.iloc[0]}, mean {K.n_other.mean():.0f} banks)", ""]
        L += ["### Predictor summary (mean, SD) and in-sample Spearman with the target", "",
              "| predictor | mean | SD | " + " | ".join(f"ρ(y_{y}@{t})" for y in [cfg["target"]["main"]] + cfg["target"]["secondary"] for t in cfg["taus"]) + " |",
              "|---" * (3 + len(cfg["taus"]) * (1 + len(cfg["target"]["secondary"]))) + "|"]
        for c in ["theory@" + t for t in cfg["taus"]] + ["pi_var@" + t for t in cfg["taus"]] + ["rho@" + t for t in cfg["taus"]] + ["d_hat"] + pc["simple"]:
            L.append(f"| {c} | {K[c].mean():+.3f} | {K[c].std():.3f} | " + " | ".join(
                f"{spearmanr(K[c], K[f'y_{y}@{t}']).statistic:+.2f}" for y in [cfg["target"]["main"]] + cfg["target"]["secondary"] for t in cfg["taus"]) + " |")
        L += ["", "### LOKO prediction (Spearman / Pearson with observed; LOKO q² vs mean-only; Δ vs best simple baseline, KB bootstrap 95%)", "",
              "| target | model | Spearman [KB-bootstrap 95%] | Pearson | q² | Δ vs max baseline [95%] | beats baselines |", "|---|---|---|---|---|---|---|"]
        rng = np.random.default_rng(stable_seed(cfg["seed"], "A7_boot", rn))
        Bi = rng.integers(0, n, size=(cfg["bootstrap_B"], n))
        for y in [cfg["target"]["main"]] + cfg["target"]["secondary"]:
            for t in cfg["taus"]:
                yv = K[f"y_{y}@{t}"].to_numpy()
                col = lambda c: K[c if c in K else f"{c}@{t}"].to_numpy()
                fits = {"M1 LOKO [pi_var, d_hat, crowd]": loko(np.column_stack([col(c) for c in pc["m1"]]), yv),
                        "M2 LOKO all 8": loko(np.column_stack([col(c) for c in pc["m2"]]), yv)}
                preds = {"M0 theory": (col("theory"), None), **fits}
                base = {f"B {c}": loko(col(c)[:, None], yv) for c in pc["simple"]}
                sp = lambda a, idx=None: spearmanr(a if idx is None else a[idx], yv if idx is None else yv[idx]).statistic
                bmax_bs = np.array([np.nanmax([sp(b[0], ix) for b in base.values()]) for ix in Bi])
                bmax = np.nanmax([sp(b[0]) for b in base.values()])
                for nm, (pr, full) in {**preds, **base}.items():
                    s0, r0 = sp(pr), pearsonr(pr, yv).statistic
                    sbs = np.array([sp(pr, ix) for ix in Bi])
                    slo, shi = np.nanquantile(sbs, [0.025, 0.975])
                    sci = f"{s0:+.3f} [{slo:+.3f}, {shi:+.3f}]"
                    qq = q2(full, yv) if full is not None else np.nan
                    row = {"run": rn, "target": f"{y}@{t}", "model": nm, "spearman": s0, "sp_lo": slo, "sp_hi": shi, "pearson": r0, "q2": qq,
                           "primary": bool(nm == cfg["primary"]["model"] and y == cfg["target"]["main"])}
                    if nm in preds:
                        dbs = sbs - bmax_bs
                        lo, hi = np.nanquantile(dbs, [0.025, 0.975])
                        dpt = s0 - bmax
                        row.update({"d_vs_maxbase": dpt, "lo": lo, "hi": hi, "beats": bool(lo > 0)})
                        L.append(f"| {y}@{t} | {nm}{' (primary)' if row['primary'] else ''} | {sci} | {r0:+.3f} | {qq:+.3f} | {dpt:+.3f} [{lo:+.3f}, {hi:+.3f}] | {'yes' if lo > 0 else 'no'} |")
                    else:
                        L.append(f"| {y}@{t} | {nm} | {sci} | {r0:+.3f} | {qq:+.3f} | | |")
                    srows.append(row)
        L += ["", "### crowd variants: LOKO prediction Spearman (same statistic as the table above; KB-bootstrap 95%)", "",
              "| variant | " + " | ".join(f"{y}@{t}" for y in [cfg["target"]["main"]] + cfg["target"]["secondary"] for t in cfg["taus"]) + " |",
              "|---" * (1 + len(cfg["taus"]) * (1 + len(cfg["target"]["secondary"]))) + "|"]
        for c in ("crowd", "crowd_all", "crowd_percell"):
            cells = []
            for y in [cfg["target"]["main"]] + cfg["target"]["secondary"]:
                for t in cfg["taus"]:
                    yv = K[f"y_{y}@{t}"].to_numpy()
                    pr = loko(K[c].to_numpy()[:, None], yv)[0]
                    bs = np.array([spearmanr(pr[ix], yv[ix]).statistic for ix in Bi])
                    lo, hi = np.nanquantile(bs, [0.025, 0.975])
                    s0 = spearmanr(pr, yv).statistic
                    cells.append(f"{s0:+.2f} [{lo:+.2f}, {hi:+.2f}]")
                    srows.append({"run": rn, "target": f"{y}@{t}", "model": f"LOKO {c}", "spearman": s0, "sp_lo": lo, "sp_hi": hi})
            L.append(f"| {c} | " + " | ".join(cells) + " |")
        L += ["", "### crowd as a pre-computable diagnostic (in-sample Spearman with the KB gain; KB-bootstrap 95%)", "",
              "crowd ≈ the comparison term that relative scoring subtracts from ID queries, so part of its predictive power is mechanical; it is reported as a diagnostic computable without OOD data, not as a mechanism. Variants: crowd (config other_banks rule), crowd_all (all other banks; no knowledge of which sites are O, includes O banks), crowd_percell (each cell's O excluded, averaged over the KB's cells). One value per KB in every variant.", "",
              "| variant | " + " | ".join(f"ρ(y_{y}@{t})" for y in [cfg["target"]["main"]] + cfg["target"]["secondary"] for t in cfg["taus"]) + " |",
              "|---" * (1 + len(cfg["taus"]) * (1 + len(cfg["target"]["secondary"]))) + "|"]
        for c in ("crowd", "crowd_all", "crowd_percell"):
            cells = []
            for y in [cfg["target"]["main"]] + cfg["target"]["secondary"]:
                for t in cfg["taus"]:
                    xv, yv = K[c].to_numpy(), K[f"y_{y}@{t}"].to_numpy()
                    bs = np.array([spearmanr(xv[ix], yv[ix]).statistic for ix in Bi])
                    lo, hi = np.nanquantile(bs, [0.025, 0.975])
                    cells.append(f"{spearmanr(xv, yv).statistic:+.2f} [{lo:+.2f}, {hi:+.2f}]")
                    srows.append({"run": rn, "target": f"{y}@{t}", "model": f"in-sample {c}", "spearman": spearmanr(xv, yv).statistic, "sp_lo": lo, "sp_hi": hi})
            L.append(f"| {c} | " + " | ".join(cells) + " |")
        L.append("")
        log.info(f"{rn} done")
    L += ["## Model level (30-KB subset, descriptive)", "", "| run | " + " | ".join(f"pi_var@{t} | y_ddp@{t}" for t in cfg["taus"]) + " |",
          "|---" * (1 + 2 * len(cfg["taus"])) + "|"]
    for rn in cfg["runs"]:
        if rn.startswith("a5_"):
            K = pd.read_csv(res / "tables" / f"kb_{rn}.csv")
            L.append(f"| {rn} | " + " | ".join(f"{K[f'pi_var@{t}'].mean():+.3f} | {K[f'y_ddp@{t}'].mean():+.3f}" for t in cfg["taus"]) + " |")
    pd.DataFrame(srows).to_csv(res / "tables" / "scores.csv", index=False)
    (res / "A7_summary.md").write_text("\n".join(L) + "\n")
    meta = run_metadata(cfg, "A7.01_rq5")
    meta["inputs"] = fingerprint([PROJECT_ROOT / r["emb"] / f for r in cfg["runs"].values() for f in ("topk.npy", "q_index.parquet", "sites.json")]
                                 + [p(cfg, "a0_processed", "sites.json")]
                                 + sorted({p(cfg, "a0_emb", r["retriever"], f"{s_}__cal.npy") for r in cfg["runs"].values()
                                           for s_ in load_json(PROJECT_ROOT / r["emb"] / "sites.json")})
                                 + [PROJECT_ROOT / r["a2"] / "tables" / f for r in cfg["runs"].values() for f in ("rows.parquet", "sets.csv")]
                                 + [p(cfg, "p9_emb", r, "pool.npy") for r in sorted({v["retriever"] for v in cfg["runs"].values()})])
    save_json(meta, res / "meta.json")


if __name__ == "__main__":
    main()
