from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import fingerprint, load_config, load_json, p, run_metadata, save_json

TAUS = {"inf": None, "0.01": 0.01}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = p(cfg, "results")
    rows = []
    for r in cfg["retrievers"]:
        edir = p(cfg, "emb", r)
        Lm = np.asarray(np.load(edir / "topk.npy", mmap_mode="r")).mean(2)
        sites = load_json(edir / "sites.json")
        sidx = {s: i for i, s in enumerate(sites)}
        qi = pd.read_parquet(edir / "q_index.parquet")
        rows_of = {k: g.index.to_numpy() for k, g in qi.groupby(["site", "role"])}
        dr = pd.read_csv(res / r / "tables" / "draws.csv")
        zz = dr[dr.role.isin(["Z1", "Z2"])]
        for (kb, o, d, fx, zr), g in zz.groupby(["kb", "ood", "draw", "fixed", "z_relaxed"]):
            i_id, i_ood = rows_of[(kb, "test")], rows_of[(o, "test")]
            sk_id, sk_ood = Lm[i_id, sidx[kb]], Lm[i_ood, sidx[kb]]
            sd = sk_id.std()
            for z in g.slot:
                C = [sidx[x] for x in fx.split("+")] + [sidx[z]]
                for tn, t in TAUS.items():
                    def fn(A):
                        return A.mean(1) if t is None else t * (logsumexp(A / t, axis=1) - np.log(A.shape[1]))
                    c_id, c_ood = fn(Lm[np.ix_(i_id, C)]), fn(Lm[np.ix_(i_ood, C)])
                    sep_abs = (sk_id.mean() - sk_ood.mean()) / sd
                    c_gap = (c_id.mean() - c_ood.mean()) / sd
                    sr = c_id.std() / sd
                    co = np.corrcoef(sk_id, c_id)[0, 1]
                    sep_rel = (sep_abs - c_gap) / np.sqrt(max(1 + sr ** 2 - 2 * co * sr, 1e-12))
                    rows.append({"retriever": r, "kb": kb, "ood": o, "draw": d, "z": z, "z_relaxed": int(zr), "tau": tn,
                                 "sep_abs": sep_abs, "c_gap": c_gap, "sd_ratio": sr, "corr_id": co, "sep_rel": sep_rel,
                                 "pred": sep_rel - sep_abs})
    df = pd.DataFrame(rows)
    cols = ["sep_abs", "c_gap", "sd_ratio", "corr_id", "pred"]
    cell = df.groupby(["retriever", "tau", "kb", "ood"])[cols].mean().reset_index()
    cell_strict = df[df.z_relaxed == 0].groupby(["retriever", "tau", "kb", "ood"])[cols].mean().reset_index()
    pairs = pd.read_csv(res / cfg["retrievers"][0] / "tables" / "pairs.csv")[["kb", "ood", "tier"]]
    cell = cell.merge(pairs, on=["kb", "ood"])
    obs = []
    for r in cfg["retrievers"]:
        C = pd.read_parquet(res / r / "tables" / "contrasts.parquet", columns=["score", "kb", "ood", "draw", "off_ddp"])
        C = C[C.score.isin(["S_tau@inf", "S_tau@0.01"])].groupby(["score", "kb", "ood"]).off_ddp.mean().reset_index()
        C["tau"] = C.score.str.split("@").str[1]
        C["retriever"] = r
        obs.append(C[["retriever", "tau", "kb", "ood", "off_ddp"]])
    cell = cell.merge(pd.concat(obs), on=["retriever", "tau", "kb", "ood"], how="left")
    cell.to_csv(res / "offdiag_cells.csv", index=False)

    L = ["# A5 post hoc: the comparison term without a neighbour (fixed members + Z), added after the A5 results", "",
         "Why a tau-normalised rerun was not run: (iii) − abs for Qwen3 is already negative at tau = inf (no temperature) and the per-model similarity scales differ by ~1.5x only, so rescaling tau could only shift the curves sideways.",
         "Columns (ID-sd units of s_KB): sep_abs = ID−OOD gap of s_KB; c_gap = ID−OOD gap of the comparison term (> 0: ID queries are closer to the unrelated banks, so subtracting c removes separation); sd_ratio = SD_ID(c)/SD_ID(s_KB); corr_id = corr_ID(s_KB, c); pred = predicted change of separation (Gaussian approximation). Cell means (medians in brackets); off_ddp = observed (iii) − abs Δd′ (03).", ""]
    for tn in TAUS:
        L += [f"## tau = {tn}", "", "| retriever | subset | n cells | " + " | ".join(cols) + " | off_ddp | Spearman(pred, off_ddp) |", "|---" * (len(cols) + 5) + "|"]
        for r in cfg["retrievers"]:
            for lab, sub in (("all", cell), *(((f"tier {t}"), cell[cell.tier == t]) for t in ("near", "mid", "far")),
                             ("z strict", cell_strict.merge(cell[["retriever", "tau", "kb", "ood", "off_ddp", "tier"]], on=["retriever", "tau", "kb", "ood"]))):
                g = sub[(sub.retriever == r) & (sub.tau == tn)]
                rho = spearmanr(g.pred, g.off_ddp).statistic if len(g) > 3 else np.nan
                L.append(f"| {r} | {lab} | {len(g)} | " + " | ".join(f"{g[c].mean():+.3f} ({g[c].median():+.3f})" for c in cols)
                         + f" | {g.off_ddp.mean():+.3f} | {rho:+.2f} |")
        L.append("")
    (res / "A5_offdiag.md").write_text("\n".join(L) + "\n")
    meta = run_metadata(cfg, "A5.05_offdiag (post hoc)")
    meta["inputs"] = fingerprint([p(cfg, "emb", r, "topk.npy") for r in cfg["retrievers"]]
                                 + [res / r / "tables" / f for r in cfg["retrievers"] for f in ("draws.csv", "contrasts.parquet")])
    save_json(meta, res / "meta_05.json")


if __name__ == "__main__":
    main()
