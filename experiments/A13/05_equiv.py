from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import PROJECT_ROOT, get_logger, load_config, p, run_metadata, save_json, stable_seed, fingerprint

spec = importlib.util.spec_from_file_location("a13_02", PROJECT_ROOT / "experiments" / "A13" / "02_report.py")
R2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R2)
REF = R2.REF
Q90 = (0.05, 0.95)


def readout(d, col, B, seed):
    m2, l2, h2, n = R2.twoway(d, col, B, seed, q=Q90)
    _, lk, hk, _ = R2.twoway(d, col, B, seed + 1, kb_only=True, q=Q90)
    return {"mean": m2, "lo90_2w": l2, "hi90_2w": h2, "lo90_kb": lk, "hi90_kb": hk, "n": n,
            "dmin_2w": max(abs(l2), abs(h2)), "dmin_kb": max(abs(lk), abs(hk)), "dmin": max(abs(l2), abs(h2), abs(lk), abs(hk))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--res", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = Path(args.res) if args.res else p(cfg, "results")
    log = get_logger("A13.05", res / "logs" / "05_equiv.log")
    B, ceil = cfg["metrics"]["bootstrap_B"], cfg["metrics"]["ceiling"]
    runs = [r for r in cfg["runs"] if not args.runs or r in args.runs]
    out = []
    for rn in runs:
        C = pd.read_parquet(res / "tables" / f"cellsG_{rn}.parquet")
        ref = C[C.score == REF]
        gainA, gainG = ref.dA.mean(), ref.G.mean()
        for sc in ("a2_site@inf", "a2_site@0.01"):
            d = C[C.score == sc]
            for col, unit, gain in (("dA_ref", "AUROC", gainA), ("G_ref", "G", gainG)):
                rd = readout(d, col, B, stable_seed(cfg["seed"], "A13_equiv", rn, sc, col))
                out.append({"run": rn, "claim": "E1 random sites ~ FLatS" if sc == "a2_site@inf" else "E1' (tau 0.01, not claimed)",
                            "contrast": f"{sc} − {REF}", "unit": unit, "ref_gain": gain, **rd})
        A2r = pd.read_parquet(PROJECT_ROOT / cfg["runs"][rn]["a2"] / "tables" / "rows.parquet",
                              columns=["kb", "ood", "m", "corpus", "partition", "family", "param", "auroc", "auroc_abs"])
        A2r = A2r[(A2r.m == 4) & (A2r.corpus == "se") & (A2r.family == "S_tau") & A2r.param.isin(["0.01", "inf"])]
        cell = A2r.groupby(["kb", "ood", "partition", "param"])[["auroc", "auroc_abs"]].mean().reset_index()
        cell["G"] = np.where(cell.auroc_abs <= ceil, (cell.auroc - cell.auroc_abs) / (1 - cell.auroc_abs), np.nan)
        for par in ("0.01", "inf"):
            site = cell[(cell.partition == "site") & (cell.param == par)].set_index(["kb", "ood"])
            for other in ("tag", "cross", "same"):
                oth = cell[(cell.partition == other) & (cell.param == par)].set_index(["kb", "ood"]).reindex(site.index)
                d = pd.DataFrame({"kb": site.index.get_level_values(0), "ood": site.index.get_level_values(1),
                                  "dA": (site.auroc - oth.auroc).to_numpy(), "dG": (site.G - oth.G).to_numpy()})
                for col, unit, gain in (("dA", "AUROC", (site.auroc - site.auroc_abs).mean()), ("dG", "G", site.G.mean())):
                    rd = readout(d, col, B, stable_seed(cfg["seed"], "A13_equiv", rn, other, par, col))
                    out.append({"run": rn, "claim": "E2 curation = clustering", "contrast": f"site − {other} @tau {par}", "unit": unit, "ref_gain": gain, **rd})
        log.info(f"{rn} done")
    D = pd.DataFrame(out)
    D["dmin_over_gain"] = D.dmin / D.ref_gain.abs()
    D.to_csv(res / "tables" / "equiv.csv", index=False)
    L = ["# A13 — equivalence readouts (margin not yet fixed; JOURNAL 2026-10-07 20:14)", "",
         "- δ_min = the smallest margin for which TOST (α = 0.05 per side) would declare equivalence = max |end| of the 90% interval; reported for the KB × O two-way and the KB bootstrap; 'δ_min both' = the larger. 'ratio' = δ_min both / the reference contrast's own gain over abs (E1: FLatS's gain; E2: the site partition's gain). Equivalence holds for a chosen margin δ iff δ ≥ δ_min both.",
         "- Margins were not fixed before the point estimates were seen (A2, A13); any margin chosen now is post hoc.", "",
         "| run | claim | contrast | unit | mean | 90% two-way | 90% KB | δ_min both | ref gain | ratio |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, x in D.iterrows():
        nd = 4 if x.unit == "AUROC" else 3
        L.append(f"| {x.run} | {x.claim} | {x.contrast} | {x.unit} | {x['mean']:+.{nd}f} | [{x.lo90_2w:+.{nd}f}, {x.hi90_2w:+.{nd}f}] | [{x.lo90_kb:+.{nd}f}, {x.hi90_kb:+.{nd}f}] | "
                 f"{x.dmin:.{nd}f} | {x.ref_gain:+.{nd}f} | {x.dmin_over_gain:.2f} |")
    (res / "A13_equivalence.md").write_text("\n".join(L) + "\n")
    meta = run_metadata(cfg, "A13.05_equiv")
    meta["inputs"] = fingerprint([res / "tables" / f"cellsG_{rn}.parquet" for rn in runs] + [PROJECT_ROOT / cfg["runs"][rn]["a2"] / "tables" / "rows.parquet" for rn in runs])
    save_json(meta, res / "meta_05.json")


if __name__ == "__main__":
    main()
