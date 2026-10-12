from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import PROJECT_ROOT, fingerprint, get_logger, load_config, p, run_metadata, save_json, stable_seed

_spec = importlib.util.spec_from_file_location("a8_report", Path(__file__).with_name("03_report.py"))
R3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R3)
twoway, fmt = R3.twoway, R3.fmt

SCORES = ["S_tau@0.01", "S_tau@inf", "mix@0.5", "mix@1"]
EST = [("main", "(ii)−(iii)"), ("off", "(iii)−abs"), ("leak", "(i)−(iii)"), ("matched", "N1−Zm")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--retrievers", nargs="*", default=None)
    ap.add_argument("--base", default=None, help="smoke: directory holding results_a1{,_full} and results_extra (default experiments/A8)")
    ap.add_argument("--out", default=None, help="smoke: output directory (default config results)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    B = cfg["a2"]["bootstrap_B"]
    base = Path(args.base) if args.base else PROJECT_ROOT / "experiments" / "A8"
    res = Path(args.out) if args.out else p(cfg, "results")
    (res / "tables").mkdir(parents=True, exist_ok=True)
    log = get_logger("A8.05", res / "logs" / "05_extras_report.log")
    RR = args.retrievers or cfg["retrievers"]
    d_main = {"nopos": "results_a1", "full": "results_a1_full"}
    d_ex = {k: f"results_extra/{k}" for k in ("a1_reach", "a1_full_reach", "a1_ip", "a1_ip_nomask")}
    rd = lambda d, r, f: base / d / r / "tables" / f
    inputs = []
    L = ["# A8 — open analysis questions: mask-reach subset, source-matched level (i′), 12-site bootstrap", "",
         "- Plan and pre-run forecasts: JOURNAL 2026-10-07 11:57. Values are Δd′ (or ΔAUROC where marked); mean over (KB, O) cells of the draw mean;"
         " [ ] = KB × O two-way bootstrap 95% over the 12 CQADupStack sites (B = " + str(B) + ").", ""]
    out1, out2, out3 = [], [], []

    L += ["## 1. Mask-reach subset: nopos vs unmasked", "",
          "ID = KB test queries whose own-forum top-10 changes under the mask (OOD = all O test queries, calibration = all KB calibration queries, as in A8).",
          "Paired difference = nopos − unmasked per draw → cell → mean, same 12-site bootstrap. Ceiling control: the reach subset has high s_KB, so the unmasked arm is closer to the AUROC clip"
          " (truncation biases the paired difference towards 0); the last columns restrict to cells with A_abs ≤ " + str(cfg["a2"]["ceiling"]) + " in both arms and show the clip shares.", ""]
    L += ["| model | reached KB test queries (per KB min–max) | with a masked doc (test) | A_abs all: nopos / unmasked | A_abs reach: nopos / unmasked | draws at clip @0.01, all: nopos / unmasked | at clip, reach: nopos / unmasked | cells A_abs ≤ ceiling in both arms: all / reach |", "|---|---|---|---|---|---|---|---|"]
    for r in RR:
        R_ = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "A8" / f"reach_{r}.parquet")
        inputs.append(PROJECT_ROOT / "data" / "processed" / "A8" / f"reach_{r}.parquet")
        t = R_[R_.role == "test"]
        per = t.groupby("site").changed.sum()
        aa, clp, ab = {}, {}, {}
        for lab, d in (("all nopos", d_main["nopos"]), ("all full", d_main["full"]), ("reach nopos", d_ex["a1_reach"]), ("reach full", d_ex["a1_full_reach"])):
            c = pd.read_parquet(rd(d, r, "contrasts.parquet"), columns=["score", "kb", "ood", "a_abs", "at_clip"]).query("score == 'S_tau@0.01'")
            aa[lab] = c.groupby(["kb", "ood"]).a_abs.mean().mean()
            clp[lab] = c.at_clip.mean()
            ab[lab] = c.groupby(["kb", "ood"]).a_abs.first()
        nb = {sc_: int(((ab[f"{sc_} nopos"] <= cfg["a2"]["ceiling"]) & (ab[f"{sc_} full"] <= cfg["a2"]["ceiling"])).sum()) for sc_ in ("all", "reach")}
        L.append(f"| {r} | {int(t.changed.sum())} ({int(per.min())}–{int(per.max())}) | {int(t.has_mask.sum())} | {aa['all nopos']:.4f} / {aa['all full']:.4f} | {aa['reach nopos']:.4f} / {aa['reach full']:.4f} | "
                 f"{clp['all nopos']:.3f} / {clp['all full']:.3f} | {clp['reach nopos']:.3f} / {clp['reach full']:.3f} | {nb['all']} / {nb['reach']} of {len(ab['all nopos'])} |")
    for metric, mlab in (("ddp", "Δd′"), ("dA", "ΔAUROC")):
        L += ["", f"### {mlab}: nopos ‖ unmasked ‖ paired nopos − unmasked", "",
              "| model | score | estimand | all queries | reach subset | reach: paired diff, A_abs ≤ ceiling in both arms | reach: draws at clip nopos / unmasked |", "|---|---|---|---|---|---|---|"]
        for r in RR:
            Cs = {}
            for lab, d in (("all nopos", d_main["nopos"]), ("all full", d_main["full"]), ("reach nopos", d_ex["a1_reach"]), ("reach full", d_ex["a1_full_reach"])):
                Cs[lab] = pd.read_parquet(rd(d, r, "contrasts.parquet"))
                inputs.append(rd(d, r, "contrasts.parquet"))
            for sc in SCORES:
                for est, elab in EST:
                    col = f"{est}_{metric}"
                    cells = []
                    for scope in ("all", "reach"):
                        a = Cs[f"{scope} nopos"].query("score == @sc")[["kb", "ood", "draw", col, "a_abs", "at_clip"]]
                        b = Cs[f"{scope} full"].query("score == @sc")[["kb", "ood", "draw", col, "a_abs", "at_clip"]]
                        m = a.merge(b, on=["kb", "ood", "draw"], suffixes=("_n", "_f"))
                        assert len(m) == len(a) == len(b)
                        m["diff"] = m[f"{col}_n"] - m[f"{col}_f"]
                        cl = m.groupby(["kb", "ood"])[[f"{col}_n", f"{col}_f", "diff", "a_abs_n", "a_abs_f"]].mean().reset_index()
                        seed = stable_seed(cfg["seed"], "A8x_reach", r, sc, est, metric, scope)
                        tn, tf, td = (twoway(cl, c_, B, seed) for c_ in (f"{col}_n", f"{col}_f", "diff"))
                        ceil = cfg["a2"]["ceiling"]
                        tc = twoway(cl[(cl.a_abs_n <= ceil) & (cl.a_abs_f <= ceil)], "diff", B, seed)
                        out1.append({"model": r, "score": sc, "estimand": est, "metric": metric, "scope": scope,
                                     "nopos": tn[0], "nopos_lo": tn[1], "nopos_hi": tn[2], "unmasked": tf[0],
                                     "diff": td[0], "diff_lo": td[1], "diff_hi": td[2], "n_cells": td[3],
                                     "diff_ceil": tc[0], "diff_ceil_lo": tc[1], "diff_ceil_hi": tc[2], "n_cells_ceil": tc[3],
                                     "at_clip_nopos": m.at_clip_n.mean(), "at_clip_unmasked": m.at_clip_f.mean()})
                        cells.append(f"{tn[0]:+.3f} ‖ {tf[0]:+.3f} ‖ {fmt(td)}")
                    cells.append(f"{fmt(tc)} ({tc[3]})" if tc[3] else "—")
                    cells.append(f"{m.at_clip_n.mean():.3f} / {m.at_clip_f.mean():.3f}")
                    L.append(f"| {r} | {sc} | {elab} | " + " | ".join(cells) + " |")
    pd.DataFrame(out1).to_csv(res / "tables" / "extras_reach.csv", index=False)

    L += ["", "## 2. Source-matched level (i′): O's twin A0 bank (2021 dump) in the 4th slot", ""]
    ipm = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "A8" / "ip_mask_counts.csv")
    inputs.append(PROJECT_ROOT / "data" / "processed" / "A8" / "ip_mask_counts.csv")
    L += ["Title mask on the twin bank (own title or a qrels positive's title, exact after normalisation): "
          + "; ".join(f"{r}: queries masked {int(g.queries_masked.sum())} of {int(g.queries.sum())}, test queries whose twin top-10 changes {int(g.queries_test_top10_changed.sum())}" for r, g in ipm.groupby("retriever", sort=False)), ""]
    chk = []
    for r in RR:
        for ex, mn in (("a1_ip", "nopos"), ("a1_ip_nomask", "nopos")):
            a = pd.read_parquet(rd(d_ex[ex], r, "rows.parquet"))
            b = pd.read_parquet(rd(d_main[mn], r, "rows.parquet"))
            a = a[a.role != "Op"].reset_index(drop=True)
            assert a.shape == b.shape and (a[["kb", "ood", "draw", "role", "slot", "family"]].astype(str).to_numpy() == b[["kb", "ood", "draw", "role", "slot", "family"]].astype(str).to_numpy()).all()
            da = pd.read_csv(rd(d_ex[ex], r, "draws.csv"))
            db = pd.read_csv(rd(d_main[mn], r, "draws.csv"))
            assert da[da.role != "Op"].reset_index(drop=True).astype(str).equals(db.astype(str)), (r, ex, "draws differ")
            dmax = float(np.nanmax(np.abs(a.auroc.to_numpy() - b.auroc.to_numpy())))
            assert dmax == 0.0, (r, ex, dmax)
            chk.append(f"{r} {ex}: max |ΔAUROC| = {dmax:g}")
    L += ["- Check: non-Op rows of the (i′) runs equal the A8 nopos rows (" + "; ".join(chk) + ").", ""]
    L += ["| model | score | (i)−(iii) | (i′)−(iii) masked | (i′)−(iii) unmasked | recovery (i) | recovery (i′) masked | recovery (i′) unmasked | (i′)−(i) Δd′ masked | (i′)−(i) ΔAUROC masked | (i′)−(i) Δd′ unmasked | Op−O mean S: ID / OOD (Z ID sd) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in RR:
        E0 = pd.read_csv(rd(d_main["nopos"], r, "estimands.csv"))
        Em = pd.read_csv(rd(d_ex["a1_ip"], r, "estimands.csv"))
        Eu = pd.read_csv(rd(d_ex["a1_ip_nomask"], r, "estimands.csv"))
        rows = pd.read_parquet(rd(d_ex["a1_ip"], r, "rows.parquet"))
        rows["score"] = rows.family + "@" + rows.param.astype(str)
        g_ = lambda E, sc, est, mt="ddp": E[(E.score == sc) & (E.estimand == est) & (E.metric == mt)].iloc[0]
        for sc in SCORES:
            q = {k_: g_(*v) for k_, v in {"leak": (E0, sc, "leak"), "lm": (Em, sc, "leakp"), "lu": (Eu, sc, "leakp"), "rec": (E0, sc, "recovery"),
                                          "recm": (Em, sc, "recoveryp"), "recu": (Eu, sc, "recoveryp"), "om": (Em, sc, "opmo"), "omA": (Em, sc, "opmo", "dA"), "ou": (Eu, sc, "opmo")}.items()}
            W = rows[rows.score == sc].pivot_table(index=["kb", "ood", "draw"], columns="role", values=["mu_id", "mu_ood", "sd_id"], aggfunc="first")
            zsd = (W[("sd_id", "Z1")] + W[("sd_id", "Z2")]) / 2
            sh = {s_: ((W[(f"mu_{s_}", "Op")] - W[(f"mu_{s_}", "O")]) / zsd).groupby(["kb", "ood"]).mean().mean() for s_ in ("id", "ood")}
            ff = lambda t: f"{t.point:+.3f} [{t.lo:+.3f}, {t.hi:+.3f}]"
            L.append(f"| {r} | {sc} | {ff(q['leak'])} | {ff(q['lm'])} | {ff(q['lu'])} | {q['rec'].point:.2f} [{q['rec'].lo:.2f}, {q['rec'].hi:.2f}] | "
                     f"{q['recm'].point:.2f} [{q['recm'].lo:.2f}, {q['recm'].hi:.2f}] | {q['recu'].point:.2f} [{q['recu'].lo:.2f}, {q['recu'].hi:.2f}] | {ff(q['om'])} | {ff(q['omA'])} | {ff(q['ou'])} | {sh['id']:+.3f} / {sh['ood']:+.3f} |")
            out2.append({"model": r, "score": sc, **{f"{k_}_{a_}": getattr(v, a_) for k_, v in q.items() for a_ in ("point", "lo", "hi")},
                         "op_minus_o_mu_id": sh["id"], "op_minus_o_mu_ood": sh["ood"]})
            inputs += [rd(d_ex["a1_ip"], r, "estimands.csv"), rd(d_ex["a1_ip_nomask"], r, "estimands.csv")]
    L += ["", "(i′)−(i) compares two source bundles, not only the dump year: text preprocessing (P6 whitespace folding), bank sampling, and mask rule / reach (id-based mask on 1,549 queries vs title mask) also differ (review 2026-10-07).",
          "", "Op − O mean S shift: (mean S with Op) − (mean S with O) on KB test (ID) and O test (OOD) queries; S = ID-ness, so a positive OOD shift means O's queries look more in-scope with the twin bank than with their own CQADupStack bank (less push).", ""]
    pd.DataFrame(out2).to_csv(res / "tables" / "extras_level_ip.csv", index=False)

    L += ["## 3. A1/03 bootstrap: 98-column multinomial (old) vs 12-site resampling (new)", ""]
    L += ["| model | mask | rows | max |Δ point| | median width ratio new/old (estimands) | IQR | median width ratio (γ regression) | zero-inclusion changed (estimands / γ) |", "|---|---|---|---|---|---|---|---|"]
    for r in RR:
        for mk, d in d_main.items():
            o = pd.read_csv(rd(d, r, "estimands_98col.csv"))
            n = pd.read_csv(rd(d, r, "estimands.csv"))
            inputs += [rd(d, r, "estimands_98col.csv"), rd(d, r, "estimands.csv"), rd(d, r, "regression_98col.csv"), rd(d, r, "regression.csv")]
            key = ["score", "estimand", "metric"]
            m = o.merge(n, on=key, suffixes=("_o", "_n"))
            assert len(m) == len(o) == len(n)
            dpt = float(np.nanmax(np.abs(m.point_o - m.point_n)))
            m = m[m.estimand != "recovery"].dropna(subset=["lo_o", "lo_n"])
            wr = (m.hi_n - m.lo_n) / (m.hi_o - m.lo_o)
            z = lambda lo, hi: (lo <= 0) & (hi >= 0)
            ch = int((z(m.lo_o, m.hi_o) != z(m.lo_n, m.hi_n)).sum())
            go = pd.read_csv(rd(d, r, "regression_98col.csv"))
            gn = pd.read_csv(rd(d, r, "regression.csv"))
            g = go.merge(gn, on=["score", "distance", "coef"], suffixes=("_o", "_n"))
            dpt = max(dpt, float(np.abs(g.point_o - g.point_n).max()))
            assert dpt < 1e-12, (r, mk, dpt, "points differ: stale or mislabelled *_98col backup")
            gw = (g.hi_n - g.lo_n) / (g.hi_o - g.lo_o)
            gch = int((z(g.lo_o, g.hi_o) != z(g.lo_n, g.hi_n)).sum())
            L.append(f"| {r} | {mk} | {len(m)} | {dpt:.1e} | {wr.median():.3f} | {wr.quantile(.25):.3f}–{wr.quantile(.75):.3f} | {gw.median():.3f} | {ch} / {gch} |")
            out3.append({"model": r, "mask": mk, "n_rows": len(m), "max_dpoint": dpt, "width_ratio_median": wr.median(),
                         "width_ratio_q25": wr.quantile(.25), "width_ratio_q75": wr.quantile(.75), "gamma_width_ratio_median": gw.median(),
                         "zero_change_estimands": ch, "zero_change_gamma": gch})
            if ch:
                cc = m[z(m.lo_o, m.hi_o) != z(m.lo_n, m.hi_n)]
                L.append("|  | rows whose zero inclusion changed | " + "; ".join(f"{a} {b} {c}: [{x:+.3f}, {y:+.3f}] → [{u:+.3f}, {v:+.3f}]" for a, b, c, x, y, u, v in
                                                                         zip(cc.score, cc.estimand, cc.metric, cc.lo_o, cc.hi_o, cc.lo_n, cc.hi_n)) + " | | | | | |")
    pd.DataFrame(out3).to_csv(res / "tables" / "extras_bootstrap.csv", index=False)
    (res / "A8_extras.md").write_text("\n".join(L) + "\n")
    meta = run_metadata(cfg, "A8.05_extras_report")
    meta["inputs"] = fingerprint(sorted(set(inputs)))
    meta["outputs"] = fingerprint([res / "A8_extras.md", res / "tables" / "extras_reach.csv", res / "tables" / "extras_level_ip.csv", res / "tables" / "extras_bootstrap.csv"])
    save_json(meta, res / "meta_05.json")
    log.info("A8_extras.md written")


if __name__ == "__main__":
    main()
