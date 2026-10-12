from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "experiments" / "A16"
RS = ["bge-m3", "contriever-msmarco", "qwen3-emb-0.6b"]
SC = ["S_tau@0.01", "S_tau@inf", "mix@0.5"]


def rdir(r, kind=""):
    if r == "bge-m3":
        return A / ("results" + (f"_{kind}" if kind else ""))
    return A / ("results_models" + (f"_{kind}" if kind else "")) / r


def ci(row):
    return f"{row.point:+.3f} [{row.lo:+.3f}, {row.hi:+.3f}]"


def main():
    E = {r: pd.read_csv(rdir(r) / "tables" / "estimands.csv") for r in RS}
    Rg = {r: pd.read_csv(rdir(r) / "tables" / "regression.csv") for r in RS}
    St = {r: pd.read_csv(rdir(r) / "tables" / "strata.csv") for r in RS}
    En = {r: pd.read_csv(rdir(r, "nbcat") / "tables" / "estimands.csv") for r in RS}
    T = pd.read_csv(A / "results" / "a10" / "tables" / "A_terms.csv")
    L = ["# A16 — model generality on the Amazon test bed (bge-m3 / contriever-msmarco / qwen3-emb-0.6b)", "",
         "- Same build, rules and seeds for all retrievers (experiments/A16/PLAN.md; JOURNAL 2026-10-08). Values: mean over (KB, O) cells, [ ] two-way 95%.", ""]
    signs = []
    for sc in SC:
        L += [f"## {sc}", "", "| retriever | (ii)−(iii) Δd′ | (iii)−abs | (i)−(iii) | N1−Zm | ΔAUROC (ii)−(iii) | γ_O rank | γ_KB rank | near / mid / far (ii)−(iii) | (ii)−(iii) A1 rule |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in RS:
            e = E[r][E[r].score == sc].set_index(["estimand", "metric"])
            g = Rg[r][(Rg[r].score == sc) & (Rg[r].distance == "rank")].set_index("coef")
            s = St[r][St[r].score == sc].set_index("stratum")
            en = En[r][(En[r].score == sc) & (En[r].metric == "ddp")].set_index("estimand")
            tiers = " / ".join(f"{s.loc[f'tier {t}'].point:+.3f}" for t in ("near", "mid", "far"))
            L.append(f"| {r} | {ci(e.loc[('main', 'ddp')])} | {ci(e.loc[('off', 'ddp')])} | {ci(e.loc[('leak', 'ddp')])} | {ci(e.loc[('matched', 'ddp')])} | "
                     f"{ci(e.loc[('main', 'dA')])} | {ci(g.loc['gamma_O'])} | {ci(g.loc['gamma_KB'])} | {tiers} | {ci(en.loc['main'])} |")
            for est in ("main", "off", "matched"):
                signs.append({"score": sc, "estimand": est, "retriever": r, "sign": int(e.loc[(est, "ddp")].point > 0),
                              "ci_excl0": bool(e.loc[(est, "ddp")].lo > 0 or e.loc[(est, "ddp")].hi < 0)})
        L.append("")
    L += ["## Moment accounting (A10; Δd′_AUC = parts + R_shape)", "",
          "| retriever | contrast | Δd′ | T_var / C_var | T_push / C_idcost | C_oodpush | R_shape |", "|---|---|---|---|---|---|---|"]
    want = ["A1 (ii)−(iii) @0.01", "A1 (ii)−(iii) @inf", "A1 (iii)−abs @0.01", "A1 (iii)−abs @inf", "A2 MS pooled mix vs abs @0.5"]
    for r in RS:
        t = T[T.run == f"a16_{r}"].set_index("contrast")
        for c in want:
            if c not in t.index:
                continue
            x = t.loc[c]
            two = pd.notna(x.get("C_oodpush"))
            v1, v2 = (x.C_var, x.C_idcost) if two else (x.T_var, x.T_push)
            oo = f"{x.C_oodpush:+.3f}" if two else "—"
            L.append(f"| {r} | {c} | {x.ddp:+.3f} [{x.ddp_lo:+.3f}, {x.ddp_hi:+.3f}] | {v1:+.3f} | {v2:+.3f} | {oo} | {x.R_shape:+.3f} |")
    S = pd.DataFrame(signs)
    agg = S.groupby(["score", "estimand"]).agg(n_pos=("sign", "sum"), n_ci=("ci_excl0", "sum")).reset_index()
    L += ["", "## Sign agreement across the three retrievers (Δd′ point > 0; intervals excluding 0)", "",
          "| score | estimand | positive | interval excludes 0 |", "|---|---|---|---|"]
    L += [f"| {a.score} | {a.estimand} | {a.n_pos}/3 | {a.n_ci}/3 |" for a in agg.itertuples()]
    (A / "results" / "A16_models.md").write_text("\n".join(L) + "\n")
    print("written", A / "results" / "A16_models.md")


if __name__ == "__main__":
    main()
