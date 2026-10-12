from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from oodrag.common import fingerprint, get_logger, load_config, p, run_metadata, save_json, set_determinism, stable_seed

set_determinism()
import torch

DEV = "cuda"


def topk_sims(Q, D, k, cosine, chunk=4096):
    Qg = torch.as_tensor(Q, device=DEV)
    Dg = torch.as_tensor(D, device=DEV)
    if cosine:
        Qg = Qg / Qg.norm(dim=1, keepdim=True)
        Dg = Dg / Dg.norm(dim=1, keepdim=True)
    out = np.empty((len(Q), k), np.float64)
    for i in range(0, len(Q), chunk):
        out[i:i + chunk] = (Qg[i:i + chunk] @ Dg.T).topk(k, dim=1).values.double().cpu().numpy()
    return out


def auroc(neg, pos):
    r = rankdata(np.concatenate([neg, pos]))
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def li_stats(T, tau):
    sm = np.exp(T - T.max(1, keepdims=True))
    P = sm / sm.sum(1, keepdims=True)
    return {"MSS": -T[:, 0], "KNN": -T[:, -1], "AvgKNN": -T.mean(1), "Entropy": -(P * np.log(P)).sum(1),
            "Energy": -tau * np.log(np.exp(T / tau).sum(1))}


def boot_auc(S_id, S_ood, Wid, Wood):
    s = torch.as_tensor(np.concatenate([S_id, S_ood]), device=DEV, dtype=torch.float64)
    lab = torch.cat([torch.zeros(len(S_id), device=DEV), torch.ones(len(S_ood), device=DEV)])
    order = torch.argsort(s, stable=True)
    W = torch.cat([Wid, Wood], dim=1)[:, order]
    lab = lab[order]
    cid = torch.cumsum(W * (1 - lab), dim=1) - W * (1 - lab)
    num = (cid * W * lab).sum(1)
    tot = Wid.sum(1) * Wood.sum(1)
    return (1 - num / tot).cpu().numpy()


def write_summary(R, V, C, cfg, res, retrievers):
    r0 = retrievers[0]
    ood = cfg["queries"]["ood"]
    L = ["# A6 — Li et al. 2024 bridge condition (KB = MedRAG Textbooks, ID = MedQA-US)", ""]
    L += ["## (A) Reproduction of Li Table 1(a) (inner product, k = 32, 300 vs 300, mean of 10 runs): ours / Li", ""]
    for (r, col), g in R.groupby(["retriever", "query_text"], sort=False):
        L += [f"### {r}, query text = {col}", "", "| dataset | " + " | ".join(cfg["reproduction"]["stats"]) + " | max |Δ| |", "|---" * 7 + "|"]
        for ds in ood:
            q = g[g.dataset == ds].set_index("stat")
            if q.empty:
                continue
            L.append(f"| {ds} | " + " | ".join(f"{q.loc[s_, 'auroc']:.4f} / {q.loc[s_, 'li']:.4f}" for s_ in cfg["reproduction"]["stats"])
                     + f" | {(q.auroc - q.li).abs().max():.4f} |")
        L.append("")
    key_scores = [("S_tau", "0.01"), ("S_tau", "inf"), ("mix", "0.5"), ("mix", "1"), ("mixIP", "0.5"), ("mixIP", "1")]
    sets = list(cfg["relative"]["sets"])
    for r in retrievers:
        g = V[V.retriever == r]
        L += [f"## (B) Relative judgment, {r} (cosine top-10; Δd′ vs abs [query-bootstrap 95%]; mixIP vs absIP)", ""]
        a = g[g.set == "abs"].set_index("dataset").auroc
        ai = g[g.set == "absIP"].set_index("dataset").auroc
        L.append("Absolute AUROC (cosine top-10 / inner-product AvgKNN32): " + "; ".join(f"{d} {a.get(d, np.nan):.3f} / {ai.get(d, np.nan):.3f}" for d in ood))
        L.append("")
        for fam, par in key_scores:
            L += [f"### {fam}@{par}", "", "| dataset | " + " | ".join(sets) + " |", "|---" * (len(sets) + 1) + "|"]
            for ds in ood:
                q = g[(g.dataset == ds) & (g.family == fam) & (g.param == par)].set_index("set")
                if q.empty:
                    continue
                L.append(f"| {ds} | " + " | ".join(f"{q.loc[s_, 'ddp']:+.2f} [{q.loc[s_, 'ddp_lo']:+.2f}, {q.loc[s_, 'ddp_hi']:+.2f}]" for s_ in sets) + " |")
            L.append("")
        L += ["### Operating point: OOD acceptance at the threshold accepting 95% of calibration MedQA (abs → set)", "",
              "| dataset | abs | " + " | ".join(f"{s_} S_τ@0.01" for s_ in sets) + " |", "|---" * (len(sets) + 2) + "|"]
        for ds in ood:
            q = g[g.dataset == ds]
            if q.empty:
                continue
            ab = q[q.set == "abs"].op_ood_acc.iloc[0]
            L.append(f"| {ds} | {ab:.3f} | " + " | ".join(f"{q[(q.set == s_) & (q.family == 'S_tau') & (q.param == '0.01')].op_ood_acc.iloc[0]:.3f}" for s_ in sets) + " |")
        L.append("")
        L += ["### ID / OOD mean shift of S vs abs (ID-sd units), S_τ@0.01", "", "| dataset | " + " | ".join(f"{s_} ID / OOD" for s_ in sets) + " |", "|---" * (len(sets) + 1) + "|"]
        for ds in ood:
            q = g[(g.dataset == ds) & (g.family == "S_tau") & (g.param == "0.01")].set_index("set")
            if q.empty:
                continue
            L.append(f"| {ds} | " + " | ".join(f"{q.loc[s_, 'id_shift']:+.2f} / {q.loc[s_, 'ood_shift']:+.2f}" for s_ in sets) + " |")
        L.append("")
        for design in [d_ for d_ in cfg["relative"]["rq3_designs"] if not d_.startswith("_")]:
            sa, sb = cfg["relative"]["rq3_designs"][design]
            c = C[(C.retriever == r) & (C.design == design)]
            L += [f"## (C) RQ3 analogue {design}, {r}: Δd′({sa}) − Δd′({sb}) per OOD dataset [95%] (ΔAUROC in brackets below); * = clipped cell (excluded from DiD)", "",
                  "| dataset | neighbour | abs AUROC | " + " | ".join(f"{f_}@{p_}" for f_, p_ in key_scores) + " |", "|---" * (len(key_scores) + 3) + "|"]
            for ds in ood + ["DiD(research-general)"]:
                q = c[c.dataset == ds]
                if q.empty:
                    continue
                cells = []
                for f_, p_ in key_scores:
                    x = q[(q.family == f_) & (q.param == p_)]
                    if x.empty:
                        cells.append("—")
                        continue
                    x = x.iloc[0]
                    cells.append(f"{x.contrast_ddp:+.2f} [{x.lo:+.2f}, {x.hi:+.2f}]" + ("*" if x.clipped else "")
                                 + (f" (ΔA {x.contrast_auroc:+.3f})" if ds != "DiD(research-general)" else ""))
                ab = q.auroc_abs.iloc[0] if "auroc_abs" in q and pd.notna(q.auroc_abs.iloc[0]) else np.nan
                L.append(f"| {ds} | {q.neighbour.iloc[0]} | {ab:.4f} | " + " | ".join(cells) + " |")
            L.append("")
    (res / "A6_summary.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    ap.add_argument("--emb-smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = Path(args.out) if args.out else p(cfg, "results")
    (res / "tables").mkdir(parents=True, exist_ok=True)
    log = get_logger("A6.03", res / "logs" / "03_eval.log")
    t0 = time.time()
    Q = pd.read_parquet(p(cfg, "processed", "queries.parquet"))
    rp, rl = cfg["reproduction"], cfg["relative"]
    comp = cfg["corpora"]["comparison"]
    clip = rl["dprime_clip"]
    dp = lambda a: np.sqrt(2) * norm.ppf(np.clip(a, clip, 1 - clip))
    rep_rows, rel_rows, rq3_rows = [], [], []
    L = ["# A6 — Li et al. 2024 bridge condition (Textbooks KB)", ""]
    retrievers = [r for r in cfg["retrievers"] if not r.startswith("_")]
    for r in retrievers:
        edir = p(cfg, "emb", "_smoke" if args.emb_smoke else "", r)
        Ck = {n: np.load(edir / f"corpus_{n}.npy") for n in [cfg["corpora"]["kb"]] + comp}
        nq = len(np.load(edir / "queries_question.npy", mmap_mode="r"))
        Qs = Q.iloc[:nq].reset_index(drop=True)
        for col in ("question", "question_options"):
            E = np.load(edir / f"queries_{col}.npy")
            T = topk_sims(E, Ck["textbooks"], rp["k"], cosine=False)
            st = li_stats(T, rp["energy_tau"])
            ik = np.flatnonzero(Qs.dataset == "medqa")
            for ds in cfg["queries"]["ood"]:
                ok = np.flatnonzero(Qs.dataset == ds)
                if len(ok) < rp["n_per_side"] or len(ik) < rp["n_per_side"]:
                    continue
                for name, v in st.items():
                    aucs = []
                    for run in range(rp["runs"]):
                        rng = np.random.default_rng(stable_seed(cfg["seed"], "A6_repro", ds, run))
                        a = rng.choice(ik, rp["n_per_side"], replace=False)
                        b = rng.choice(ok, rp["n_per_side"], replace=False)
                        aucs.append(auroc(v[a], v[b]))
                    rep_rows.append({"retriever": r, "query_text": col, "dataset": ds, "stat": name, "auroc": float(np.mean(aucs)),
                                     "auroc_sd": float(np.std(aucs)), "li": rp["li_table1a"][ds][name]})
        log.info(f"{r}: reproduction done ({time.time() - t0:.0f}s)")
        E = np.load(edir / "queries_question.npy")
        Tc = {n: topk_sims(E, X, rl["k"], cosine=True) for n, X in Ck.items()}
        Ti = {n: topk_sims(E, X, rp["k"], cosine=False) for n, X in Ck.items()}
        s_kb, s_kb_ip = Tc["textbooks"].mean(1), Ti["textbooks"].mean(1)
        scores = {("abs", "-", "-"): s_kb, ("absIP", "-", "-"): s_kb_ip}
        for sn, members in rl["sets"].items():
            G = np.column_stack([Tc[c].mean(1) for c in members])
            m = len(members)
            for t in rl["tau_grid"]:
                scores[(sn, "S_tau", str(t))] = s_kb - (G.mean(1) if t == "inf" else t * (np.logaddexp.reduce(G / t, axis=1) - np.log(m)))
            Uc = np.concatenate([Tc[c] for c in members], axis=1)
            pc = -np.sort(-Uc, axis=1)[:, : rl["k"]].mean(1)
            Ui = np.concatenate([Ti[c] for c in members], axis=1)
            pi = -np.sort(-Ui, axis=1)[:, : rp["k"]].mean(1)
            for a in rl["alpha_grid"]:
                scores[(sn, "mix", str(a))] = s_kb - a * pc
                scores[(sn, "mixIP", str(a))] = s_kb_ip - a * pi
        keys = list(scores)
        S = np.stack([scores[k_] for k_ in keys])
        i_id = np.flatnonzero((Qs.dataset == "medqa") & (Qs.split == "test"))
        i_cal = np.flatnonzero((Qs.dataset == "medqa") & (Qs.split == "cal"))
        B = rl["bootstrap_B"]
        rng = np.random.default_rng(stable_seed(cfg["seed"], "A6_boot_id"))
        Wid = torch.as_tensor(rng.multinomial(len(i_id), [1 / len(i_id)] * len(i_id), size=B), device=DEV, dtype=torch.float64)
        AUC = {}
        for ds in cfg["queries"]["ood"]:
            i_ood = np.flatnonzero((Qs.dataset == ds) & ~Qs.source_in_comparison)
            if len(i_ood) == 0:
                continue
            rng = np.random.default_rng(stable_seed(cfg["seed"], "A6_boot_ood", ds))
            Wood = torch.as_tensor(rng.multinomial(len(i_ood), [1 / len(i_ood)] * len(i_ood), size=B), device=DEV, dtype=torch.float64)
            ab = {"abs": None, "absIP": None}
            AUC[ds] = {}
            for j, k_ in enumerate(keys):
                v = S[j]
                A = auroc(-v[i_id], -v[i_ood])
                bs = boot_auc(v[i_id], v[i_ood], Wid, Wood)
                t_cal = np.quantile(v[i_cal], 0.05, method="lower")
                AUC[ds][k_] = (A, bs)
                base = ("absIP", "-", "-") if k_[1] == "mixIP" or k_[0] == "absIP" else ("abs", "-", "-")
                vb = scores[base]
                rel_rows.append({"retriever": r, "dataset": ds, "set": k_[0], "family": k_[1], "param": k_[2], "auroc": A,
                                 "op_ood_acc": float(np.mean(v[i_ood] >= t_cal)), "op_id_acc": float(np.mean(v[i_id] >= t_cal)),
                                 "id_shift": float((v[i_id].mean() - vb[i_id].mean()) / vb[i_id].std()),
                                 "ood_shift": float((v[i_ood].mean() - vb[i_ood].mean()) / vb[i_id].std()), "n_ood": len(i_ood)})
            for k_ in keys:
                base = ("absIP", "-", "-") if k_[1] == "mixIP" or k_[0] == "absIP" else ("abs", "-", "-")
                A, bs = AUC[ds][k_]
                A0, b0 = AUC[ds][base]
                rel_rows[-len(keys) + keys.index(k_)].update({"auroc_base": A0, "ddp": float(dp(A) - dp(A0)),
                                                              "ddp_lo": float(np.quantile(dp(bs) - dp(b0), 0.025)),
                                                              "ddp_hi": float(np.quantile(dp(bs) - dp(b0), 0.975))})
        nb = rl["neighbour"]
        for design, (sa, sb) in ((k_, v_) for k_, v_ in rl["rq3_designs"].items() if not k_.startswith("_")):
            fam_params = sorted({(k_[1], k_[2]) for k_ in keys if k_[0] == sa})
            for fam, par in fam_params:
                contr = {}
                for ds in AUC:
                    base = ("absIP", "-", "-") if fam == "mixIP" else ("abs", "-", "-")
                    (a1, b1), (a2, b2), (a0, b0) = AUC[ds][(sa, fam, par)], AUC[ds][(sb, fam, par)], AUC[ds][base]
                    clipped = bool(max(a0, a1, a2) >= 1 - clip)
                    c, cb = dp(a1) - dp(a2), dp(b1) - dp(b2)
                    contr[ds] = (c, cb, clipped)
                    rq3_rows.append({"retriever": r, "design": design, "family": fam, "param": par, "dataset": ds, "neighbour": nb.get(ds) or "none",
                                     "contrast_ddp": float(c), "lo": float(np.quantile(cb, 0.025)), "hi": float(np.quantile(cb, 0.975)),
                                     "contrast_auroc": float(a1 - a2), "auroc_lo": float(np.quantile(b1 - b2, 0.025)),
                                     "auroc_hi": float(np.quantile(b1 - b2, 0.975)), "clipped": clipped, "auroc_abs": float(a0)})
                res_ds = [d for d in contr if nb.get(d) == "pubmed" and not contr[d][2]]
                gen_ds = [d for d in contr if nb.get(d) == "wikipedia" and not contr[d][2]]
                if res_ds and gen_ds:
                    did = np.mean([contr[d][0] for d in res_ds]) - np.mean([contr[d][0] for d in gen_ds])
                    didb = np.mean([contr[d][1] for d in res_ds], axis=0) - np.mean([contr[d][1] for d in gen_ds], axis=0)
                    rq3_rows.append({"retriever": r, "design": design, "family": fam, "param": par, "dataset": "DiD(research-general)",
                                     "neighbour": f"research: {'+'.join(res_ds)}; general: {'+'.join(gen_ds)}",
                                     "contrast_ddp": float(did), "lo": float(np.quantile(didb, 0.025)), "hi": float(np.quantile(didb, 0.975)),
                                     "clipped": False})
        log.info(f"{r}: relative done ({time.time() - t0:.0f}s)")
    R = pd.DataFrame(rep_rows)
    V = pd.DataFrame(rel_rows)
    C = pd.DataFrame(rq3_rows)
    R.to_csv(res / "tables" / "repro.csv", index=False)
    V.to_csv(res / "tables" / "relative.csv", index=False)
    C.to_csv(res / "tables" / "rq3.csv", index=False)
    write_summary(R, V, C, cfg, res, retrievers)
    meta = run_metadata(cfg, "A6.03_eval")
    ed = lambda r_: p(cfg, "emb", "_smoke" if args.emb_smoke else "", r_)
    meta["inputs"] = fingerprint([p(cfg, "processed", "queries.parquet")]
                                 + [f_ for r_ in retrievers for f_ in sorted(ed(r_).glob("*.npy")) + [ed(r_) / "meta.json"] if f_.exists()])
    meta["minutes"] = (time.time() - t0) / 60
    save_json(meta, res / "meta_03.json")
    log.info(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
