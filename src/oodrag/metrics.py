from __future__ import annotations

import numpy as np
from scipy.stats import beta, rankdata
from sklearn.metrics import average_precision_score


def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    r = rankdata(np.concatenate([neg, pos]))
    rp = r[len(neg):].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fpr_at_tpr95(neg: np.ndarray, pos: np.ndarray) -> float:
    pos = np.sort(np.asarray(pos, float))
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    k = int(np.floor(0.05 * len(pos)))
    t = pos[k]
    return float(np.mean(np.asarray(neg, float) >= t))


def auprc(neg: np.ndarray, pos: np.ndarray) -> float:
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    s = np.r_[np.asarray(neg, float), np.asarray(pos, float)]
    return float(average_precision_score(y, s))


def paired_winrate(s0: np.ndarray, s1: np.ndarray) -> float:
    s0, s1 = np.asarray(s0, float), np.asarray(s1, float)
    return float(np.mean(s1 > s0) + 0.5 * np.mean(s1 == s0))


def bootstrap_paired(s0, s1, fns: dict, B: int, seed: int) -> dict:
    s0, s1 = np.asarray(s0, float), np.asarray(s1, float)
    rng = np.random.default_rng(seed)
    n = len(s0)
    vals = {k: np.empty(B) for k in fns}
    for b in range(B):
        idx = rng.integers(0, n, n)
        for k, f in fns.items():
            vals[k][b] = f(s0[idx], s1[idx])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in vals.items()}


def bootstrap_unpaired(neg, pos, fns: dict, B: int, seed: int) -> dict:
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    rng = np.random.default_rng(seed)
    vals = {k: np.empty(B) for k in fns}
    for b in range(B):
        a = neg[rng.integers(0, len(neg), len(neg))]
        c = pos[rng.integers(0, len(pos), len(pos))]
        for k, f in fns.items():
            vals[k][b] = f(a, c)
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in vals.items()}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def ndcg_at_k(ranked, rel: dict, k: int = 10) -> float:
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    g = np.array([rel.get(int(d), 0) for d in ranked[:k]], float)
    ideal = np.sort(np.array([x for x in rel.values() if x > 0], float))[::-1][:k]
    return float((g * disc[:len(g)]).sum() / (ideal * disc[:len(ideal)]).sum())
