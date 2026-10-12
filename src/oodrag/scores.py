from __future__ import annotations

import numpy as np
import torch
from scipy.special import logsumexp


def _masked(S):
    return np.where(np.isnan(S), -np.inf, S)


def logit_scores(S: np.ndarray, tau: float, corpus_mean: np.ndarray, k_gap: int, k_nqc: int, k_wig: int) -> dict:
    Sm = _masked(S)
    s1 = S[:, 0]
    lse = logsumexp(Sm / tau, axis=1)
    p = np.exp(Sm / tau - lse[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    out = {
        "S1_maxlogit": -s1,
        "S2_msp": -np.exp(s1 / tau - lse),
        "S3_energy": -tau * lse,
        "S4_entropy": ent,
        "S5_gap": -(s1 - S[:, k_gap - 1]),
        "S6_nqc": -np.nanstd(S[:, :k_nqc], axis=1) / np.abs(corpus_mean),
        "S7_wig": -(np.nanmean(S[:, :k_wig], axis=1) - corpus_mean),
    }
    return out


def profile_logratio(C: np.ndarray) -> np.ndarray:
    d = 1.0 - C
    dmin = np.maximum(np.nanmin(d, axis=1), 1e-12)
    return np.log(dmin) - np.log(np.nanvar(d, axis=1))


def principal_space(X: np.ndarray, D: int) -> np.ndarray:
    X = np.asarray(X, np.float64)
    w, V = np.linalg.eigh(X.T @ X)
    return V[:, np.argsort(w)[::-1][:D]]


def residual_norm(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    X = np.asarray(X, np.float64)
    return np.linalg.norm(X - (X @ P) @ P.T, axis=1)


def vim_score(S: np.ndarray, tau: float, r: np.ndarray, alpha: float) -> np.ndarray:
    lse = logsumexp(_masked(S) / tau, axis=1)
    v = alpha * r
    return v - np.logaddexp(lse, v)


def vim_alpha(S_fit: np.ndarray, tau: float, r_fit: np.ndarray) -> float:
    return float(np.sum(S_fit[:, 0] / tau) / np.sum(r_fit))


def local_residual(qn: np.ndarray, Dn: np.ndarray, rcond: float = 1e-6) -> np.ndarray:
    out = np.empty(len(qn))
    for i in range(len(qn)):
        Dm = Dn[i][~np.isnan(Dn[i]).any(axis=1)].astype(np.float64)
        q = qn[i].astype(np.float64)
        if len(Dm) == 0:
            out[i] = np.nan
            continue
        G = Dm @ Dm.T
        c = Dm @ q
        proj_sq = c @ np.linalg.pinv(G, rcond=rcond, hermitian=True) @ c
        out[i] = np.sqrt(max(q @ q - proj_sq, 0.0))
    return out


def sq_maha(X: np.ndarray, mu: np.ndarray, Prec: np.ndarray) -> np.ndarray:
    Z = np.asarray(X, np.float64) - mu
    return ((Z @ Prec) * Z).sum(1)


def gaussian_fit(X: np.ndarray, rcond: float = 1e-10):
    X = np.asarray(X, np.float64)
    mu = X.mean(0)
    Sig = np.cov(X, rowvar=False, bias=True)
    return mu, np.linalg.pinv(Sig, rcond=rcond, hermitian=True)


def local_rmd(qn: np.ndarray, Dn: np.ndarray, mu_D: np.ndarray, Prec_D: np.ndarray) -> np.ndarray:
    mu_loc = np.nanmean(Dn.astype(np.float64), axis=1)
    q = np.asarray(qn, np.float64)
    Zl = q - mu_loc
    md_loc = np.einsum("ij,jk,ik->i", Zl, Prec_D, Zl)
    return md_loc - sq_maha(q, mu_D, Prec_D)


def knn_distance(qn: np.ndarray, bankn: np.ndarray, k: int, device="cuda") -> np.ndarray:
    B = torch.from_numpy(np.ascontiguousarray(bankn, np.float32)).to(device)
    out = []
    for i in range(0, len(qn), 2048):
        Q = torch.from_numpy(np.ascontiguousarray(qn[i:i + 2048], np.float32)).to(device)
        sims = Q @ B.T
        out.append((1.0 - sims.topk(k, dim=1).values[:, -1]).double().cpu().numpy())
    return np.concatenate(out)


def empirical_pvalue(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(ref, float))
    ref = ref[np.isfinite(ref)]
    x = np.asarray(x, float)
    n_ge = len(ref) - np.searchsorted(ref, x, side="left")
    pv = (1.0 + n_ge) / (1.0 + len(ref))
    return np.where(np.isfinite(x), pv, np.nan)


def fisher_combine(cols: list[np.ndarray], refs: list[np.ndarray]) -> np.ndarray:
    return -2.0 * sum(np.log(empirical_pvalue(c, r)) for c, r in zip(cols, refs))


def head_scores(maxsims: list[torch.Tensor], lex: list[float]) -> dict:
    if not maxsims:
        return {"S11_sparse": np.nan, "S12_doc_mincov": np.nan, "S13_conj_gap": np.nan}
    M = torch.stack(maxsims).double()
    c_doc = M.mean(1).max().item()
    c_union = M.max(0).values.mean().item()
    return {
        "S11_sparse": -max(lex),
        "S12_doc_mincov": -M.min(1).values.max().item(),
        "S13_conj_gap": c_union - c_doc,
    }
