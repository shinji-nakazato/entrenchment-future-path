from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import digamma

SEED = 20261028


def vb_bms(L, n_iter=500, tol=1e-10):
    n, K = L.shape
    a0 = np.ones(K)
    a = a0.copy()
    for _ in range(n_iter):
        u = L + digamma(a) - digamma(a.sum())
        u -= u.max(axis=1, keepdims=True)
        g = np.exp(u)
        g /= g.sum(axis=1, keepdims=True)
        a_new = a0 + g.sum(axis=0)
        if np.max(np.abs(a_new - a)) < tol:
            a = a_new
            break
        a = a_new
    rng = np.random.default_rng(SEED)
    r = rng.dirichlet(a, size=200_000)
    xp = np.bincount(r.argmax(axis=1), minlength=K) / len(r)
    return a, a / a.sum(), xp


def bor_pxp(L, a, xp):
    n, K = L.shape
    F0 = np.sum(np.log(np.mean(np.exp(L - L.max(axis=1, keepdims=True)), axis=1)) + L.max(axis=1))
    u = L + digamma(a) - digamma(a.sum())
    g = np.exp(u - u.max(axis=1, keepdims=True))
    g /= g.sum(axis=1, keepdims=True)
    from scipy.special import gammaln
    a0 = np.ones(K)
    Elogr = digamma(a) - digamma(a.sum())
    F1 = (np.sum(g * (L + Elogr)) - np.sum(g * np.log(g + 1e-300))
          + gammaln(a0.sum()) - gammaln(a0).sum() + np.sum((a0 - 1) * Elogr)
          - (gammaln(a.sum()) - gammaln(a).sum() + np.sum((a - 1) * Elogr)))
    bor = 1.0 / (1.0 + np.exp(F1 - F0))
    return bor, xp * (1 - bor) + bor / K
