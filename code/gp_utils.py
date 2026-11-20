import numpy as np

from config import RHO_WU_PRIORVAR1


def S_factor(gamma: float, A: int) -> float:
    if abs(gamma - 1.0) < 1e-12:
        return float(A)
    return (1.0 - gamma ** (2 * A)) / (1.0 - gamma ** 2)


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, lscale: float) -> np.ndarray:
    d2 = (np.sum(X1 ** 2, 1)[:, None]
          + np.sum(X2 ** 2, 1)[None, :]
          - 2 * X1.dot(X2.T))
    return np.exp(-d2 / (2 * lscale ** 2))


def gp_posterior(X_train, y_train, X_grid, lscale, rho2):
    n = len(X_train)
    K   = rbf_kernel(X_train, X_train, lscale) + rho2 * np.eye(n)
    Ks  = rbf_kernel(X_grid,  X_train, lscale)
    Kss = rbf_kernel(X_grid,  X_grid,  lscale)
    L   = np.linalg.cholesky(K + 1e-8 * np.eye(n))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    v   = np.linalg.solve(L, Ks.T)
    mu  = Ks.dot(alpha)
    cov = Kss - v.T.dot(v)
    return mu, cov


def V_q_uniform(mu: np.ndarray, cov: np.ndarray) -> float:
    D = len(mu)
    V_mu  = float(((mu - mu.mean()) ** 2).mean())
    V_sig = float(np.trace(cov) / D - cov.sum() / (D * D))
    return V_mu + V_sig


def kappa_q(dmu: float, Sigma: float, V_q: float, S: float, rho2: float) -> float:
    denom = Sigma + 2.0 * (S - 1.0) * V_q + 2.0 * S * rho2
    if denom <= 0:
        return float('nan')
    return dmu / np.sqrt(denom)


SPECS = {
    'priorvar1': dict(m0=25.0, sf2=1.0, noise2=RHO_WU_PRIORVAR1 ** 2),
    'priorvar300': dict(m0=50.0, sf2=300.0, noise2=1.0),
    'wu': dict(m0=0.0, sf2=1.0, noise2=1e-4, raw=True),
}


def observations(sh, ndim, r, spec):
    if SPECS[spec].get('raw'):
        key = 'zcollect' if ndim == 2 else 'ycollect'
        return (np.array(sh[key][r], float) - 50.0) / 100.0
    key = 'zcollectScaled' if ndim == 2 else 'ycollectScaled'
    return np.array(sh[key][r], float)


def grid(ndim, side):
    if ndim == 1:
        return np.arange(side).reshape(-1, 1).astype(float)
    xs = np.arange(side)
    gx, gy = np.meshgrid(xs, xs, indexing='xy')
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(float)


def posterior(X, y, Xg, ls, m0, sf2, noise2):
    n = len(X)
    K = sf2 * rbf_kernel(X, X, ls) + noise2 * np.eye(n) + 1e-8 * np.eye(n)
    Ks = sf2 * rbf_kernel(Xg, X, ls)
    Kss = sf2 * rbf_kernel(Xg, Xg, ls)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y - m0))
    v = np.linalg.solve(L, Ks.T)
    return m0 + Ks @ alpha, Kss - v.T @ v
