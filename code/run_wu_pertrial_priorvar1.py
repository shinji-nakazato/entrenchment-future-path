import json
import numpy as np
import pandas as pd
from scipy.stats import norm

from config import (WU1D_CSV, WU2D_CSV, RESULTS,
                    RHO_WU_PRIORVAR1, GAMMA,
                    WU_LSCALE_1D, WU_LSCALE_2D,
                    WU_GRIDSIDE_1D, WU_GRIDSIDE_2D)
from gp_utils import S_factor, gp_posterior, V_q_uniform, kappa_q


def _build_grid(ndim: int, gridside: int) -> np.ndarray:
    if ndim == 1:
        return np.arange(gridside).reshape(-1, 1).astype(float)
    xs = np.arange(gridside)
    gx, gy = np.meshgrid(xs, xs, indexing='xy')
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(float)


def process_wu(path, ndim: int, gridside: int, lscale: float, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    X_grid = _build_grid(ndim, gridside)
    rho2 = RHO_WU_PRIORVAR1 ** 2
    rows = []

    for _, row in df.iterrows():
        try:
            sh = json.loads(row['searchHistory'])
        except Exception:
            continue
        x_collect = sh['xcollect']
        if ndim == 2:
            y_collect = sh['ycollect']
            z_collect = sh['zcollectScaled']
        else:
            y_collect = None
            z_collect = sh['ycollectScaled']

        for round_idx in range(len(z_collect)):
            zs   = z_collect[round_idx]
            xs_r = x_collect[round_idx]
            ys_r = sh['ycollect'][round_idx] if ndim == 2 else None
            T = len(zs)

            for t in range(1, T):
                if ndim == 1:
                    Xtr = np.array(xs_r[:t]).reshape(-1, 1).astype(float)
                else:
                    Xtr = np.stack([xs_r[:t], ys_r[:t]], axis=1).astype(float)
                ytr = np.array(zs[:t]).astype(float) - 25.0
                try:
                    mu, cov = gp_posterior(Xtr, ytr, X_grid, lscale, rho2)
                except np.linalg.LinAlgError:
                    continue
                mu = mu + 25.0

                star = int(np.argmax(mu))
                if ndim == 1:
                    chosen = int(xs_r[t])
                else:
                    chosen = int(ys_r[t]) * gridside + int(xs_r[t])
                non_greedy = int(chosen != star)

                order = np.argsort(mu)[::-1]
                alt   = int(order[1]) if order[0] == star else int(order[0])
                dmu   = float(mu[star] - mu[alt])
                Sigma = float(cov[star, star] + cov[alt, alt] - 2 * cov[star, alt])
                if Sigma <= 0:
                    Sigma = 1e-8

                V_q = V_q_uniform(mu, cov)
                A = T - t
                S = S_factor(GAMMA, A)
                k_v2 = kappa_q(dmu, Sigma, V_q, S, rho2)
                if np.isnan(k_v2):
                    continue
                k_v1 = kappa_q(dmu, Sigma, 0.0, S, rho2)
                k_naive = dmu / np.sqrt(Sigma)

                rows.append(dict(
                    dataset=label,
                    subject=row.get('id', np.nan),
                    scenario=row['scenario'],
                    kernel=row['kernel'],
                    horizon=row['horizon'],
                    round=round_idx,
                    t=t + 1, T=T, A=A,
                    dmu=dmu, Sigma=Sigma, V_q=V_q,
                    mu_star=float(mu[star]), mu_alt=float(mu[alt]),
                    kappa_v2=k_v2, kappa_v1=k_v1, kappa_naive=k_naive,
                    p_nongreedy_v2=1.0 - norm.cdf(k_v2),
                    p_nongreedy_v1=1.0 - norm.cdf(k_v1),
                    non_greedy=non_greedy,
                ))
    return pd.DataFrame(rows)


def main():
    R1 = process_wu(WU1D_CSV, ndim=1, gridside=WU_GRIDSIDE_1D, lscale=WU_LSCALE_1D, label='Wu1D')
    R1.to_csv(RESULTS / 'wu1d_pertrial_priorvar1.csv', index=False)
    R2 = process_wu(WU2D_CSV, ndim=2, gridside=WU_GRIDSIDE_2D, lscale=WU_LSCALE_2D, label='Wu2D')
    R2.to_csv(RESULTS / 'wu2d_pertrial_priorvar1.csv', index=False)
    for lab, R in (('Wu1D', R1), ('Wu2D', R2)):
        print(f'{lab} (prior-variance-1 specification): n_trials={len(R)}, subjects={R.subject.nunique()}, '
              f'non-greedy rate={R.non_greedy.mean():.3f}, mean V_q={R.V_q.mean():.3f}')


if __name__ == '__main__':
    main()
