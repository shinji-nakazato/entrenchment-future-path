import json
import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import (RESULTS, WU1D_CSV, WU2D_CSV, WU_LSCALE_1D, WU_LSCALE_2D,
                    WU_GRIDSIDE_1D, WU_GRIDSIDE_2D, RHO_WILSON, N_WORKERS)
from gp_utils import SPECS, grid, posterior, observations, V_q_uniform

OUT = RESULTS
SEED = 20261102
K_PER_A = 30
N_MC = 50_000


def kdist(n):
    pa = np.array([0.25, 0.5, 0.25])
    pk = np.array([1.0])
    for _ in range(n):
        pk = np.convolve(pk, pa)
    return np.arange(-n, n + 1), pk


def wilson():
    w = pd.read_csv(RESULTS / 'indicator_wilson_pertrial.csv',
                    usecols=['H', 'A', 'info', 'dmu', 'Sigma', 'V_q'])
    rho2 = RHO_WILSON ** 2
    rows = []
    for a in range(1, 7):
        sub = w[(w.H == 6) & (w.A == a)]
        dmu, Sig, Vq = sub.dmu.values, sub.Sigma.values, sub.V_q.values
        ks, pk = kdist(a - 1)
        pe = np.zeros(len(sub))
        for k, p in zip(ks, pk):
            c = 1.0 + k
            pe += p * norm.cdf(c * dmu / np.sqrt(c * c * Sig + 2.0 * a * rho2))
        pm = norm.cdf(dmu / np.sqrt(Sig + 2.0 * (a - 1) * Vq + 2.0 * a * rho2))
        d = np.abs(pe - pm)
        for lab, m in [('all', np.ones(len(sub), bool)),
                       ('equal', (sub['info'] == 'equal').values),
                       ('unequal', (sub['info'] != 'equal').values)]:
            if not m.any():
                continue
            rows.append(dict(task='Wilson', spec='flat', info=lab, D=2, A=a, n_trials=int(m.sum()),
                             mean_abs=d[m].mean(), p95_abs=np.quantile(d[m], 0.95),
                             max_abs=d[m].max()))
    return pd.DataFrame(rows)


def wu_candidates(path, ndim):
    df = pd.read_csv(path)
    zkey = 'zcollectScaled' if ndim == 2 else 'ycollectScaled'
    cands = []
    for i, row in df.iterrows():
        try:
            sh = json.loads(row['searchHistory'])
        except Exception:
            continue
        for r in range(len(sh[zkey])):
            T = len(sh[zkey][r])
            for t in range(1, T):
                cands.append((i, r, t, T - t))
    return df, cands


def mc_one(args):
    sh, ndim, side, ls, spec, t, A, seed = args
    p = SPECS[spec]
    rng = np.random.default_rng(seed)
    r = sh['r']
    zs = observations(sh['sh'], ndim, r, spec)
    xs = sh['sh']['xcollect'][r]
    ys = sh['sh']['ycollect'][r] if ndim == 2 else None
    X = (np.array(xs[:t], float).reshape(-1, 1) if ndim == 1
         else np.stack([xs[:t], ys[:t]], axis=1).astype(float))
    mu, cov = posterior(X, np.array(zs[:t], float), grid(ndim, side), ls, p['m0'], p['sf2'], p['noise2'])
    rho2 = p['noise2']
    order = np.argsort(mu)[::-1]
    ia, ib = int(order[0]), int(order[1])
    Sig = float(cov[ia, ia] + cov[ib, ib] - 2 * cov[ia, ib])
    dmu = float(mu[ia] - mu[ib])
    D = len(mu)
    Vq = V_q_uniform(mu, cov)
    sig2 = Sig + 2.0 * (A - 1) * Vq + 2.0 * A * rho2
    w, U = np.linalg.eigh(cov)
    L = U * np.sqrt(np.clip(w, 0, None))
    F = mu[:, None] + L @ rng.standard_normal((D, N_MC))
    cols = np.arange(N_MC)
    Y = F[ia] - F[ib]
    for _ in range(1, A):
        Y += F[rng.integers(0, D, N_MC), cols] - F[rng.integers(0, D, N_MC), cols]
    Y += rng.normal(0.0, np.sqrt(2.0 * A * rho2), N_MC)
    pe = float(np.mean(Y > 0))
    pm = float(norm.cdf(dmu / np.sqrt(sig2)))
    z = np.sort((Y - dmu) / np.sqrt(sig2))
    ks = float(np.max(np.abs(np.arange(1, N_MC + 1) / N_MC - norm.cdf(z))))
    return dict(dmu=dmu, Sigma=Sig, V_q=Vq, kappa=dmu / np.sqrt(sig2), p_exact=pe, p_probit=pm,
                abs_diff=abs(pe - pm), mc_se=np.sqrt(pe * (1 - pe) / N_MC), d_ks=ks,
                var_ratio=float(np.var(Y)) / sig2)


def wu(specs, a_all=None):
    rng = np.random.default_rng(SEED)
    rows = []
    for path, ndim, side, ls, label in [(WU1D_CSV, 1, WU_GRIDSIDE_1D, WU_LSCALE_1D, 'Wu1D'),
                                        (WU2D_CSV, 2, WU_GRIDSIDE_2D, WU_LSCALE_2D, 'Wu2D')]:
        df, cands = wu_candidates(path, ndim)
        C = pd.DataFrame(cands, columns=['row', 'r', 't', 'A'])
        pick = []
        for a, g in C[C.A >= 2].groupby('A'):
            if a_all is not None:
                if a in a_all:
                    pick.append(g)
                continue
            pick.append(g.sample(n=min(K_PER_A, len(g)), random_state=int(rng.integers(1 << 31))))
        P = pd.concat(pick)
        cache = {}
        for spec in specs:
            tasks, meta = [], []
            for _, q in P.iterrows():
                key = int(q.row)
                if key not in cache:
                    cache[key] = json.loads(df.loc[key, 'searchHistory'])
                tasks.append((dict(sh=cache[key], r=int(q.r)), ndim, side, ls, spec, int(q.t), int(q.A),
                              int(rng.integers(1 << 31))))
                meta.append(dict(task=label, spec=spec, row=int(q.row), round=int(q.r), t=int(q.t), A=int(q.A)))
            with Pool(N_WORKERS) as pool:
                res = pool.map(mc_one, tasks, chunksize=4)
            rows += [{**m, **r} for m, r in zip(meta, res)]
            print(label, spec, 'done', flush=True)
    return pd.DataFrame(rows)


def main_a2():
    R = wu(['wu', 'priorvar1'], a_all=(2,))
    R.to_csv(OUT / 'probit_error_wu_A2_all.csv', index=False)
    S = (R.groupby(['task', 'spec', 'A'])
         .agg(n_trials=('abs_diff', 'size'), mean_abs=('abs_diff', 'mean'),
              p95_abs=('abs_diff', lambda v: np.quantile(v, 0.95)), p99_abs=('abs_diff', lambda v: np.quantile(v, 0.99)),
              max_abs=('abs_diff', 'max'), mc_se_max=('mc_se', 'max'), d_ks_max=('d_ks', 'max'))
         .reset_index())
    txt = '\n'.join(['', f'[Wu, A = 2, every trial, Monte Carlo {N_MC:,} draws per trial]',
                     S.to_string(index=False, float_format=lambda v: f'{v:.4f}')])
    with open(OUT / 'probit_error_horizons.txt', 'a', encoding='utf-8') as f:
        f.write('\n' + txt + '\n')
    print(txt)


def main():
    W = wilson()
    R = wu(['wu', 'priorvar1'])
    R.to_csv(OUT / 'probit_error_wu_pertrial.csv', index=False)
    S = (R.groupby(['task', 'spec', 'A'])
         .agg(n_trials=('abs_diff', 'size'), mean_abs=('abs_diff', 'mean'),
              p95_abs=('abs_diff', lambda v: np.quantile(v, 0.95)), max_abs=('abs_diff', 'max'),
              mc_se_max=('mc_se', 'max'), d_ks_max=('d_ks', 'max'), var_ratio=('var_ratio', 'mean'))
         .reset_index())
    T = (R.groupby(['task', 'spec'])
         .agg(n_trials=('abs_diff', 'size'), A_min=('A', 'min'), A_max=('A', 'max'),
              mean_abs=('abs_diff', 'mean'), max_abs=('abs_diff', 'max'), d_ks_max=('d_ks', 'max'))
         .reset_index())
    WA = W[W['info'] == 'all']
    lines = ['Moment-matched probit vs exact future-path choice probability (tau = 1, no bias, q = U_D, gamma = 1)', '',
             '[Wilson, all free choices of H = 6 games (closed form, every trial)]',
             W.to_string(index=False, float_format=lambda v: f'{v:.4f}'), '',
             f'  Wilson overall A = 2-6: mean {WA[WA.A >= 2].mean_abs.mean():.4f}, max {WA[WA.A >= 2].max_abs.max():.4f}', '',
             f'[Wu, Monte Carlo, {K_PER_A} random trials per A, {N_MC:,} draws per trial; pairwise statistic greedy vs second-ranked]',
             T.to_string(index=False, float_format=lambda v: f'{v:.4f}'), '',
             '[Wu by A]', S.to_string(index=False, float_format=lambda v: f'{v:.4f}')]
    txt = '\n'.join(lines)
    (OUT / 'probit_error_horizons.txt').write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'a2':
        main_a2()
    else:
        main()
        main_a2()
