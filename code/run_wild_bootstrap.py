import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import RESULTS

SEED = 20240625
B    = 999


def main():
    d = pd.read_csv(RESULTS / 'all_pertrial.csv')
    d['dmu_std'] = d['dmu'] / np.sqrt(d['Sigma'].clip(lower=1e-8))
    y = d['non_greedy'].to_numpy(float)
    X = np.column_stack([np.ones(len(d)), d['log_A'], d['log_D'], d['dmu_std']])
    names = ['Intercept', 'log_A', 'log_D', 'dmu_std']
    clusters = d['subject_uid'].to_numpy()

    res = sm.Logit(y, X).fit(disp=0)
    beta = res.params
    p = res.predict(X)

    w = p * (1.0 - p)
    A = X.T @ (X * w[:, None])
    Ainv = np.linalg.inv(A)
    g = X * (y - p)[:, None]
    uniq, inv = np.unique(clusters, return_inverse=True)
    G = np.zeros((len(uniq), X.shape[1]))
    np.add.at(G, inv, g)
    Gn, N, K = len(uniq), len(d), X.shape[1]

    meat = G.T @ G
    corr = (Gn / (Gn - 1)) * ((N - 1) / (N - K))
    V = corr * (Ainv @ meat @ Ainv)
    se = np.sqrt(np.diag(V))
    t_obs = beta / se

    rng = np.random.default_rng(SEED)
    GA = G @ Ainv.T
    signs = rng.choice([-1.0, 1.0], size=(B, Gn))
    deltas = signs @ GA
    deltas *= np.sqrt(corr)
    tboot = deltas / se
    pvals = (1 + np.sum(np.abs(tboot) >= np.abs(t_obs), axis=0)) / (B + 1)
    ci_lo = beta + np.percentile(deltas, 2.5, axis=0)
    ci_hi = beta + np.percentile(deltas, 97.5, axis=0)

    sizes = pd.Series(clusters).value_counts().to_numpy()
    G_eff = sizes.sum() ** 2 / (sizes ** 2).sum()
    by_ds = d.groupby('dataset')['subject_uid'].nunique()

    lines = []
    lines.append(f'Pooled logit wild cluster bootstrap (Rademacher, B={B}, seed={SEED})')
    lines.append(f'n = {N:,};  clusters G = {Gn}  '
                 f'(Wilson {by_ds.get("Wilson",0)} + Wu1D {by_ds.get("Wu1D",0)}'
                 f' + Wu2D {by_ds.get("Wu2D",0)})')
    lines.append(f'cluster sizes: min {sizes.min()}, median {int(np.median(sizes))}, '
                 f'max {sizes.max()};  Herfindahl effective #clusters G_eff = {G_eff:.1f}')
    lines.append('')
    hdr = f'{"coef":<10}{"beta":>9}{"clustSE":>9}{"z":>8}{"WCB p":>9}{"WCB 95% CI":>22}'
    lines.append(hdr)
    lines.append('-' * len(hdr))
    for j, nm in enumerate(names):
        lines.append(f'{nm:<10}{beta[j]:>9.3f}{se[j]:>9.3f}{t_obs[j]:>8.2f}'
                     f'{pvals[j]:>9.3f}   [{ci_lo[j]:+.3f}, {ci_hi[j]:+.3f}]')
    out = '\n'.join(lines)
    (RESULTS / 'wild_bootstrap.txt').write_text(out, encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
