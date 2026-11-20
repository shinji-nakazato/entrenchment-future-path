import sys

import numpy as np
import pandas as pd

from wilson_choice import *
from wilson_choice import OUT, SEED


def run(models, R, D, y, folds, label):
    n = len(y)
    obs_eff = R.non_greedy[R.H == 6].mean() - R.non_greedy[R.H == 1].mean()
    rows, cells, params = [], [], []
    gl = D['greedy_left'] == 1
    for name, kind, fn, x0 in models:
        th, ll = fit(fn, x0, D, y)
        k = len(x0)
        pl = fn(th, D)
        png = np.where(gl, 1.0 - pl, pl)
        heff = png[R.H.to_numpy() == 6].mean() - png[R.H.to_numpy() == 1].mean()
        cv = 0.0
        for f in range(5):
            tr, te = folds != f, folds == f
            th_f, _ = fit(fn, th, subset(D, tr), y[tr])
            cv += -nll(fn(th_f, subset(D, te)), y[te])
        rows.append(dict(set=label, model=name, kind=kind, k=k, LL=ll, AIC=2 * k - 2 * ll,
                         BIC=k * np.log(n) - 2 * ll, horizon_effect=heff, cv_test_LL=cv))
        params.append(dict(model=name, theta=' '.join(f'{v:.4f}' for v in th)))
        tmp = R[['H', 'info', 'dI', 'non_greedy']].copy()
        tmp['p_ng'] = png
        tmp['p_more_inf'] = np.where(R.dI.to_numpy() == 1, pl, np.where(R.dI.to_numpy() == -1, 1 - pl, np.nan))
        for (h, inf), g in tmp.groupby(['H', 'info']):
            cells.append(dict(model=name, H=h, info=inf, n=len(g), pred_ng=g.p_ng.mean(),
                              pred_more_inf=g.p_more_inf.mean() if inf == 'unequal' else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(cells), pd.DataFrame(params), obs_eff


def observed_cells(R):
    out = []
    for (h, inf), g in R.groupby(['H', 'info']):
        mi = np.nan
        if inf == 'unequal':
            mi = np.mean(np.where(g.dI == 1, g.left, 1 - g.left))
        out.append(dict(model='OBSERVED', H=h, info=inf, n=len(g), pred_ng=g.non_greedy.mean(),
                        pred_more_inf=mi))
    return pd.DataFrame(out)


def main():
    R = build_first_free()
    D = as_dict(R)
    y_ng = R['non_greedy'].to_numpy(float)
    y_left = R['left'].to_numpy(float)

    subj = R['subjectID'].to_numpy()
    rng = np.random.default_rng(SEED)
    uniq = np.unique(subj)
    fold = {s: i % 5 for i, s in enumerate(rng.permutation(uniq))}
    folds = np.array([fold[s] for s in subj])

    T0, C0, P0, obs = run(REPLICATE, R, D, y_left, folds, 'horizon-free and single-mechanism models without side bias (Table H.1, top panel)')
    T1, C1, P1, _ = run(NEW, R, D, y_left, folds, 'competing models with side bias B, no lapse (Table H.1, middle panel)')
    T2, C2, P2, _ = run(LINK, R, D, y_left, folds, 'link diagnostics, probit vs logistic (Table H.1, bottom panel)')
    T3, C3, P3, _ = run(LAPSE, R, D, y_left, folds, 'with lapse component e, P = e/2 + (1-e) P_model (Table 5)')
    T = pd.concat([T0, T1, T2, T3], ignore_index=True)
    C = pd.concat([observed_cells(R), C0, C1, C2, C3], ignore_index=True)
    P = pd.concat([P0, P1, P2, P3], ignore_index=True)

    T.to_csv(OUT / 'wilson_models.csv', index=False)
    C.to_csv(OUT / 'wilson_models_cells.csv', index=False)
    P.to_csv(OUT / 'wilson_models_params.csv', index=False)

    lines = [f'Wilson first free choice (n={len(R):,}, {len(uniq)} subjects)',
             f'Observed horizon effect P(ng|H=6)-P(ng|H=1) = {obs:+.3f}',
             'Likelihood is on left/right choice; for models without B it equals the non-greedy likelihood.',
             '']
    with pd.option_context('display.width', 250, 'display.max_colwidth', 60,
                           'display.float_format', lambda v: f'{v:.2f}'):
        for s, g in T.groupby('set', sort=False):
            g = g.copy()
            g['dAIC_vs_best'] = g.AIC - T.AIC.min()
            g['horizon_effect'] = g['horizon_effect'].map(lambda v: f'{v:+.3f}')
            lines += [f'[{s}]', g[['model', 'kind', 'k', 'LL', 'AIC', 'BIC', 'dAIC_vs_best',
                                   'horizon_effect', 'cv_test_LL']].to_string(index=False), '']
        lines += ['[cells: observed vs predicted P(non-greedy), and P(choose more informative) in unequal games]']
        Cw = C.copy()
        Cw['cell'] = 'H' + Cw.H.astype(str) + '-' + Cw['info']
        piv = Cw.pivot_table(index='model', columns='cell', values='pred_ng', sort=False)
        piv2 = Cw[Cw['info'] == 'unequal'].pivot_table(index='model', columns='cell',
                                                       values='pred_more_inf', sort=False)
        piv2.columns = [c + ' P(more inf)' for c in piv2.columns]
        lines.append(pd.concat([piv, piv2], axis=1).to_string(float_format=lambda v: f'{v:.3f}'))
        lines += ['', '[parameters (raw scale; exp/logit transforms as in code)]', P.to_string(index=False)]
    out = '\n'.join(lines)
    (OUT / 'wilson_models.txt').write_text(out, encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
