import sys
import warnings
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import ttest_1samp, wilcoxon

warnings.filterwarnings('ignore')
import wilson_choice as m1
from wilson_choice import build_first_free, as_dict, fit, nll, subset

from config import RESULTS

from config import N_WORKERS

OUT = RESULTS

KEEP = ['softmax (per-H tau)', 'future-path, q=U_D', 'future-path, calib. q',
        'C1  Wilson, A_H + sigma_H + B', 'F1  future-path calib. q + B', 'F2u future-path q=U_D + A_H + B',
        'C1+e Wilson, A_H + sigma_H + B + lapse', "C1'+e Wilson, A_H + sigma_HxI + B + lapse",
        'F1+e future-path calib. q + B + lapse', 'F2u+e future-path q=U_D + A_H + B + lapse',
        'F0+e future-path lambda=0 + A_H + B + lapse']


def all_models():
    return m1.REPLICATE + m1.NEW + m1.LAPSE


def cell_labels(R):
    unequal = R.dI != 0
    greedy_less = np.where(unequal, ((R.dI == 1) & (R.greedy_left == 1)) | ((R.dI == -1) & (R.greedy_left == 0)), False)
    cell = np.where(~unequal, 'equal prior observations',
                    np.where(greedy_less, 'unequal, non-greedy to more-observed arm', 'unequal, non-greedy to less-observed arm'))
    return pd.Series(cell, index=R.index)


def wilson_subject_fit(args):
    sid, h, d = args
    y = d['left']
    def f(th):
        A, ls, B = th
        p = expit((d['dR'] + A * d['dI'] + B) / np.exp(ls))
        return nll(p, y)
    best = None
    for st in ([0.0, 2.0, 0.0], [5.0, 2.5, 0.0], [-5.0, 1.5, 0.0]):
        r = minimize(f, st, method='L-BFGS-B', bounds=[(-60, 60), (-3, 5), (-30, 30)])
        if best is None or r.fun < best.fun:
            best = r
    A, ls, B = best.x
    return dict(subjectID=sid, H=h, A=A, sigma=np.exp(ls), B=B, n=len(y))


def main():
    R = build_first_free()
    D = as_dict(R)
    y = R['left'].to_numpy(float)
    R['cell'] = cell_labels(R)
    R['ng_less'] = ((R.non_greedy == 1) & (R.cell == 'unequal, non-greedy to less-observed arm')).astype(int)
    lines = ['Information condition of the Wilson first free choice (n=19,200)', '']

    obs = R.groupby(['cell', 'H']).agg(n=('non_greedy', 'size'), p_ng=('non_greedy', 'mean')).unstack('H')
    lines += ['[observed P(non-greedy) by cell and horizon]', obs.to_string(float_format=lambda v: f'{v:.3f}')]
    eff = R.groupby(['cell']).apply(lambda g: g[g.H == 6].non_greedy.mean() - g[g.H == 1].non_greedy.mean())
    lines += ['  horizon effect (H6 - H1): ' + ', '.join(f'{k}: {v:+.3f}' for k, v in eff.items())]
    ue = R[R.dI != 0]
    for h in (1, 6):
        g = ue[ue.H == h]
        more_inf = np.where(g.dI == 1, g.left, 1 - g.left)
        lines.append(f'  H={h}: P(choose less-observed arm | unequal) = {more_inf.mean():.3f} (n={len(g):,})')
    lines.append('')

    tasks = []
    for (sid, h), g in R.groupby(['subjectID', 'H']):
        tasks.append((sid, h, dict(dR=g.dR.to_numpy(float), dI=g.dI.to_numpy(float), left=g.left.to_numpy(float))))
    with Pool(N_WORKERS) as pool:
        S = pd.DataFrame(pool.map(wilson_subject_fit, tasks))
    S.to_csv(OUT / 'info_condition_subject_params.csv', index=False)
    P = S.pivot(index='subjectID', columns='H', values=['A', 'sigma'])
    lines.append('[per-subject Wilson et al. (2014) model, P(left) = logistic((dR + A dI + B) / sigma), fit per subject x H]')
    for v in ('A', 'sigma'):
        a1, a6 = P[(v, 1)], P[(v, 6)]
        d = a6 - a1
        lines.append(f'  {v:5s}: median H1 = {a1.median():.2f}, H6 = {a6.median():.2f}; mean H1 = {a1.mean():.2f}, '
                     f'H6 = {a6.mean():.2f}; H6 - H1 median = {d.median():+.2f}, t = {ttest_1samp(d, 0).statistic:+.2f}, '
                     f'Wilcoxon p = {wilcoxon(d).pvalue:.2g}, share > 0 = {(d > 0).mean():.2f}')
    lines.append(f'  share of subjects with A > 0: H1 = {(P[("A", 1)] > 0).mean():.2f}, H6 = {(P[("A", 6)] > 0).mean():.2f}')
    lines.append('')

    lookup = {n: (n, k, f, x0) for n, k, f, x0 in all_models()}
    rows, ll_rows = [], []
    gl = D['greedy_left'] == 1
    for name in KEEP:
        _, kind, fn, x0 = lookup[name]
        th, ll = fit(fn, x0, D, y)
        pl = fn(th, D)
        png = np.where(gl, 1 - pl, pl)
        p_obs = np.clip(pl, 1e-9, 1 - 1e-9)
        lli = y * np.log(p_obs) + (1 - y) * np.log(1 - p_obs)
        tmp = R[['cell', 'H', 'non_greedy']].assign(p_ng=png, ll=lli)
        for (c, h), g in tmp.groupby(['cell', 'H']):
            rows.append(dict(model=name, cell=c, H=h, pred=g.p_ng.mean(), obs=g.non_greedy.mean(), n=len(g)))
            ll_rows.append(dict(model=name, cell=c, H=h, LL=g.ll.sum()))
    C = pd.DataFrame(rows)
    C.to_csv(OUT / 'info_condition_cells.csv', index=False)
    piv = C.pivot_table(index='model', columns=['cell', 'H'], values='pred', sort=False)
    obsrow = C.drop_duplicates(['cell', 'H']).pivot_table(columns=['cell', 'H'], values='obs')
    obsrow.index = ['OBSERVED']
    T = pd.concat([obsrow, piv])
    lines += ['[P(non-greedy) by cell x H: observed and model-implied (pooled MLE)]',
              T.to_string(float_format=lambda v: f'{v:.3f}'), '']
    L = pd.DataFrame(ll_rows).pivot_table(index='model', columns=['cell', 'H'], values='LL', sort=False)
    L['total'] = L.sum(axis=1)
    best = L.drop(columns='total').max(axis=0)
    lines += ['[log-likelihood by cell x H (left/right likelihood)]', L.to_string(float_format=lambda v: f'{v:.1f}'), '',
              '[log-likelihood deficit vs best model in each cell]',
              (L.drop(columns='total') - best).to_string(float_format=lambda v: f'{v:.1f}'), '']
    dev = C.assign(err=C.pred - C.obs).pivot_table(index='model', columns=['cell', 'H'], values='err', sort=False)
    lines += ['[prediction error pred - obs]', dev.to_string(float_format=lambda v: f'{v:+.3f}'), '']

    out = '\n'.join(lines)
    (OUT / 'info_condition.txt').write_text(out, encoding='utf-8')
    print(out)



def equal_only():
    from scipy.stats import norm
    R = build_first_free()
    R = R[R.dI == 0].reset_index(drop=True)
    D = as_dict(R)
    y = R['left'].to_numpy(float)
    W = lambda lam: np.sqrt(D['Sigma'] + 2.0 * (D['A'] - 1.0) * lam * D['V_q'] + 2.0 * D['A'] * m1.RHO2)
    lap = lambda e, p: e / 2 + (1 - e) * p
    models = [
        ('softmax tau_H + B + lapse', lambda th, D: lap(expit(th[3]), expit((D['dR'] + th[2]) / m1.hsel(D, np.exp(th[0]), np.exp(th[1])))), [1.0, 2.3, 0.0, -1.4]),
        ('probit tau_H + B + lapse', lambda th, D: lap(expit(th[3]), norm.cdf((D['dR'] + th[2]) / m1.hsel(D, np.exp(th[0]), np.exp(th[1])))), [1.5, 2.8, 0.0, -1.4]),
        ('FP q=U_D + B + lapse (no free horizon parameter)', lambda th, D: lap(expit(th[2]), norm.cdf((D['dR'] + th[1]) / (np.exp(th[0]) * np.sqrt(D['Sigma'] + 2.0 * (D['A'] - 1.0) * D['V_q'] + 2.0 * D['A'] * m1.RHO2)))), [-1.0, 0.0, -1.4]),
        ('FP lambda=0 + B + lapse (no V_q)', lambda th, D: lap(expit(th[2]), norm.cdf((D['dR'] + th[1]) / (np.exp(th[0]) * np.sqrt(D['Sigma'] + 2.0 * D['A'] * m1.RHO2)))), [-1.0, 0.0, -1.4]),
        ('FP calib. lambda + B + lapse', lambda th, D: lap(expit(th[3]), norm.cdf((D['dR'] + th[2]) / (np.exp(th[0]) * np.sqrt(D['Sigma'] + 2.0 * (D['A'] - 1.0) * expit(th[1]) * D['V_q'] + 2.0 * D['A'] * m1.RHO2)))), [-1.0, 0.0, 0.0, -1.4]),
        ('softmax tau_H + B (no lapse)', lambda th, D: expit((D['dR'] + th[2]) / m1.hsel(D, np.exp(th[0]), np.exp(th[1]))), [2.0, 2.5, 0.0]),
        ('FP q=U_D + B (no lapse)', lambda th, D: norm.cdf((D['dR'] + th[1]) / (np.exp(th[0]) * np.sqrt(D['Sigma'] + 2.0 * (D['A'] - 1.0) * D['V_q'] + 2.0 * D['A'] * m1.RHO2))), [0.0, 0.0]),
    ]
    subj = R.subjectID.to_numpy()
    rng = np.random.default_rng(20240625)
    uniq = np.unique(subj)
    fold = {s: i % 5 for i, s in enumerate(rng.permutation(uniq))}
    folds = np.array([fold[s] for s in subj])
    gl = D['greedy_left'] == 1
    rows = []
    for name, fn, x0 in models:
        th, ll = fit(fn, x0, D, y)
        k = len(x0)
        pl = fn(th, D)
        png = np.where(gl, 1 - pl, pl)
        cv = 0.0
        for f in range(5):
            tr, te = folds != f, folds == f
            th_f, _ = fit(fn, th, subset(D, tr), y[tr])
            cv += -nll(fn(th_f, subset(D, te)), y[te])
        rows.append(dict(model=name, k=k, LL=ll, AIC=2 * k - 2 * ll, BIC=k * np.log(len(y)) - 2 * ll, cv_LL=cv,
                         pred_H1=png[D['H'] == 1].mean(), pred_H6=png[D['H'] == 6].mean(),
                         theta=' '.join(f'{v:.3f}' for v in th)))
    T = pd.DataFrame(rows)
    T['dAIC'] = T.AIC - T.AIC.min()
    o1, o6 = R[R.H == 1].non_greedy.mean(), R[R.H == 6].non_greedy.mean()
    with pd.option_context('display.width', 250, 'display.max_colwidth', 60):
        out = '\n'.join([f'[games with equal prior observations only (n={len(y):,}): horizon dependence of noise without any information difference]',
                         f'  observed P(non-greedy): H1 = {o1:.3f}, H6 = {o6:.3f}, effect = {o6 - o1:+.3f}',
                         T.to_string(index=False, float_format=lambda v: f'{v:.3f}')])
    with open(OUT / 'info_condition.txt', 'a', encoding='utf-8') as fh:
        fh.write('\n\n' + out + '\n')
    print(out)


if __name__ == '__main__':
    main()
    equal_only()
