import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

import wilson_choice as m1
from wilson_choice import build_first_free, as_dict, fit, nll, subset, hsel

from config import RESULTS

OUT = RESULTS
RHO2 = m1.RHO2
DG = np.linspace(-150, 150, 2401)
NMAX = 12
AMAX = 6
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(31)
GH_W = GH_W / GH_W.sum()


def build_dp():
    v = {0: np.zeros((NMAX + 1, NMAX + 1, len(DG)))}
    dq = {}
    for A in range(1, AMAX + 1):
        vA = np.zeros_like(v[0])
        dA = np.zeros_like(v[0])
        prev = v[A - 1]
        for n1 in range(1, NMAX):
            for n2 in range(1, NMAX):
                s1 = np.sqrt(RHO2 / (n1 * (n1 + 1)))
                s2 = np.sqrt(RHO2 / (n2 * (n2 + 1)))
                e1 = np.zeros(len(DG)); e2 = np.zeros(len(DG))
                for x, w in zip(GH_X, GH_W):
                    e1 += w * np.interp(DG + s1 * x, DG, prev[n1 + 1, n2])
                    e2 += w * np.interp(DG - s2 * x, DG, prev[n1, n2 + 1])
                q1 = DG + e1
                q2 = e2
                vA[n1, n2] = np.maximum(q1, q2)
                dA[n1, n2] = q1 - q2
        v[A] = vA
        dq[A] = dA
    return dq


def lap(e, p):
    return e / 2 + (1 - e) * p


def main():
    R = build_first_free()
    dq = build_dp()
    n1 = R.n1.to_numpy(); n2 = R.n2.to_numpy(); A = R.A.to_numpy().astype(int); dR = R.dR.to_numpy(float)
    R['dQ'] = [np.interp(d, DG, dq[a][i, j]) for d, a, i, j in zip(dR, A, n1, n2)]
    lines = ['Comparison with existing mechanistic accounts of horizon-dependent random exploration',
             'Wilson first free choice. Bayes-optimal finite-horizon Q-values: flat prior (posterior mean = sample mean), '
             'rho = 8, two arms, computed by backward induction on (d, n1, n2) with Gauss-Hermite quadrature.', '']
    for (h, info), g in R.groupby(['H', 'info']):
        sl = np.polyfit(g.dR, g.dQ, 1)[0]
        lines.append(f'  H={h}, {info:7s}: slope of optimal dQ on dR = {sl:.3f}; mean dQ - dR at dR = 0 by dI: ' +
                     ', '.join(f'dI={k:+d}: {np.mean(s.dQ - s.dR):+.2f}' for k, s in g.groupby('dI')))
    lines.append('')

    def mk(R):
        D = as_dict(R)
        D['dQ'] = R['dQ'].to_numpy(float)
        D['Hk'] = R['H'].to_numpy()
        return D

    Wfp = lambda D, r2=RHO2: np.sqrt(D['Sigma'] + 2.0 * (D['A'] - 1.0) * D['V_q'] + 2.0 * D['A'] * r2)
    eps_H = lambda th_a, th_b, D: expit(hsel(D, th_a, th_b))
    MODELS_EQ = [
        ('FP q=U_D + B + lapse (rho = 8)', lambda th, D: lap(expit(th[2]), norm.cdf((D['dR'] + th[1]) / (np.exp(th[0]) * Wfp(D)))), [-1.08, -0.27, -1.57]),
        ('FP q=U_D + B + lapse (rho free)', lambda th, D: lap(expit(th[3]), norm.cdf((D['dR'] + th[1]) / (np.exp(th[0]) * Wfp(D, np.exp(2 * th[2]))))), [-1.08, -0.27, 1.0, -1.57]),
        ('Bayes-optimal Q + softmax (one tau) + B + lapse', lambda th, D: lap(expit(th[2]), expit((D['dQ'] + th[1]) / np.exp(th[0]))), [1.5, -0.3, -1.5]),
        ('Bayes-optimal Q + probit (one tau) + B + lapse', lambda th, D: lap(expit(th[2]), norm.cdf((D['dQ'] + th[1]) / np.exp(th[0]))), [2.0, -0.3, -1.5]),
        ('softmax tau_H + B + lapse (Wilson 2014 / DDM threshold_H, choice level)', lambda th, D: lap(expit(th[3]), expit((D['dR'] + th[2]) / hsel(D, np.exp(th[0]), np.exp(th[1])))), [0.75, 2.1, -0.2, -1.4]),
        ('softmax one tau + B + lapse_H (value-free random exploration)', lambda th, D: lap(eps_H(th[2], th[3], D), expit((D['dR'] + th[1]) / np.exp(th[0]))), [1.0, -0.2, -2.0, -1.0]),
        ('softmax tau_H + B + lapse_H', lambda th, D: lap(eps_H(th[3], th[4], D), expit((D['dR'] + th[2]) / hsel(D, np.exp(th[0]), np.exp(th[1])))), [0.75, 2.1, -0.2, -1.6, -1.2]),
        ('FP q=U_D + B + lapse_H (rho = 8)', lambda th, D: lap(eps_H(th[2], th[3], D), norm.cdf((D['dR'] + th[1]) / (np.exp(th[0]) * Wfp(D)))), [-1.08, -0.27, -1.57, -1.57]),
        ('Bayes-optimal Q + softmax + B + lapse_H', lambda th, D: lap(eps_H(th[2], th[3], D), expit((D['dQ'] + th[1]) / np.exp(th[0]))), [1.5, -0.3, -1.5, -1.5]),
    ]
    MODELS_ALL = [
        ('F2u+e: FP q=U_D + A_H + B + lapse (rho = 8)', lambda th, D: lap(expit(th[4]), norm.cdf((D['dR'] + hsel(D, th[1], th[2]) * D['dI'] + th[3]) / (np.exp(th[0]) * Wfp(D)))), [-0.965, -1.44, 5.27, -0.39, -1.46]),
        ('Bayes-optimal Q + softmax + B + lapse (no free bonus)', lambda th, D: lap(expit(th[2]), expit((D['dQ'] + th[1]) / np.exp(th[0]))), [1.5, -0.3, -1.5]),
        ('Bayes-optimal Q + probit + B + lapse (no free bonus)', lambda th, D: lap(expit(th[2]), norm.cdf((D['dQ'] + th[1]) / np.exp(th[0]))), [2.0, -0.3, -1.5]),
        ('Bayes-optimal Q + A_H + softmax + B + lapse', lambda th, D: lap(expit(th[4]), expit((D['dQ'] + hsel(D, th[1], th[2]) * D['dI'] + th[3]) / np.exp(th[0]))), [1.5, 0.0, 0.0, -0.3, -1.5]),
        ('C1+e: Wilson A_H + sigma_H + B + lapse', lambda th, D: lap(expit(th[5]), expit((D['dR'] + hsel(D, th[0], th[1]) * D['dI'] + th[4]) / hsel(D, np.exp(th[2]), np.exp(th[3])))), [-1.52, 6.58, 1.04, 2.32, -0.35, -1.40]),
        ('value-free: softmax one tau + A_H + B + lapse_H', lambda th, D: lap(eps_H(th[4], th[5], D), expit((D['dR'] + hsel(D, th[1], th[2]) * D['dI'] + th[3]) / np.exp(th[0]))), [1.0, -1.5, 6.0, -0.3, -2.0, -1.0]),
        ('C1+e with lapse_H', lambda th, D: lap(eps_H(th[5], th[6], D), expit((D['dR'] + hsel(D, th[0], th[1]) * D['dI'] + th[4]) / hsel(D, np.exp(th[2]), np.exp(th[3])))), [-1.52, 6.58, 1.04, 2.32, -0.35, -1.6, -1.2]),
        ('F2u+e with lapse_H', lambda th, D: lap(eps_H(th[4], th[5], D), norm.cdf((D['dR'] + hsel(D, th[1], th[2]) * D['dI'] + th[3]) / (np.exp(th[0]) * Wfp(D)))), [-0.965, -1.44, 5.27, -0.39, -1.46, -1.46]),
    ]
    subj_all = R.subjectID.to_numpy()
    rng = np.random.default_rng(20240625)
    uniq = np.unique(subj_all)
    fold = {s: i % 5 for i, s in enumerate(rng.permutation(uniq))}
    for Rs, MODELS, label in [(R[R.dI == 0].reset_index(drop=True), MODELS_EQ, 'games with equal prior observations'),
                              (R, MODELS_ALL, 'all games (information bonus where applicable)')]:
        D = mk(Rs)
        y = Rs['left'].to_numpy(float)
        folds = np.array([fold[s] for s in Rs.subjectID])
        gl = D['greedy_left'] == 1
        rows, cells = [], []
        for name, fn, x0 in MODELS:
            best = None
            for st in (np.array(x0, float), np.array(x0, float) + 0.4, np.array(x0, float) - 0.4):
                th, ll = fit(fn, st, D, y)
                if best is None or ll > best[1]:
                    best = (th, ll)
            th, ll = best
            cv = 0.0
            for f in range(5):
                tr, te = folds != f, folds == f
                th_f, _ = fit(fn, th, subset(D, tr), y[tr])
                cv += -nll(fn(th_f, subset(D, te)), y[te])
            pl = fn(th, D)
            png = np.where(gl, 1 - pl, pl)
            rows.append(dict(model=name, k=len(x0), LL=ll, AIC=2 * len(x0) - 2 * ll, BIC=len(x0) * np.log(len(y)) - 2 * ll,
                             cv_LL=cv, pred_H1=png[D['H'] == 1].mean(), pred_H6=png[D['H'] == 6].mean(),
                             theta=' '.join(f'{v:.3f}' for v in th)))
            tmp = Rs[['H', 'info', 'dI', 'greedy_left', 'non_greedy']].assign(p=png)
            for (h, inf), gg in tmp.groupby(['H', 'info']):
                cells.append(dict(model=name, cell=f'H{h}-{inf}', pred=gg.p.mean()))
        T = pd.DataFrame(rows)
        T['dAIC'] = T.AIC - T.AIC.min()
        T['dBIC'] = T.BIC - T.BIC.min()
        o = Rs.groupby('H').non_greedy.mean()
        C = pd.DataFrame(cells).pivot_table(index='model', columns='cell', values='pred', sort=False)
        obs = Rs.assign(cell='H' + Rs.H.astype(str) + '-' + Rs['info']).groupby('cell').non_greedy.mean()
        C.loc['OBSERVED'] = obs
        with pd.option_context('display.width', 260, 'display.max_colwidth', 75):
            lines += [f'[{label}, n={len(y):,}; observed P(ng) H1 = {o[1]:.3f}, H6 = {o[6]:.3f}]',
                      T[['model', 'k', 'LL', 'AIC', 'dAIC', 'BIC', 'dBIC', 'cv_LL', 'pred_H1', 'pred_H6']].to_string(index=False, float_format=lambda v: f'{v:.3f}'),
                      '', '  parameters:'] + [f'    {r.model}: {r.theta}' for r in T.itertuples()] + \
                     ['', '  P(non-greedy) by cell:', '  ' + C.to_string(float_format=lambda v: f'{v:.3f}').replace('\n', '\n  '), '']
    out = '\n'.join(lines)
    (OUT / 'competitors.txt').write_text(out, encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
