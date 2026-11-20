import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

import wilson_choice as m1
from wilson_choice import build_first_free, as_dict, fit, nll, subset
from model_selection import vb_bms, bor_pxp

from config import RESULTS

from config import N_WORKERS

OUT = RESULTS
N_SIM = 50


def lap(e, p):
    return e / 2 + (1 - e) * p


def m_soft(th, D):
    return lap(expit(th[3]), expit((D['dR'] + th[2]) / m1.hsel(D, np.exp(th[0]), np.exp(th[1]))))


def m_fp(th, D):
    W = np.sqrt(D['Sigma'] + 2.0 * (D['A'] - 1.0) * D['V_q'] + 2.0 * D['A'] * m1.RHO2)
    return lap(expit(th[2]), norm.cdf((D['dR'] + th[1]) / (np.exp(th[0]) * W)))


MODELS = [('softmax tau_H + B + lapse', m_soft, [0.75, 2.1, -0.2, -1.4]),
          ('FP q=U_D + B + lapse', m_fp, [-1.08, -0.27, -1.57])]
G = {}


def _init(D, y, subj):
    G['D'], G['y'], G['subj'] = D, y, subj


def _subj(args):
    j, s = args
    name, fn, x0 = MODELS[j]
    m = G['subj'] == s
    th, ll = fit(fn, x0, subset(G['D'], m), G['y'][m])
    return j, s, th, ll


def _refit(args):
    g, k, yy = args
    out = []
    for j, (name, fn, x0) in enumerate(MODELS):
        th, ll = fit(fn, x0, G['D'], yy)
        out.append((g, k, j, 2 * len(x0) - 2 * ll))
    return out


def main():
    R = build_first_free()
    R = R[R.dI == 0].reset_index(drop=True)
    D = as_dict(R)
    y = R['left'].to_numpy(float)
    subj = R.subjectID.to_numpy()
    uniq = np.unique(subj)
    pooled = [fit(fn, x0, D, y) for _, fn, x0 in MODELS]
    obs = (2 * 4 - 2 * pooled[0][1]) - (2 * 3 - 2 * pooled[1][1])
    with Pool(N_WORKERS, initializer=_init, initargs=(D, y, subj)) as pool:
        res = pool.map(_subj, [(j, s) for j in range(2) for s in uniq])
        S = pd.DataFrame([dict(model=MODELS[j][0], k=len(MODELS[j][2]), subjectID=s, LL=ll, n=int((subj == s).sum()))
                          for j, s, th, ll in res])
        par = {(j, s): th for j, s, th, ll in res}
        lines = [f'[games with equal prior observations: robustness of softmax tau_H vs FP q=U_D (both with B and lapse)]',
                 f'  pooled dAIC = AIC(softmax tau_H) - AIC(FP) = {obs:+.2f}']
        S['AIC'] = 2 * S.k - 2 * S.LL
        S['BIC'] = S.k * np.log(S.n) - 2 * S.LL
        for crit in ('AIC', 'BIC'):
            Wd = S.pivot(index='subjectID', columns='model', values=crit)[[m[0] for m in MODELS]]
            a, Er, xp = vb_bms((-Wd / 2).to_numpy())
            bor, pxp = bor_pxp((-Wd / 2).to_numpy(), a, xp)
            nb = Wd.idxmin(axis=1).value_counts()
            lines.append(f'  subject level {crit}: sum diff (softmax - FP) = {Wd.iloc[:, 0].sum() - Wd.iloc[:, 1].sum():+.1f}; '
                         f'n best: ' + ', '.join(f'{k}={v}' for k, v in nb.items()) +
                         f'; PXP softmax={pxp[0]:.3f}, FP={pxp[1]:.3f}; BOR={bor:.3g}')
        rng = np.random.default_rng(20261029)
        tasks = []
        for shrink in (0.0, 0.5, 1.0):
            for j, (name, fn, x0) in enumerate(MODELS):
                Th = np.array([par[(j, s)] for s in uniq])
                Th = np.clip(Th, np.percentile(Th, 10, axis=0), np.percentile(Th, 90, axis=0))
                Th = pooled[j][0] + shrink * (Th - pooled[j][0])
                p = np.empty(len(y))
                for i, s in enumerate(uniq):
                    m = subj == s
                    p[m] = fn(Th[i], subset(D, m))
                for k in range(N_SIM):
                    tasks.append(((shrink, j), k, (rng.random(len(p)) < p).astype(float)))
        out = pool.map(_refit, tasks, chunksize=2)
    Rm = pd.DataFrame([r for rr in out for r in rr], columns=['gen', 'sim', 'fit', 'AIC'])
    Rm['shrink'] = Rm.gen.map(lambda g: g[0]); Rm['gen_model'] = Rm.gen.map(lambda g: MODELS[g[1]][0])
    lines.append(f'  parametric bootstrap ({N_SIM} sims; generator = per-subject MLE clipped 10-90 pct, shrunk toward pooled):')
    for (sh, gm), g in Rm.groupby(['shrink', 'gen_model']):
        P = g.pivot(index='sim', columns='fit', values='AIC')
        d = P[0] - P[1]
        lines.append(f'    shrink {sh:.1f}, generator {gm:28s}: dAIC mean {d.mean():+7.1f} (5-95%: {d.quantile(.05):+.1f} to '
                     f'{d.quantile(.95):+.1f}); P(FP chosen) = {(d > 0).mean():.2f}; P(dAIC >= observed) = {(d >= obs).mean():.2f}')
    txt = '\n'.join(lines)
    with open(OUT / 'equal_games_robust.txt', 'w', encoding='utf-8') as fh:
        fh.write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
