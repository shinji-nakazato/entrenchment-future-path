import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

import wilson_choice as m1
from wilson_choice import build_first_free, as_dict, fit, nll, subset, hsel
from model_selection import vb_bms, bor_pxp

from config import RESULTS

from config import N_WORKERS

OUT = RESULTS
N_SIM = 50


def lap(e, p):
    return e / 2 + (1 - e) * p


def Wfp(D, r2):
    return np.sqrt(D['Sigma'] + 2.0 * (D['A'] - 1.0) * D['V_q'] + 2.0 * D['A'] * r2)


def fp8(th, D):
    return lap(expit(th[2]), norm.cdf((D['dR'] + th[1]) / (np.exp(th[0]) * Wfp(D, m1.RHO2))))


def fpr(th, D):
    return lap(expit(th[3]), norm.cdf((D['dR'] + th[1]) / (np.exp(th[0]) * Wfp(D, np.exp(2 * th[2])))))


def vfree(th, D):
    return lap(expit(hsel(D, th[2], th[3])), expit((D['dR'] + th[1]) / np.exp(th[0])))


def f2u(th, D):
    return lap(expit(th[4]), norm.cdf((D['dR'] + hsel(D, th[1], th[2]) * D['dI'] + th[3]) / (np.exp(th[0]) * Wfp(D, m1.RHO2))))


def f2ur(th, D):
    return lap(expit(th[5]), norm.cdf((D['dR'] + hsel(D, th[1], th[2]) * D['dI'] + th[3]) / (np.exp(th[0]) * Wfp(D, np.exp(2 * th[4])))))


def vfree_all(th, D):
    return lap(expit(hsel(D, th[4], th[5])), expit((D['dR'] + hsel(D, th[1], th[2]) * D['dI'] + th[3]) / np.exp(th[0])))


SETS = {
    'equal': [('FP q=U_D (rho = 8)', fp8, [-1.08, -0.27, -1.57]),
              ('FP q=U_D (rho free)', fpr, [-0.74, -0.26, 0.33, -1.57]),
              ('value-free RE: softmax + lapse_H', vfree, [0.99, -0.33, -1.67, -0.45])],
    'all': [('F2u+e (rho = 8)', f2u, [-0.965, -1.44, 5.27, -0.39, -1.46]),
            ('F2u+e (rho free)', f2ur, [-1.0, -1.44, 5.27, -0.39, 1.3, -1.46]),
            ('value-free RE + A_H', vfree_all, [1.26, -1.44, 3.84, -0.34, -1.57, -0.36])],
}
G = {}


def _init(D, y, subj, tag):
    G.update(D=D, y=y, subj=subj, tag=tag)


def _subj(args):
    j, s = args
    name, fn, x0 = SETS[G['tag']][j]
    m = G['subj'] == s
    best = None
    for st in (np.array(x0), np.array(x0) + 0.4, np.array(x0) - 0.4):
        th, ll = fit(fn, st, subset(G['D'], m), G['y'][m])
        if best is None or ll > best[1]:
            best = (th, ll)
    return j, s, best[0], best[1]


def _refit(args):
    g, k, yy, starts = args
    out = []
    for j, (name, fn, x0) in enumerate(SETS[G['tag']]):
        best = -np.inf
        for st in (starts[j], np.array(x0)):
            th, ll = fit(fn, st, G['D'], yy)
            best = max(best, ll)
        out.append((g, k, j, 2 * len(x0) - 2 * best))
    return out


def main():
    R = build_first_free()
    lines = []
    for tag, Rs in [('equal', R[R.dI == 0].reset_index(drop=True)), ('all', R)]:
        D = as_dict(Rs)
        y = Rs['left'].to_numpy(float)
        subj = Rs.subjectID.to_numpy()
        uniq = np.unique(subj)
        M = SETS[tag]
        pooled = [fit(fn, x0, D, y) for _, fn, x0 in M]
        aic = [2 * len(x0) - 2 * ll for (_, _, x0), (_, ll) in zip(M, pooled)]
        lines.append(f'=== {tag} games (n={len(y):,}): ' + ', '.join(f'{m[0]} AIC {a:.1f}' for m, a in zip(M, aic)))
        if tag == 'equal':
            gl = D['greedy_left'] == 1
            Rb = Rs.assign(gbin=pd.qcut(Rs.dmu, 5))
            for (name, fn, x0), (th, ll) in zip(M, pooled):
                pl = fn(th, D)
                Rb[name] = np.where(gl, 1 - pl, pl)
            t = Rb.groupby(['H', 'gbin'], observed=True)[['non_greedy'] + [m[0] for m in M]].mean()
            lines += ['[P(non-greedy) by posterior-gap quintile x H]', t.to_string(float_format=lambda v: f'{v:.3f}')]
            e = Rb.groupby(['gbin'], observed=True).apply(lambda g: pd.Series({c: g[g.H == 6][c].mean() - g[g.H == 1][c].mean() for c in ['non_greedy'] + [m[0] for m in M]}))
            lines += ['[horizon effect (H6 - H1) by gap quintile]', e.to_string(float_format=lambda v: f'{v:+.3f}'), '']
        with Pool(N_WORKERS, initializer=_init, initargs=(D, y, subj, tag)) as pool:
            res = pool.map(_subj, [(j, s) for j in range(len(M)) for s in uniq])
            S = pd.DataFrame([dict(model=M[j][0], k=len(M[j][2]), subjectID=s, LL=ll, n=int((subj == s).sum())) for j, s, th, ll in res])
            par = {(j, s): th for j, s, th, ll in res}
            S['AIC'] = 2 * S.k - 2 * S.LL
            S['BIC'] = S.k * np.log(S.n) - 2 * S.LL
            for crit in ('AIC', 'BIC'):
                Wd = S.pivot(index='subjectID', columns='model', values=crit)[[m[0] for m in M]]
                a, Er, xp = vb_bms((-Wd / 2).to_numpy())
                bor, pxp = bor_pxp((-Wd / 2).to_numpy(), a, xp)
                nb = Wd.idxmin(axis=1).value_counts().reindex([m[0] for m in M]).fillna(0).astype(int)
                lines.append(f'  subject level {crit}: sum ' + ', '.join(f'{m[0]}={Wd[m[0]].sum():.1f}' for m in M) +
                             ' | n best ' + ', '.join(f'{k}={v}' for k, v in nb.items()) +
                             ' | PXP ' + ', '.join(f'{m[0]}={p:.3f}' for m, p in zip(M, pxp)) + f' | BOR={bor:.3g}')
            rng = np.random.default_rng(20261031)
            tasks = []
            for shrink in (0.0, 0.5, 1.0):
                for j, (name, fn, x0) in enumerate(M):
                    if j == 1:
                        continue
                    Th = np.array([par[(j, s)] for s in uniq])
                    Th = np.clip(Th, np.percentile(Th, 10, axis=0), np.percentile(Th, 90, axis=0))
                    Th = pooled[j][0] + shrink * (Th - pooled[j][0])
                    p = np.empty(len(y))
                    for i, s in enumerate(uniq):
                        mm = subj == s
                        p[mm] = fn(Th[i], subset(D, mm))
                    for k in range(N_SIM):
                        tasks.append(((shrink, j), k, (rng.random(len(p)) < p).astype(float), [pp[0] for pp in pooled]))
            out = pool.map(_refit, tasks, chunksize=2)
        Rm = pd.DataFrame([r for rr in out for r in rr], columns=['gen', 'sim', 'fit', 'AIC'])
        lines.append(f'  parametric bootstrap ({N_SIM} sims; generators = per-subject MLE clipped 10-90 pct, shrunk toward pooled)')
        for jfp in (0, 1):
            obs = aic[2] - aic[jfp]
            lines.append(f'   dAIC = AIC(value-free) - AIC({M[jfp][0]}); observed {obs:+.1f}')
            for (sh, gj), g in Rm.groupby([Rm.gen.map(lambda v: v[0]), Rm.gen.map(lambda v: v[1])]):
                P = g.pivot(index='sim', columns='fit', values='AIC')
                d = P[2] - P[jfp]
                lines.append(f'     shrink {sh:.1f}, generator {M[gj][0]:34s}: mean {d.mean():+7.1f} (5-95%: {d.quantile(.05):+.1f} to '
                             f'{d.quantile(.95):+.1f}); P(FP chosen) = {(d > 0).mean():.2f}; P(dAIC >= observed) = {(d >= obs).mean():.2f}')
        lines.append('')
    out = '\n'.join(lines)
    (OUT / 'valuefree.txt').write_text(out, encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
