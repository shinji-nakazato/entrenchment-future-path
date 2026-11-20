import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import chi2

import wilson_choice as m1
from wilson_choice import build_first_free, as_dict, fit
from run_valuefree import fpr, fp8, vfree
from config import N_WORKERS, RESULTS

OUT = RESULTS
N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SEED = 20261118

LAP = {n.split(' ', 1)[0]: (fn, x0) for n, _, fn, x0 in m1.LAPSE}
MODELS = [('future-path, rho free', fpr, [-0.74, -0.26, 0.33, -1.57]),
          ('future-path, rho = 8', fp8, [-0.74, -0.26, -1.57]),
          ('value-free random exploration (epsilon_H)', vfree, [0.99, -0.33, -1.67, -0.45]),
          ('horizon-specific temperature (tau_H)', *LAP['C0+e'])]
G = {}


def ng_pred(fn, th, D):
    pl = fn(th, D)
    return np.where(D['greedy_left'] == 1, 1 - pl, pl)


def profile(ng, q, H):
    return np.array([ng[(q == k) & (H == 6)].mean() - ng[(q == k) & (H == 1)].mean() for k in range(5)])


def stats(E, x0s=None):
    D, y = as_dict(E), E['left'].to_numpy(float)
    q, H = E['gq'].to_numpy(), E['H'].to_numpy()
    obs = profile(E['non_greedy'].to_numpy(float), q, H)
    preds, ths = [], []
    for k, (name, fn, x0) in enumerate(MODELS):
        th, _ = fit(fn, x0 if x0s is None else x0s[k], D, y)
        ths.append(th)
        preds.append(profile(ng_pred(fn, th, D), q, H))
    return obs, np.array(preds), ths


def _boot(seed):
    E, subj, uniq, idx, ths = G['E'], G['subj'], G['uniq'], G['idx'], G['ths']
    rng = np.random.default_rng(seed)
    ss = rng.choice(uniq, len(uniq), replace=True)
    Eb = E.iloc[np.concatenate([idx[s] for s in ss])].reset_index(drop=True)
    obs, preds, _ = stats(Eb, ths)
    return obs, preds


def _init(E, ths):
    subj = E['subjectID'].to_numpy()
    uniq = np.unique(subj)
    G.update(E=E, subj=subj, uniq=uniq, idx={s: np.flatnonzero(subj == s) for s in uniq}, ths=ths)


def contrasts(v):
    w_lin = np.arange(5) - 2.0
    return np.array([w_lin @ v / (w_lin @ w_lin), v[1:4].mean() - v[[0, 4]].mean()])


def main():
    R = build_first_free()
    E = R[R.dI == 0].reset_index(drop=True)
    E['gq'] = pd.qcut(E.dmu, 5, labels=False)
    obs, preds, ths = stats(E)
    seeds = np.random.default_rng(SEED).integers(0, 2**31 - 1, N_BOOT)
    with Pool(N_WORKERS, initializer=_init, initargs=(E, ths)) as pool:
        res = pool.map(_boot, seeds)
    OB = np.array([r[0] for r in res])
    PB = np.array([r[1] for r in res])

    lines = [f'Shape of the horizon effect across posterior-gap quintiles, games with equal prior observations (n={len(E):,}, 60 subjects)',
             f'participant bootstrap with refitting of every model, {N_BOOT} resamples; quintile edges fixed at the full-data values', '',
             '[horizon effect P(ng | H=6) - P(ng | H=1) by quintile]',
             '  observed            : ' + '  '.join(f'{v:+.3f}' for v in obs)]
    for k, (name, *_) in enumerate(MODELS):
        lines.append(f'  {name:<44}: ' + '  '.join(f'{v:+.3f}' for v in preds[k]))
    lines += ['', '[Wald test of observed profile = model profile; covariance of (observed - predicted) from the bootstrap]']
    for k, (name, *_) in enumerate(MODELS):
        d = obs - preds[k]
        S = np.cov((OB - PB[:, k, :]).T)
        w = float(d @ np.linalg.solve(S, d))
        lines.append(f'  {name:<44}: chi2(5) = {w:6.2f}, p = {chi2.sf(w, 5):.4f}')
    lines += ['', '[shape contrasts: linear trend (slope per quintile) and hump (mean of quintiles 2-4 minus mean of 1 and 5)]']
    co = contrasts(obs)
    cb = np.array([contrasts(v) for v in OB])
    lo, hi = np.percentile(cb, [2.5, 97.5], axis=0)
    lines.append(f'  observed            : slope {co[0]:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}]   hump {co[1]:+.4f} [{lo[1]:+.4f}, {hi[1]:+.4f}]')
    for k, (name, *_) in enumerate(MODELS):
        cp = contrasts(preds[k])
        diff = np.array([contrasts(OB[b]) - contrasts(PB[b, k]) for b in range(N_BOOT)])
        dl, dh = np.percentile(diff, [2.5, 97.5], axis=0)
        lines.append(f'  {name:<44}: slope {cp[0]:+.4f} (obs - model 95% CI [{dl[0]:+.4f}, {dh[0]:+.4f}])   '
                     f'hump {cp[1]:+.4f} (obs - model 95% CI [{dl[1]:+.4f}, {dh[1]:+.4f}])')
    txt = '\n'.join(lines)
    (OUT / 'shape_test.txt').write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
