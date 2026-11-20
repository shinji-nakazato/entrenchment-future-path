import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

import wilson_choice as m1
from wilson_choice import build_first_free, as_dict, fit
from run_assumptions import lap, W_mm, p_exact

from config import RESULTS

OUT = RESULTS


def main():
    R = build_first_free()
    E = R[R.dI == 0].reset_index(drop=True)
    D = as_dict(E)
    y = E['left'].to_numpy(float)
    mm = lambda th, D: lap(expit(th[2]), norm.cdf((D['dR'] + th[1]) / (np.exp(th[0]) * W_mm(D, m1.RHO2, D['V_q']))))
    ex = lambda th, D: lap(expit(th[2]), p_exact(D, np.exp(th[0]), m1.RHO2, None, th[1]))
    gl = D['greedy_left'] == 1
    E = E.assign(gbin=pd.qcut(E.dmu, 5))
    out = ['', '[games with equal prior observations: P(non-greedy) by posterior-gap quintile and horizon, observed vs moment-matched vs exact mixture]']
    preds = {}
    for name, fn, x0 in [('moment-matched', mm, [-1.08, -0.27, -1.57]), ('exact mixture', ex, [-1.08, -0.27, -1.57])]:
        th, ll = fit(fn, x0, D, y)
        pl = fn(th, D)
        preds[name] = np.where(gl, 1 - pl, pl)
        out.append(f'  {name}: tau = {np.exp(th[0]):.3f}, B = {th[1]:+.3f}, lapse = {expit(th[2]):.3f}, LL = {ll:.1f}')
    E['mm'] = preds['moment-matched']; E['ex'] = preds['exact mixture']
    t = E.groupby(['H', 'gbin'], observed=True).agg(n=('dmu', 'size'), obs=('non_greedy', 'mean'), mm=('mm', 'mean'), ex=('ex', 'mean'))
    out.append('  ' + t.to_string(float_format=lambda v: f'{v:.3f}').replace('\n', '\n  '))
    txt = '\n'.join(out)
    with open(OUT / 'assumptions.txt', 'a', encoding='utf-8') as fh:
        fh.write('\n' + txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
