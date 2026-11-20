import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

import wilson_choice as m1
from wilson_choice import build_first_free, as_dict, fit
from run_recovery import S_gamma

from config import RESULTS, WILSON_CSV, WILSON_FORCED_TRIALS

OUT = RESULTS
BAD = 'EE_EE_10-Jul-2014_120840'


def accuracy_by_subject():
    df = pd.read_csv(WILSON_CSV).dropna(subset=['m1', 'm2'])
    df = df[df.gameLength.isin([5, 10])]
    rows = []
    for _, g in df.iterrows():
        best = 1 if g['m1'] > g['m2'] else 2
        for t in range(WILSON_FORCED_TRIALS + 1, int(g['gameLength']) + 1):
            rows.append((g['subjectID'], int(g[f'c{t}'] == best)))
    A = pd.DataFrame(rows, columns=['subjectID', 'correct'])
    return A.groupby('subjectID').correct.agg(['mean', 'size'])


def lap(e, p):
    return e / 2 + (1 - e) * p


def m_soft(th, D):
    return lap(expit(th[3]), expit((D['dR'] + th[2]) / m1.hsel(D, np.exp(th[0]), np.exp(th[1]))))


def m_thomp(th, D):
    return lap(expit(th[3]), norm.cdf((D['dR'] + th[2]) / (m1.hsel(D, np.exp(th[0]), np.exp(th[1])) * D['TU'])))


def m_fp(th, D):
    W = np.sqrt(D['Sigma'] + 2.0 * (D['A'] - 1.0) * D['V_q'] + 2.0 * D['A'] * m1.RHO2)
    return lap(expit(th[2]), norm.cdf((D['dR'] + th[1]) / (np.exp(th[0]) * W)))


def m_fp_gamma(th, D):
    g = expit(th[3])
    S = S_gamma(g, D['A'])
    W = np.sqrt(D['Sigma'] + 2.0 * (S - 1.0) * D['V_q'] + 2.0 * S * m1.RHO2)
    return lap(expit(th[2]), norm.cdf((D['dR'] + th[1]) / (np.exp(th[0]) * W)))


MODELS = [('FP q=U_D + B + lapse', m_fp, [-1.08, -0.27, -1.57]),
          ('FP q=U_D, gamma free + B + lapse', m_fp_gamma, [-1.08, -0.27, -1.57, 3.0]),
          ('softmax tau_H + B + lapse', m_soft, [0.75, 2.1, -0.2, -1.4]),
          ('Thompson tau_H + B + lapse', m_thomp, [-0.5, 0.6, -0.2, -1.4])]


def run(R, label):
    D = as_dict(R)
    y = R['left'].to_numpy(float)
    rows = []
    for name, fn, x0 in MODELS:
        best = None
        for st in (x0, np.array(x0) + 0.5, np.array(x0) - 0.5):
            th, ll = fit(fn, st, D, y)
            if best is None or ll > best[1]:
                best = (th, ll)
        th, ll = best
        extra = f'gamma={expit(th[3]):.3f}' if 'gamma' in name else ''
        rows.append(dict(set=label, model=name, k=len(x0), LL=ll, AIC=2 * len(x0) - 2 * ll, lapse=expit(th[-1] if 'gamma' not in name else th[2]), note=extra))
    T = pd.DataFrame(rows)
    T['dAIC'] = T.AIC - T.AIC.min()
    return T


def main():
    R = build_first_free()
    E = R[R.dI == 0].reset_index(drop=True)
    T = pd.concat([run(E, f'equal prior observations, all 60 (n={len(E):,})'),
                   run(E[~E.subjectID.str.startswith(BAD)].reset_index(drop=True), f'equal prior observations, excluding {BAD}')])
    with pd.option_context('display.width', 250):
        out = '[games with equal prior observations: additional checks (gamma free with lapse; Thompson; outlier participant)]\n' + \
              T.to_string(index=False, float_format=lambda v: f'{v:.3f}')
    acc = accuracy_by_subject()
    low = acc["mean"].idxmin()
    others = acc.drop(index=low)
    out += ('\n\n[accuracy: share of free choices of the arm with the higher true mean]\n'
            f'  lowest: {low}: {acc.loc[low, "mean"]:.3f} (n={int(acc.loc[low, "size"]):,}); '
            f'excluded participant above is the lowest: {low.startswith(BAD)}\n'
            f'  other {len(others)} participants: {others["mean"].min():.3f} to {others["mean"].max():.3f}')
    with open(OUT / 'equal_games_checks.txt', 'w', encoding='utf-8') as fh:
        fh.write(out + '\n')
    print(out)


if __name__ == '__main__':
    main()
