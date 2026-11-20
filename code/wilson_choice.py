import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

from config import WILSON_CSV, RESULTS, RHO_WILSON, WILSON_FORCED_TRIALS

OUT = RESULTS

SEED = 20240625
RHO2 = RHO_WILSON ** 2
EPS = 1e-9


def build_first_free():
    df = pd.read_csv(WILSON_CSV).dropna(subset=['m1', 'm2'])
    rows = []
    for _, g in df.iterrows():
        H = int(g['gameLength'])
        if H not in (5, 10):
            continue
        c = [g[f'c{t}'] for t in range(1, WILSON_FORCED_TRIALS + 2)]
        r = [g[f'r{t}'] for t in range(1, WILSON_FORCED_TRIALS + 1)]
        n1 = sum(1 for x in c[:4] if x == 1)
        n2 = 4 - n1
        if n1 == 0 or n2 == 0:
            continue
        mu1 = sum(r[t] for t in range(4) if c[t] == 1) / n1
        mu2 = sum(r[t] for t in range(4) if c[t] == 2) / n2
        s1, s2 = RHO2 / n1, RHO2 / n2
        dmu = abs(mu1 - mu2)
        Sigma = s1 + s2
        greedy = 1 if mu1 >= mu2 else 2
        dI = 0 if n1 == n2 else (1 if n1 < n2 else -1)
        rows.append(dict(
            subjectID=g['subjectID'], expt=g['expt_name'], game=g['game'],
            H=H - 4, A=float(H - 4), n1=n1, n2=n2,
            dR=mu1 - mu2, dI=dI, dsig=np.sqrt(s1) - np.sqrt(s2), TU=np.sqrt(Sigma),
            dmu=dmu, Sigma=Sigma, V_q=(Sigma + dmu ** 2) / 4.0,
            left=int(c[4] == 1), non_greedy=int(c[4] != greedy),
            greedy_left=int(greedy == 1),
            m_true_left=g['m1'], m_true_right=g['m2'],
        ))
    R = pd.DataFrame(rows)
    R['info'] = np.where(R.dI == 0, 'equal', 'unequal')
    return R


def lg(x):
    return expit(x)


def pos(x):
    return np.exp(x)


def unit(x):
    return expit(x)


def hsel(D, a1, a6):
    return np.where(D['H'] == 6, a6, a1)


def W(D, lam):
    return np.sqrt(D['Sigma'] + 2.0 * (D['A'] - 1.0) * lam * D['V_q'] + 2.0 * D['A'] * RHO2)


def to_left(p_ng, D):
    return np.where(D['greedy_left'] == 1, 1.0 - p_ng, p_ng)


def m_softmax(th, D):
    return to_left(1.0 / (1.0 + np.exp(D['dmu'] / pos(th[0]))), D)


def m_thompson(th, D):
    return to_left(norm.cdf(-D['dmu'] / (pos(th[0]) * np.sqrt(D['Sigma']))), D)


def m_ucb(th, D):
    ss = np.where(D['greedy_left'] == 1, D['s_left'], D['s_right'])
    sa = np.where(D['greedy_left'] == 1, D['s_right'], D['s_left'])
    gap = D['dmu'] + th[0] * (np.sqrt(ss) - np.sqrt(sa))
    return to_left(1.0 / (1.0 + np.exp(gap / pos(th[1]))), D)


def m_fp(th, D):
    return to_left(1.0 - norm.cdf(D['dmu'] / (pos(th[0]) * W(D, 1.0))), D)


def m_fp_cal(th, D):
    return to_left(1.0 - norm.cdf(D['dmu'] / (pos(th[0]) * W(D, unit(th[1])))), D)


def m_softmax_H(th, D):
    tau = hsel(D, pos(th[0]), pos(th[1]))
    return to_left(1.0 / (1.0 + np.exp(D['dmu'] / tau)), D)


def b_softmax_H(th, D):
    return lg((D['dR'] + th[2]) / hsel(D, pos(th[0]), pos(th[1])))


def b_thompson_H(th, D):
    return norm.cdf((D['dR'] + th[2]) / (hsel(D, pos(th[0]), pos(th[1])) * D['TU']))


def b_ucb_H(th, D):
    q = D['dR'] + hsel(D, th[0], th[1]) * D['dsig'] + th[4]
    return lg(q / hsel(D, pos(th[2]), pos(th[3])))


def b_wilson_H(th, D):
    q = D['dR'] + hsel(D, th[0], th[1]) * D['dI'] + th[4]
    return lg(q / hsel(D, pos(th[2]), pos(th[3])))


def b_wilson_HI(th, D):
    q = D['dR'] + hsel(D, th[0], th[1]) * D['dI'] + th[6]
    s_eq = hsel(D, pos(th[2]), pos(th[3]))
    s_un = hsel(D, pos(th[4]), pos(th[5]))
    return lg(q / np.where(D['dI'] == 0, s_eq, s_un))


def b_gershman_H(th, D):
    z = (hsel(D, th[0], th[1]) * D['dR'] + hsel(D, th[2], th[3]) * D['dsig']
         + hsel(D, th[4], th[5]) * D['dR'] / D['TU'] + th[6])
    return norm.cdf(z)


def b_fp_cal(th, D):
    return norm.cdf((D['dR'] + th[2]) / (pos(th[0]) * W(D, unit(th[1]))))


def b_fp_info(th, D):
    q = D['dR'] + hsel(D, th[2], th[3]) * D['dI'] + th[4]
    return norm.cdf(q / (pos(th[0]) * W(D, unit(th[1]))))


def b_fp_info_UD(th, D):
    q = D['dR'] + hsel(D, th[1], th[2]) * D['dI'] + th[3]
    return norm.cdf(q / (pos(th[0]) * W(D, 1.0)))


def b_wilson_H_probit(th, D):
    q = D['dR'] + hsel(D, th[0], th[1]) * D['dI'] + th[4]
    return norm.cdf(q / hsel(D, pos(th[2]), pos(th[3])))


def b_fp_info_logit(th, D):
    q = D['dR'] + hsel(D, th[2], th[3]) * D['dI'] + th[4]
    return lg(q / (pos(th[0]) * W(D, unit(th[1]))))


def b_fp_cal_logit(th, D):
    return lg((D['dR'] + th[2]) / (pos(th[0]) * W(D, unit(th[1]))))


REPLICATE = [
    ('softmax (mu)', 'horizon-free', m_softmax, [1.0]),
    ('Thompson (tempered)', 'horizon-free', m_thompson, [0.0]),
    ('UCB-softmax', 'horizon-free', m_ucb, [0.0, 1.0]),
    ('future-path, q=U_D', 'future-path', m_fp, [0.0]),
    ('future-path, calib. q', 'future-path', m_fp_cal, [0.0, 2.0]),
    ('softmax (per-H tau)', 'descriptive', m_softmax_H, [1.0, 1.0]),
]

NEW = [
    ('C0  softmax, tau_H + B', 'H-dep. noise', b_softmax_H, [2.0, 2.5, 0.0]),
    ('C3  Thompson, tau_H + B', 'H-dep. noise', b_thompson_H, [0.0, 0.5, 0.0]),
    ('C2  UCB, beta_H + tau_H + B', 'H-dep. bonus+noise', b_ucb_H, [0.0, 1.0, 2.0, 2.5, 0.0]),
    ('C1  Wilson, A_H + sigma_H + B', 'H-dep. bonus+noise', b_wilson_H, [0.0, 5.0, 2.0, 2.5, 0.0]),
    ("C1' Wilson, A_H + sigma_HxI + B", 'H-dep. bonus+noise', b_wilson_HI, [0.0, 5.0, 2.0, 2.5, 2.0, 2.5, 0.0]),
    ('C4  Gershman, w_H + b', 'H-dep. bonus+noise', b_gershman_H, [0.1, 0.05, 0.0, 0.0, 0.5, 0.5, 0.0]),
    ('F1  future-path calib. q + B', 'future-path', b_fp_cal, [0.0, 2.0, 0.0]),
    ('F2  future-path calib. q + A_H + B', 'future-path + bonus', b_fp_info, [0.0, 2.0, 0.0, 5.0, 0.0]),
    ('F2u future-path q=U_D + A_H + B', 'future-path + bonus', b_fp_info_UD, [0.0, 0.0, 5.0, 0.0]),
]

def with_lapse(fn):
    def g(th, D):
        e = expit(th[-1])
        return e / 2.0 + (1.0 - e) * fn(th[:-1], D)
    return g


def b_fp0_info(th, D):
    q = D['dR'] + hsel(D, th[1], th[2]) * D['dI'] + th[3]
    return norm.cdf(q / (pos(th[0]) * W(D, 0.0)))


LAPSE = [(n.split(' ', 1)[0] + '+e ' + n.split(' ', 1)[1].strip() + ' + lapse', kind + ' + lapse',
          with_lapse(fn), list(x0) + [-2.0])
         for n, kind, fn, x0 in NEW if n.split(' ')[0] != 'F2']
LAPSE += [
    ('F0+e future-path lambda=0 + A_H + B + lapse', 'ablation + lapse', with_lapse(b_fp0_info), [0.0, 0.0, 5.0, 0.0, -2.0]),
]

LINK = [
    ('L1  C1 with probit link', 'link diagnostic', b_wilson_H_probit, [0.0, 5.0, 2.0, 2.5, 0.0]),
    ('L2  F1 with logistic link', 'link diagnostic', b_fp_cal_logit, [-0.5, 2.0, 0.0]),
    ('L3  F2 with logistic link', 'link diagnostic', b_fp_info_logit, [-0.5, 2.0, 0.0, 5.0, 0.0]),
]


def nll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit(fn, x0, D, y):
    f = lambda th: nll(fn(th, D), y)
    best = None
    for start in (np.asarray(x0, float), np.asarray(x0, float) + 0.3):
        r1 = minimize(f, start, method='L-BFGS-B')
        r2 = minimize(f, r1.x, method='Nelder-Mead',
                      options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 40000, 'maxfev': 40000})
        r = r2 if r2.fun <= r1.fun else r1
        if best is None or r.fun < best.fun:
            best = r
    return best.x, -best.fun


def subset(D, m):
    return {k: v[m] for k, v in D.items()}


def as_dict(R):
    s_left = RHO2 / R['n1'].to_numpy(float)
    s_right = RHO2 / R['n2'].to_numpy(float)
    return dict(
        H=R['H'].to_numpy(), A=R['A'].to_numpy(float), dR=R['dR'].to_numpy(float),
        dI=R['dI'].to_numpy(float), dsig=R['dsig'].to_numpy(float), TU=R['TU'].to_numpy(float),
        dmu=R['dmu'].to_numpy(float), Sigma=R['Sigma'].to_numpy(float), V_q=R['V_q'].to_numpy(float),
        greedy_left=R['greedy_left'].to_numpy(), s_left=s_left, s_right=s_right,
    )
