import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm, binom

import wilson_choice as m1
from wilson_choice import build_first_free, as_dict, fit, nll, subset, hsel

from config import RESULTS

OUT = RESULTS
RHO_TASK = 8.0


def lap(e, p):
    return e / 2 + (1 - e) * p


def W_mm(D, rho2, Vq):
    return np.sqrt(D['Sigma'] + 2.0 * (D['A'] - 1.0) * Vq + 2.0 * D['A'] * rho2)


def Vq_theta(D, theta):
    w = expit(theta * D['dmu'])
    return w * (1 - w) * (D['Sigma'] + D['dmu'] ** 2)


def num(D, th_info, B):
    if th_info is None:
        return D['dR'] + B
    return D['dR'] + hsel(D, th_info[0], th_info[1]) * D['dI'] + B


def kdist(Am1):
    ks = np.arange(-Am1, Am1 + 1)
    pa = binom.pmf(np.arange(Am1 + 1), Am1, 0.5)
    pk = np.convolve(pa, pa[::-1])
    return ks, pk


KD = {int(a): kdist(int(a) - 1) for a in (1, 6)}


def p_exact(D, tau, rho2, info, B):
    out = np.zeros(len(D['A']))
    for a, (ks, pk) in KD.items():
        m = D['A'] == a
        dR, Sig = D['dR'][m], D['Sigma'][m]
        extra = (hsel(subset(D, m), info[0], info[1]) * D['dI'][m] if info is not None else 0.0) + B
        acc = np.zeros(m.sum())
        for k, w in zip(ks, pk):
            c = 1.0 + k
            acc += w * norm.cdf((c * dR + extra) / (tau * np.sqrt(c * c * Sig + 2.0 * a * rho2)))
        out[m] = acc
    return out


def models(info):
    M = []
    if not info:
        M.append(('FP q=U_D (rho = 8)', lambda th, D: lap(expit(th[2]), norm.cdf((D['dR'] + th[1]) / (np.exp(th[0]) * W_mm(D, m1.RHO2, D['V_q'])))), [-1.08, -0.27, -1.57]))
        M.append(('FP q_theta (theta free)', lambda th, D: lap(expit(th[3]), norm.cdf((D['dR'] + th[1]) / (np.exp(th[0]) * W_mm(D, m1.RHO2, Vq_theta(D, np.exp(th[2])))))), [-1.08, -0.27, -4.0, -1.57]))
        M.append(('FP q=U_D, rho free', lambda th, D: lap(expit(th[3]), norm.cdf((D['dR'] + th[1]) / (np.exp(th[0]) * W_mm(D, np.exp(2 * th[2]), D['V_q'])))), [-1.08, -0.27, np.log(8.0), -1.57]))
        M.append(('exact mixture q=U_D (rho = 8)', lambda th, D: lap(expit(th[2]), p_exact(D, np.exp(th[0]), m1.RHO2, None, th[1])), [-1.08, -0.27, -1.57]))
        M.append(('exact mixture q=U_D, rho free', lambda th, D: lap(expit(th[3]), p_exact(D, np.exp(th[0]), np.exp(2 * th[2]), None, th[1])), [-1.08, -0.27, np.log(8.0), -1.57]))
    else:
        M.append(('F2u+e: FP q=U_D + A_H + B (rho = 8)', lambda th, D: lap(expit(th[4]), norm.cdf((D['dR'] + hsel(D, th[1], th[2]) * D['dI'] + th[3]) / (np.exp(th[0]) * W_mm(D, m1.RHO2, D['V_q'])))), [-0.965, -1.44, 5.27, -0.39, -1.46]))
        M.append(('+ q_theta (theta free)', lambda th, D: lap(expit(th[5]), norm.cdf((D['dR'] + hsel(D, th[1], th[2]) * D['dI'] + th[3]) / (np.exp(th[0]) * W_mm(D, m1.RHO2, Vq_theta(D, np.exp(th[4])))))), [-0.965, -1.44, 5.27, -0.39, -4.0, -1.46]))
        M.append(('+ rho free', lambda th, D: lap(expit(th[5]), norm.cdf((D['dR'] + hsel(D, th[1], th[2]) * D['dI'] + th[3]) / (np.exp(th[0]) * W_mm(D, np.exp(2 * th[4]), D['V_q'])))), [-0.965, -1.44, 5.27, -0.39, np.log(8.0), -1.46]))
        M.append(('exact mixture q=U_D + A_H + B (rho = 8)', lambda th, D: lap(expit(th[4]), p_exact(D, np.exp(th[0]), m1.RHO2, (th[1], th[2]), th[3])), [-0.965, -1.44, 5.27, -0.39, -1.46]))
        M.append(('exact mixture + rho free', lambda th, D: lap(expit(th[5]), p_exact(D, np.exp(th[0]), np.exp(2 * th[4]), (th[1], th[2]), th[3])), [-0.965, -1.44, 5.27, -0.39, np.log(8.0), -1.46]))
    return M


def run(R, info, label):
    D = as_dict(R)
    y = R['left'].to_numpy(float)
    rows = []
    for name, fn, x0 in models(info):
        best = None
        for st in (np.array(x0), np.array(x0) + 0.4, np.array(x0) - 0.4):
            th, ll = fit(fn, st, D, y)
            if best is None or ll > best[1]:
                best = (th, ll)
        th, ll = best
        note = ''
        if 'theta' in name:
            note = f'theta = {np.exp(th[-2]):.4f}'
        if 'rho free' in name:
            note = f'rho = {np.exp(th[-2]):.2f}'
        rows.append(dict(set=label, model=name, k=len(x0), LL=ll, AIC=2 * len(x0) - 2 * ll, lapse=expit(th[-1]), note=note))
    T = pd.DataFrame(rows)
    T['dAIC'] = T.AIC - T.AIC.min()
    return T, D, y


def profile_rho(R, info, label):
    D = as_dict(R)
    y = R['left'].to_numpy(float)
    rows = []
    for rho in (2, 4, 6, 7, 8, 9, 10, 12, 16, 24):
        r2 = float(rho) ** 2
        if info:
            fn = lambda th, D: lap(expit(th[4]), norm.cdf((D['dR'] + hsel(D, th[1], th[2]) * D['dI'] + th[3]) / (np.exp(th[0]) * W_mm(D, r2, D['V_q']))))
            x0 = [-0.965, -1.44, 5.27, -0.39, -1.46]
        else:
            fn = lambda th, D: lap(expit(th[2]), norm.cdf((D['dR'] + th[1]) / (np.exp(th[0]) * W_mm(D, r2, D['V_q']))))
            x0 = [-1.08, -0.27, -1.57]
        th, ll = fit(fn, np.array(x0) + np.r_[np.log(8.0 / rho), np.zeros(len(x0) - 1)], D, y)
        gl = D['greedy_left'] == 1
        pl = fn(th, D)
        png = np.where(gl, 1 - pl, pl)
        rows.append(dict(set=label, rho=rho, LL=ll, pred_H1=png[D['H'] == 1].mean(), pred_H6=png[D['H'] == 6].mean()))
    P = pd.DataFrame(rows)
    P['dLL'] = P.LL - P.LL.max()
    return P


def exact_vs_mm_gap(D):
    rows = []
    for a in (1, 6):
        m = D['A'] == a
        sub = subset(D, m)
        pe = p_exact(sub, 1.0, m1.RHO2, None, 0.0)
        pm = norm.cdf(sub['dR'] / W_mm(sub, m1.RHO2, sub['V_q']))
        rows.append(dict(A=a, mean_abs_diff=np.mean(np.abs(pe - pm)), max_abs_diff=np.max(np.abs(pe - pm))))
    return pd.DataFrame(rows)


def main():
    R = build_first_free()
    E = R[R.dI == 0].reset_index(drop=True)
    lines = ['Assumption 1 (q), the noise term 2 S rho^2, and the moment-matched probit (Wilson first free choice)', '']
    with pd.option_context('display.width', 250, 'display.max_colwidth', 60):
        for Rs, info, label in [(E, False, f'games with equal prior observations (n={len(E):,})'), (R, True, f'all games (n={len(R):,})')]:
            T, D, y = run(Rs, info, label)
            lines += [f'[model variants, {label}; all with side bias B and lapse]', T.to_string(index=False, float_format=lambda v: f'{v:.3f}'), '']
            P = profile_rho(Rs, info, label)
            o1 = Rs[Rs.H == 1].non_greedy.mean(); o6 = Rs[Rs.H == 6].non_greedy.mean()
            lines += [f'[profile likelihood over rho (task value 8), {label}; observed P(ng) H1 = {o1:.3f}, H6 = {o6:.3f}]',
                      P.to_string(index=False, float_format=lambda v: f'{v:.3f}'), '']
        lines += ['[exact mixture vs moment-matched probit, tau = 1, no bias, q = U_D, rho = 8, equal games: |P_exact - P_mm|]',
                  exact_vs_mm_gap(as_dict(E)).to_string(index=False, float_format=lambda v: f'{v:.4f}')]
    ks, pk = KD[6]
    lines += ['', f'[H = 6: probability of mixture components with 1 + k <= 0] {pk[ks + 1 <= 0].sum():.3f}']
    out = '\n'.join(lines)
    (OUT / 'assumptions.txt').write_text(out, encoding='utf-8')
    print(out)
    import assumptions_gap
    assumptions_gap.main()


if __name__ == '__main__':
    main()
