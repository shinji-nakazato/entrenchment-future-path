import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from scipy.special import expit, logit

from config import RESULTS, RHO_WILSON
from gp_utils import S_factor

GAMMAS  = [0.7, 0.85, 1.0]
RHO_MUL = [0.5, 1.0, 1.5]
THETAS  = [0.0, 0.05, 0.1]
EPS     = 1e-4


def predict_nongreedy(dmu, Sigma, A, theta, gamma, rho2):
    w   = expit(theta * dmu)
    V_q = w * (1.0 - w) * (Sigma + dmu ** 2)
    S   = np.array([S_factor(gamma, int(a)) for a in A])
    kap = dmu / np.sqrt(Sigma + 2.0 * (S - 1.0) * V_q + 2.0 * S * rho2)
    return 1.0 - norm.cdf(kap)


def cox_slope(y, p):
    lp = logit(np.clip(p, EPS, 1.0 - EPS))
    X  = sm.add_constant(lp)
    res = sm.Logit(y, X).fit(disp=0)
    return float(res.params[1]), float(res.params[0])


def linear_slope(y, p, nbins=10):
    d = pd.DataFrame({'p': p, 'y': y})
    d['bin'] = pd.qcut(d['p'], nbins, duplicates='drop')
    cal = d.groupby('bin', observed=True).agg(pred=('p', 'mean'),
                                              emp=('y', 'mean')).reset_index()
    b, a = np.polyfit(cal['pred'].to_numpy(), cal['emp'].to_numpy(), 1)
    return float(b)


def main():
    R = pd.read_csv(RESULTS / 'wilson_pertrial.csv')
    R = R[R.t == 5].dropna(subset=['dmu', 'Sigma', 'A', 'non_greedy']).copy()
    dmu   = R['dmu'].to_numpy()
    Sigma = R['Sigma'].to_numpy()
    A     = R['A'].to_numpy()
    y     = R['non_greedy'].to_numpy().astype(float)
    obs_rate = float(y.mean())
    print(f'Wilson t=5: n={len(R)}, observed non-greedy rate = {obs_rate:.4f}')

    rows = []
    for g in GAMMAS:
        for rm in RHO_MUL:
            rho2 = (RHO_WILSON * rm) ** 2
            for th in THETAS:
                p = predict_nongreedy(dmu, Sigma, A, th, g, rho2)
                cslope, intercept = cox_slope(y, p)
                lslope = linear_slope(y, p)
                rows.append(dict(gamma=g, rho_mult=rm, theta=th,
                                 V_q_mean=float((expit(th * dmu) *
                                                 (1 - expit(th * dmu)) *
                                                 (Sigma + dmu ** 2)).mean()),
                                 pred_rate=float(p.mean()), obs_rate=obs_rate,
                                 calib_slope=lslope, cox_slope=cslope,
                                 calib_intercept=intercept))
    G = pd.DataFrame(rows)
    G.to_csv(RESULTS / 'calib_sensitivity_grid.csv', index=False)

    smin, smax = G['calib_slope'].min(), G['calib_slope'].max()
    print(f'\nLinear calibration slope range: [{smin:.3f}, {smax:.3f}]')
    print(f'Cox slope range: [{G.cox_slope.min():.3f}, {G.cox_slope.max():.3f}]')
    print(f'pred_rate range: [{G.pred_rate.min():.3f}, {G.pred_rate.max():.3f}]'
          f'  (observed = {obs_rate:.3f})')
    with pd.option_context('display.float_format', lambda v: f'{v:.3f}'):
        print('\n=== Full 27-point grid (sorted by |pred_rate - obs|) ===')
        Gs = G.assign(gap=(G.pred_rate - obs_rate).abs()).sort_values('gap')
        print(Gs[['gamma', 'rho_mult', 'theta', 'V_q_mean', 'pred_rate',
                  'calib_slope', 'cox_slope', 'gap']].to_string(index=False))

    print(f"\nGrid CSV: {RESULTS / 'calib_sensitivity_grid.csv'} "
          f"(Appendix E, Table calib-sens)")


if __name__ == '__main__':
    main()
