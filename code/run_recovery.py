import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

import wilson_choice as m1
from wilson_choice import RHO2, nll, fit, as_dict, build_first_free

from config import RESULTS, N_WORKERS, WILSON_CSV, WILSON_FORCED_TRIALS

OUT = RESULTS
SEED = 20261026
_DIRECT = Path(sys.argv[0]).stem == 'run_recovery'
N_SIM = int(sys.argv[2]) if _DIRECT and len(sys.argv) > 2 else 100
N_PAR = int(sys.argv[3]) if _DIRECT and len(sys.argv) > 3 else 200
NPROC = N_WORKERS

FAM_NOLAPSE = [m for m in m1.REPLICATE if m[0] != 'Thompson (tempered)']
FAM_NOLAPSE_NAMES = {
    'softmax (mu)': 'softmax', 'UCB-softmax': 'UCB', 'future-path, q=U_D': 'FP U_D',
    'future-path, calib. q': 'FP calib', 'softmax (per-H tau)': 'softmax tau_H',
}
LAPSE_KEEP = ['C0+e', 'C1+e', "C1'+e", 'C4+e', 'F1+e', 'F2u+e', 'F0+e']
FAM_LAPSE = [m for m in m1.LAPSE if m[0].split(' ')[0] in LAPSE_KEEP]

G = {}


FAMS = {'nolapse': FAM_NOLAPSE, 'lapse': FAM_LAPSE}


def _init(D, tag):
    G['D'] = D
    G['fam'] = FAMS.get(tag)


def _refit_all(task):
    gi, s, y, starts = task
    D, fam = G['D'], G['fam']
    out = []
    for j, (name, kind, fn, x0) in enumerate(fam):
        best_ll = -np.inf
        for st in (starts[j], x0):
            th, ll = fit_fast(fn, st, D, y)
            if ll > best_ll:
                best_ll, best_th = ll, th
        k = len(x0)
        n = len(y)
        out.append(dict(gen=gi, sim=s, fit=j, LL=best_ll, AIC=2 * k - 2 * best_ll,
                        BIC=k * np.log(n) - 2 * best_ll))
    return out


def fit_fast(fn, x0, D, y, bounds=None):
    f = lambda th: nll(fn(th, D), y)
    x0 = np.asarray(x0, float)
    r1 = minimize(f, x0, method='L-BFGS-B', bounds=bounds)
    r2 = minimize(f, r1.x, method='Nelder-Mead',
                  options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 20000, 'maxfev': 20000})
    r = r2 if r2.fun <= r1.fun else r1
    if bounds is not None:
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        if np.any(r.x < lo) or np.any(r.x > hi):
            r = r1
    return r.x, -r.fun


def model_recovery(fam, short, D, y_real, label, pool_size=NPROC):
    real = []
    for name, kind, fn, x0 in fam:
        th, ll = fit(fn, x0, D, y_real)
        real.append((th, ll))
    rng = np.random.default_rng(SEED + len(fam))
    tasks = []
    for gi, (name, kind, fn, x0) in enumerate(fam):
        p_left = fn(real[gi][0], D)
        for s in range(N_SIM):
            y = (rng.random(len(p_left)) < p_left).astype(float)
            tasks.append((gi, s, y, [r[0] for r in real]))
    t0 = time.time()
    with Pool(pool_size, initializer=_init, initargs=(D, label.split(':')[0])) as pool:
        res = pool.map(_refit_all, tasks, chunksize=2)
    R = pd.DataFrame([r for rr in res for r in rr])
    names = [short(m[0]) for m in fam]
    R['gen_name'] = R.gen.map(dict(enumerate(names)))
    R['fit_name'] = R.fit.map(dict(enumerate(names)))
    real_tab = pd.DataFrame(dict(model=names, LL=[r[1] for r in real],
                                 k=[len(m[3]) for m in fam]))
    real_tab['AIC'] = 2 * real_tab.k - 2 * real_tab.LL
    real_tab['BIC'] = real_tab.k * np.log(len(y_real)) - 2 * real_tab.LL
    real_tab['theta'] = [' '.join(f'{v:.4f}' for v in r[0]) for r in real]
    return R, real_tab, time.time() - t0


def confusion(R, crit, names):
    W = R.loc[R.groupby(['gen', 'sim'])[crit].idxmin()]
    C = pd.crosstab(W.gen_name, W.fit_name, normalize='index').reindex(index=names, columns=names).fillna(0)
    return C


def inversion(C):
    M = C.to_numpy()
    J = M / M.sum(axis=0, keepdims=True)
    return pd.DataFrame(J, index=C.index, columns=C.columns)


def pair_stats(R, a, b, names, obs):
    rows = []
    for g in names:
        sub = R[R.gen_name == g]
        A = sub[sub.fit_name == a].set_index('sim').AIC
        B = sub[sub.fit_name == b].set_index('sim').AIC
        d = (A - B)
        rows.append(dict(generator=g, mean_dAIC=d.mean(), sd_dAIC=d.std(),
                         q05=d.quantile(.05), q95=d.quantile(.95),
                         P_b_better=(d > 0).mean(), P_dAIC_ge_obs=(d >= obs).mean()))
    return pd.DataFrame(rows)


def p_fp_tl(th, X):
    tau, lam = np.exp(th[0]), th[1]
    W = np.sqrt(X['Sigma'] + 2.0 * (X['A'] - 1.0) * lam * X['V_q'] + 2.0 * X['A'] * RHO2)
    return 1.0 - norm.cdf(X['dmu'] / (tau * W))


def S_gamma(g, A):
    g2 = g * g
    if abs(1.0 - g2) < 1e-10:
        return A
    return (1.0 - g2 ** A) / (1.0 - g2)


def Vq_theta(theta, dmu, Sigma):
    w = expit(theta * dmu)
    return w * (1.0 - w) * (Sigma + dmu ** 2)


def p_fp_theta(th, X, gamma=1.0):
    tau, theta = np.exp(th[0]), th[1]
    g = th[2] if len(th) > 2 else gamma
    S = S_gamma(g, X['A'])
    W = np.sqrt(X['Sigma'] + 2.0 * (S - 1.0) * Vq_theta(theta, X['dmu'], X['Sigma']) + 2.0 * S * RHO2)
    return 1.0 - norm.cdf(X['dmu'] / (tau * W))


PAR_MODELS = {
    'P1 FP (tau, lambda), t=5': dict(fn=p_fp_tl, names=['log_tau', 'lambda'],
                                     bounds=[(-5, 5), (0, 1)],
                                     draw=lambda r: [r.uniform(-1.5, 0.5), r.uniform(0, 1)]),
    'P2 FP (tau, theta), gamma=1, t=5': dict(fn=p_fp_theta, names=['log_tau', 'theta'],
                                             bounds=[(-5, 5), (0, 2)],
                                             draw=lambda r: [r.uniform(-1.5, 0.5), r.uniform(0, 0.3)]),
    'P3 FP (tau, theta, gamma), t=5': dict(fn=p_fp_theta, names=['log_tau', 'theta', 'gamma'],
                                           bounds=[(-5, 5), (0, 2), (0.05, 1)],
                                           draw=lambda r: [r.uniform(-1.5, 0.5), r.uniform(0, 0.3),
                                                           r.uniform(0.5, 1.0)]),
}


def _par_task(task):
    key, i, true, y, X = task
    spec = PAR_MODELS[key]
    best = None
    rng = np.random.default_rng(i)
    starts = [np.array(true, float)] + [np.array(spec['draw'](rng)) for _ in range(3)]
    for st in starts:
        r = minimize(lambda th: nll(spec['fn'](th, X), y), st, method='L-BFGS-B', bounds=spec['bounds'])
        if best is None or r.fun < best.fun:
            best = r
    return dict(model=key, i=i, **{f'true_{n}': v for n, v in zip(spec['names'], true)},
                **{f'est_{n}': v for n, v in zip(spec['names'], best.x)}, LL=-best.fun)


def real_fit_par(key, X, y):
    spec = PAR_MODELS[key]
    rng = np.random.default_rng(1)
    best = None
    for _ in range(12):
        st = np.array(spec['draw'](rng))
        r = minimize(lambda th: nll(spec['fn'](th, X), y), st, method='L-BFGS-B', bounds=spec['bounds'])
        if best is None or r.fun < best.fun:
            best = r
    return best.x, -best.fun


def param_recovery_t5(R5):
    X = dict(dmu=R5['dmu'].to_numpy(float), Sigma=R5['Sigma'].to_numpy(float),
             V_q=R5['V_q'].to_numpy(float), A=R5['A'].to_numpy(float))
    y_real = R5['non_greedy'].to_numpy(float)
    rng = np.random.default_rng(SEED)
    tasks, reals = [], {}
    for key, spec in PAR_MODELS.items():
        reals[key] = real_fit_par(key, X, y_real)
        for i in range(N_PAR):
            true = spec['draw'](rng)
            p = spec['fn'](np.array(true), X)
            y = (rng.random(len(p)) < p).astype(float)
            tasks.append((key, i, true, y, X))
    with Pool(NPROC) as pool:
        res = pool.map(_par_task, tasks, chunksize=4)
    return pd.DataFrame(res), reals


def build_games():
    df = pd.read_csv(WILSON_CSV).dropna(subset=['m1', 'm2'])
    df = df[df.gameLength.isin([5, 10])].reset_index(drop=True)
    F = WILSON_FORCED_TRIALS
    c = df[[f'c{t}' for t in range(1, F + 1)]].to_numpy()
    r = df[[f'r{t}' for t in range(1, F + 1)]].to_numpy(float)
    n1 = (c == 1).sum(1).astype(float); n2 = F - n1
    s1 = np.where(c == 1, r, 0).sum(1); s2 = np.where(c == 2, r, 0).sum(1)
    keep = (n1 > 0) & (n2 > 0)
    return dict(n1=n1[keep], n2=n2[keep], s1=s1[keep], s2=s2[keep],
                H=df.gameLength.to_numpy()[keep] - F,
                m1=df.m1.to_numpy(float)[keep], m2=df.m2.to_numpy(float)[keep],
                choices=df[[f'c{t}' for t in range(F + 1, 11)]].to_numpy(float)[keep],
                rewards=df[[f'r{t}' for t in range(F + 1, 11)]].to_numpy(float)[keep])


def game_trials(Gm, tau, theta, gamma, rng=None):
    n1, n2, s1, s2 = Gm['n1'].copy(), Gm['n2'].copy(), Gm['s1'].copy(), Gm['s2'].copy()
    rows = {k: [] for k in ('dmu', 'Sigma', 'A', 'y')}
    for k in range(6):
        act = Gm['H'] > k
        A = (Gm['H'] - k).astype(float)
        mu1, mu2 = s1 / n1, s2 / n2
        dmu = np.abs(mu1 - mu2)
        Sigma = RHO2 / n1 + RHO2 / n2
        greedy1 = mu1 >= mu2
        if rng is None:
            ch = Gm['choices'][:, k]
            rw = Gm['rewards'][:, k]
            ng = np.where(greedy1, ch != 1, ch != 2).astype(float)
        else:
            p = p_fp_theta(np.array([np.log(tau), theta, gamma]), dict(dmu=dmu, Sigma=Sigma, A=A))
            ng = (rng.random(len(p)) < p).astype(float)
            ch = np.where(greedy1, np.where(ng == 1, 2, 1), np.where(ng == 1, 1, 2)).astype(float)
            rw = np.where(ch == 1, Gm['m1'], Gm['m2']) + RHO2 ** 0.5 * rng.standard_normal(len(ch))
        rows['dmu'].append(dmu[act]); rows['Sigma'].append(Sigma[act])
        rows['A'].append(A[act]); rows['y'].append(ng[act])
        a1 = act & (ch == 1); a2 = act & (ch == 2)
        n1 = n1 + a1; s1 = s1 + np.where(a1, rw, 0)
        n2 = n2 + a2; s2 = s2 + np.where(a2, rw, 0)
    return {k: np.concatenate(v) for k, v in rows.items()}


BOUNDS_ALL = [(-5, 5), (0, 2), (0.05, 1)]


def fit_all_trials(X, y, starts):
    best = None
    for st in starts:
        r = minimize(lambda th: nll(p_fp_theta(th, X), y), st, method='L-BFGS-B', bounds=BOUNDS_ALL)
        if best is None or r.fun < best.fun:
            best = r
    return best.x, -best.fun


def _all_task(task):
    i, true = task
    rng = np.random.default_rng(10_000 + i)
    X = game_trials(G['games'], np.exp(true[0]), true[1], true[2], rng)
    y = X.pop('y')
    starts = [np.array(true)] + [np.array([rng.uniform(-1.5, 0.5), rng.uniform(0, .3), rng.uniform(.5, 1)])
                                 for _ in range(3)]
    th, ll = fit_all_trials(X, y, starts)
    return dict(model='P4 FP (tau, theta, gamma), all free trials', i=i,
                true_log_tau=true[0], true_theta=true[1], true_gamma=true[2],
                est_log_tau=th[0], est_theta=th[1], est_gamma=th[2], LL=ll, n=len(y))


def _init_games(games):
    G['games'] = games


def param_recovery_all():
    games = build_games()
    X = game_trials(games, None, None, None, None)
    y = X.pop('y')
    rng = np.random.default_rng(SEED + 7)
    starts = [np.array([rng.uniform(-1.5, 0.5), rng.uniform(0, .3), rng.uniform(.5, 1)]) for _ in range(12)]
    real = fit_all_trials(X, y, starts)
    trues = [[rng.uniform(-1.5, 0.5), rng.uniform(0, 0.3), rng.uniform(0.5, 1.0)] for _ in range(N_PAR)]
    with Pool(NPROC, initializer=_init_games, initargs=(games,)) as pool:
        res = pool.map(_all_task, list(enumerate(trues)), chunksize=2)
    return pd.DataFrame(res), real, len(y)


F2U_NAMES = ['log_tau', 'A_H1', 'A_H6', 'B', 'logit_e']


def _f2u_task(task):
    i, true, y = task
    fn = [m for m in m1.LAPSE if m[0].startswith('F2u+e')][0][2]
    th, ll = fit(fn, true, G['D'], y)
    return dict(model='P5 F2u+e (future-path + information bonus)', i=i, **{f'true_{n}': v for n, v in zip(F2U_NAMES, true)},
                **{f'est_{n}': v for n, v in zip(F2U_NAMES, th)}, LL=ll)


def param_recovery_f2u(D, y_real):
    name, kind, fn, x0 = [m for m in m1.LAPSE if m[0].startswith('F2u+e')][0]
    th_hat, _ = fit(fn, x0, D, y_real)
    rng = np.random.default_rng(SEED + 5)
    tasks = []
    for i in range(N_PAR):
        true = th_hat + rng.normal(0, [0.3, 1.0, 2.0, 0.3, 0.4])
        p = fn(true, D)
        tasks.append((i, true, (rng.random(len(p)) < p).astype(float)))
    with Pool(NPROC, initializer=_init, initargs=(D, None)) as pool:
        res = pool.map(_f2u_task, tasks, chunksize=2)
    return pd.DataFrame(res), th_hat


def summarize_par(P):
    rows = []
    for key, g in P.groupby('model', sort=False):
        for c in [c for c in g.columns if c.startswith('true_')]:
            n = c[5:]
            t, e = g[c], g['est_' + n]
            if t.isna().all():
                continue
            rows.append(dict(model=key, param=n, r=np.corrcoef(t, e)[0, 1],
                             bias=(e - t).mean(), rmse=np.sqrt(((e - t) ** 2).mean()),
                             sd_true=t.std(), med_abs_err=(e - t).abs().median()))
    return pd.DataFrame(rows)


def ftab(T, fmt='{:.3f}'):
    with pd.option_context('display.width', 250, 'display.max_colwidth', 80):
        return T.to_string(float_format=lambda v: fmt.format(v))


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'
    R = build_first_free()
    D = as_dict(R)
    y_left = R['left'].to_numpy(float)
    lines = []

    if part in ('model', 'all'):
        for fam, short, label in [
            (FAM_NOLAPSE, lambda s: FAM_NOLAPSE_NAMES[s], 'nolapse: models without side bias or lapse (Table H.1, top panel)'),
            (FAM_LAPSE, lambda s: s.split(' ')[0], 'lapse: models with side bias B and lapse e (Table G.1)'),
        ]:
            lines = []
            Rm, real_tab, sec = model_recovery(fam, short, D, y_left, label)
            tag = label.split(':')[0]
            Rm.to_csv(OUT / f'recovery_model_{tag}.csv', index=False)
            names = list(real_tab.model)
            lines += [f'=== Model recovery {label}: {N_SIM} sims per generator, n={len(y_left):,} '
                      f'({sec / 60:.1f} min) ===', '[real-data fits]', ftab(real_tab, '{:.2f}'), '']
            for crit in ('AIC', 'BIC'):
                C = confusion(Rm, crit, names)
                lines += [f'[confusion P(best fit | generator), {crit}; rows = generator]', ftab(C), '',
                          f'[inversion P(generator | best fit), {crit}, uniform prior; columns sum to 1]',
                          ftab(inversion(C)), '']
            if tag == 'nolapse':
                obs = float(real_tab.set_index('model').AIC['FP calib'] -
                            real_tab.set_index('model').AIC['softmax tau_H'])
                lines += [f'[key pair: dAIC = AIC(FP calib) - AIC(softmax tau_H); observed {obs:+.1f}]',
                          ftab(pair_stats(Rm, 'FP calib', 'softmax tau_H', names, obs), '{:.2f}'), '']
            else:
                rt = real_tab.set_index('model').AIC
                for a, b in [("C1'+e", 'F2u+e'), ('C1+e', 'F2u+e'), ('C4+e', 'F2u+e'), ('F0+e', 'F2u+e')]:
                    obs = float(rt[a] - rt[b])
                    lines += [f'[pair: dAIC = AIC({a}) - AIC({b}); observed {obs:+.2f}]',
                              ftab(pair_stats(Rm, a, b, names, obs), '{:.2f}'), '']
            (OUT / f'recovery_model_{tag}.txt').write_text('\n'.join(lines), encoding='utf-8')
            print('\n'.join(lines), flush=True)

    if part in ('param', 'all'):
        plines = []
        P1, reals = param_recovery_t5(R.assign())
        plines += ['=== Parameter recovery, Wilson t=5 (n=19,200), non-greedy likelihood, '
                   f'{N_PAR} parameter sets each ===', '[real-data fits]']
        for k, (th, ll) in reals.items():
            plines.append(f'  {k}: ' + ' '.join(f'{v:.4f}' for v in th) + f'  LL={ll:.2f}')
        P4, real4, n4 = param_recovery_all()
        plines.append(f'  P4 FP (tau, theta, gamma), all free trials (n={n4:,}): '
                      + ' '.join(f'{v:.4f}' for v in real4[0]) + f'  LL={real4[1]:.2f}')
        P5, th5 = param_recovery_f2u(D, y_left)
        plines.append('  P5 F2u+e real MLE: ' + ' '.join(f'{v:.4f}' for v in th5))
        P = pd.concat([P1, P4, P5], ignore_index=True)
        P.to_csv(OUT / 'recovery_param.csv', index=False)
        plines += ['', '[recovery summary: correlation true-est, bias, RMSE]', ftab(summarize_par(P)), '']
        th = P[P.true_theta.notna() & (P.true_theta <= 0.2)]
        mae = th.assign(ae=(th.est_theta - th.true_theta).abs()).groupby('model', sort=False).ae.median()
        plines += ['[theta recovery for true theta <= 0.2: median absolute error]'] + \
                  [f'  {k}: {v:.4f}' for k, v in mae.items()] + ['']
        for key in ['P3 FP (tau, theta, gamma), t=5', 'P4 FP (tau, theta, gamma), all free trials']:
            g = P[P.model == key]
            if len(g):
                plines += [f'[{key}: correlation among estimates (ridge diagnostic)]',
                           ftab(g[['est_log_tau', 'est_theta', 'est_gamma']].corr()), '',
                           f'  share gamma estimate at bound (<0.06 or >0.999): '
                           f'{((g.est_gamma < 0.06) | (g.est_gamma > 0.999)).mean():.3f}', '']
        (OUT / 'recovery_param.txt').write_text('\n'.join(plines), encoding='utf-8')
        print('\n'.join(plines), flush=True)


if __name__ == '__main__':
    main()
