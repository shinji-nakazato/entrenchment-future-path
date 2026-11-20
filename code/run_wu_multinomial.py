import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import chi2

from config import RESULTS

OUT = RESULTS
SEED = 20240625
BINS_2D = [1, 2, 3, 4, 5, 6, 11, 16, 21, 31, 41]


def load(spec, ds):
    z = np.load(OUT / f'wu_mn_{spec}_{ds}.npz', allow_pickle=True)
    mu = z['mu'].astype(float)
    s2 = z['s2'].astype(float)
    A = z['A']
    n, D = mu.shape
    srt = np.sort(mu, axis=1)
    dmu = srt[:, -1] - srt[:, -2]
    if ds == 'Wu1D':
        abin = (A - 1).astype(int)
    else:
        abin = np.digitize(A, BINS_2D) - 1
    Vq = z['Vq']
    d = dict(mu=mu, s2=s2, sd=np.sqrt(s2), ch=z['chosen'].astype(int), A=A, logA=np.log(A),
             abin=abin, nbin=int(abin.max() + 1), Vq=Vq, zVq=(Vq - Vq.mean()) / Vq.std(),
             sbar2=s2.mean(axis=1), noise2=float(z['noise2']), dmu=dmu, D=D, n=n,
             subject=z['subject'], gdec=pd.qcut(pd.Series(dmu).rank(method='first'), 10, labels=False).to_numpy(),
             a1=(A == 1).astype(float))
    d['mu_c'] = mu - mu.mean(axis=1, keepdims=True)
    return d


def sub(d, m):
    out = {}
    for k, v in d.items():
        out[k] = v[m] if isinstance(v, np.ndarray) and v.shape[:1] == (d['n'],) else v
    out['n'] = int(m.sum()) if m.dtype == bool else len(m)
    return out


def loglik_rows(u, T, d, eps=None):
    z = u / T[:, None]
    lp = z[np.arange(len(T)), d['ch']] - logsumexp(z, axis=1)
    if eps is not None:
        lp = np.logaddexp(np.log(eps / d['D']), np.log1p(-eps) + lp)
    return lp


def W(d, lam, kap=1.0):
    return np.sqrt(d['sbar2'] + (d['A'] - 1.0) * lam * d['Vq'] + d['A'] * kap * d['noise2'])


def tauA(th0, th1, d):
    return np.exp(th0 + np.clip(th1, -5, 5) * d['logA'])


def free_bin(th, d):
    return np.exp(th[d['abin']])


def M_S0(th, d):
    return d['mu_c'], np.exp(th[0]) * np.ones(d['n'])


def M_S1(th, d):
    return d['mu_c'], tauA(th[0], th[1], d)


def M_S2(th, d):
    return d['mu_c'], free_bin(th, d)


def ucb_u(b, d):
    return d['mu_c'] + b[:, None] * d['sd']


def M_U0(th, d):
    return ucb_u(np.full(d['n'], th[0]), d), np.exp(th[1]) * np.ones(d['n'])


def M_U1(th, d):
    return ucb_u(th[0] + th[1] * d['logA'], d), tauA(th[2], th[3], d)


def M_U2(th, d):
    return ucb_u(th[0] + th[1] * d['logA'], d), free_bin(th[2:], d)


def M_F0(th, d):
    return d['mu_c'], np.exp(th[0]) * W(d, 0.0)


def M_FU(th, d):
    return d['mu_c'], np.exp(th[0]) * W(d, 1.0)


def M_FL(th, d):
    return d['mu_c'], np.exp(th[0]) * W(d, expit(th[1]))


def M_FK(th, d):
    return d['mu_c'], np.exp(th[0]) * W(d, expit(th[1]), np.exp(th[2]))


def M_FA(th, d):
    c = np.concatenate([[0.0], th[2:]])[d['abin']]
    return d['mu_c'], np.exp(th[0] + c) * W(d, expit(th[1]))


def M_UF(th, d):
    return ucb_u(th[0] + th[1] * d['logA'], d), np.exp(th[2]) * W(d, expit(th[3]))


def M_UFK(th, d):
    return ucb_u(th[0] + th[1] * d['logA'], d), np.exp(th[2]) * W(d, expit(th[3]), np.exp(th[4]))


def M_UF0(th, d):
    return ucb_u(th[0] + th[1] * d['logA'], d), np.exp(th[2]) * W(d, 0.0)


def M_V0(th, d):
    nb = d['nbin']
    return ucb_u(th[0] + th[1] * d['logA'], d), free_bin(th[2:2 + nb], d) * np.exp(
        np.concatenate([[0.0], th[2 + nb:2 + nb + 9]])[d['gdec']])


def M_V1(th, d):
    nb = d['nbin']
    base = free_bin(th[2:2 + nb], d) * np.exp(np.concatenate([[0.0], th[2 + nb:2 + nb + 9]])[d['gdec']])
    v = th[-2] * d['a1'] + th[-1] * (1.0 - d['a1'])
    return ucb_u(th[0] + th[1] * d['logA'], d), base * np.exp(v * d['zVq'])


def models(d):
    nb = d['nbin']
    t0 = float(np.log(np.std(d['mu_c']) + 1e-6))
    w0 = float(np.log(np.std(d['mu_c']) / np.median(W(d, 0.5)) + 1e-9))
    return [
        ('S0 softmax', 'horizon-free', M_S0, [t0]),
        ('S1 softmax, tau(A)', 'A-dependent', M_S1, [t0, 0.0]),
        ('S2 softmax, free tau per A-bin', 'descriptive', M_S2, [t0] * nb),
        ('U0 GP-UCB softmax (Wu 2018)', 'horizon-free', M_U0, [0.0, t0]),
        ('U1 GP-UCB, beta(A) + tau(A)', 'A-dependent', M_U1, [0.0, 0.0, t0, 0.0]),
        ('U2 GP-UCB, beta(A) + free tau per A-bin', 'descriptive', M_U2, [0.0, 0.0] + [t0] * nb),
        ('F0 future-path temp., lambda=0', 'future-path', M_F0, [w0]),
        ('FU future-path temp., q=U_D', 'future-path', M_FU, [w0]),
        ('FL future-path temp., calib. lambda', 'future-path', M_FL, [w0, 0.0]),
        ('FK future-path temp., lambda + rho^2 scale', 'future-path', M_FK, [w0, 0.0, 0.0]),
        ('FA future-path temp. x free per-A-bin scale', 'shape test', M_FA, [w0, 0.0] + [0.0] * (nb - 1)),
        ('UF0 UCB beta(A) + future-path temp., lambda=0', 'UCB + future-path', M_UF0, [0.0, 0.0, w0]),
        ('UF  UCB beta(A) + future-path temp., lambda', 'UCB + future-path', M_UF, [0.0, 0.0, w0, 0.0]),
        ('UFK UCB beta(A) + future-path temp., lambda + rho^2', 'UCB + future-path', M_UFK, [0.0, 0.0, w0, 0.0, 0.0]),
        ('V0 U2 + gap-decile temp.', 'gap-decile temperature control', M_V0, [0.0, 0.0] + [t0] * nb + [0.0] * 9),
        ('V1 V0 + V_q on temp. (A=1, A>=2)', 'partial contribution of V_q', M_V1, [0.0, 0.0] + [t0] * nb + [0.0] * 9 + [0.0, 0.0]),
    ]


LAPSE_MODELS = ('S1', 'S2', 'U1', 'U2', 'F0', 'FU', 'FL', 'FK', 'FA', 'UF0', 'UF ', 'UFK')


def make_obj(fn, d, lapse):
    def f(th):
        if lapse:
            u, T = fn(th[:-1], d)
            return -np.sum(loglik_rows(u, T, d, expit(th[-1])))
        u, T = fn(th, d)
        return -np.sum(loglik_rows(u, T, d))
    return f


def fit(fn, x0, d, lapse):
    f = make_obj(fn, d, lapse)
    x0 = np.asarray(x0, float)
    best = None
    starts = [x0] if len(x0) > 6 else [x0, x0 + 0.5, x0 - 0.5]
    for s in starts:
        r = minimize(f, s, method='L-BFGS-B', options={'maxiter': 3000})
        if len(x0) <= 5:
            r2 = minimize(f, r.x, method='Nelder-Mead',
                          options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 20000, 'maxfev': 20000})
            if r2.fun < r.fun:
                r = r2
        if best is None or r.fun < best.fun:
            best = r
    return best.x, -best.fun


def bin_logA(d):
    nb = d['nbin']
    return np.array([d['logA'][d['abin'] == b].mean() for b in range(nb)])


def warm(code, B, d):
    nb = d['nbin']
    L = bin_logA(d)
    g = lambda c: B.get(c)
    if code == 'S1' and g('S0') is not None:
        return [g('S0')[0], 0.0]
    if code == 'S2' and g('S1') is not None:
        return list(g('S1')[0] + g('S1')[1] * L)
    if code == 'U1' and g('U0') is not None:
        return [g('U0')[0], 0.0, g('U0')[1], 0.0]
    if code == 'U2' and g('U1') is not None:
        u = g('U1')
        return [u[0], u[1]] + list(u[2] + u[3] * L)
    if code == 'FL' and g('F0') is not None:
        return [g('F0')[0], -10.0]
    if code == 'FK' and g('FL') is not None:
        return list(g('FL')) + [0.0]
    if code == 'FA' and g('FL') is not None:
        return list(g('FL')) + [0.0] * (nb - 1)
    if code == 'UF' and g('UF0') is not None:
        return list(g('UF0')) + [-10.0]
    if code == 'UFK' and g('UF') is not None:
        return list(g('UF')) + [0.0]
    if code == 'V0' and g('U2') is not None:
        return list(g('U2')) + [0.0] * 9
    if code == 'V1' and g('V0') is not None:
        return list(g('V0')) + [0.0, 0.0]
    return None


def fit_multi(fn, starts, d, lapse):
    best = None
    for s0 in starts:
        th, ll = fit(fn, s0, d, lapse)
        if best is None or ll > best[1]:
            best = (th, ll)
    return best


def folds_of(d):
    uniq = np.unique(d['subject'])
    rng = np.random.default_rng(SEED)
    fm = {s: i % 5 for i, s in enumerate(rng.permutation(uniq))}
    return np.array([fm[s] for s in d['subject']])


def run(spec, ds):
    d = load(spec, ds)
    folds = folds_of(d)
    null = d['n'] * np.log(d['D'])
    rows, prof = [], []
    B = {}
    for name, kind, fn, x0 in models(d):
        code = name.split()[0]
        variants = [False]
        if name.startswith(LAPSE_MODELS):
            variants.append(True)
        for lapse in variants:
            if not lapse:
                starts = [x0]
                w = warm(code, B, d)
                if w is not None:
                    starts.append(w)
                th, ll = fit_multi(fn, starts, d, False)
                B[code] = th
                xx = x0
            else:
                base = B[code]
                starts = [list(base) + [e] for e in (-6.0, -2.0, 0.0, 1.0)]
                th, ll = fit_multi(fn, starts, d, True)
                xx = list(x0) + [0.0]
            k = len(xx)
            cv = 0.0
            for f in range(5):
                tr, te = folds != f, folds == f
                dtr, dte = sub(d, tr), sub(d, te)
                thf, _ = fit(fn, th, dtr, lapse)
                cv += -make_obj(fn, dte, lapse)(thf)
            ex = {}
            pars = th[:-1] if lapse else th
            if name.startswith(('FL', 'FK', 'FA')):
                ex['lambda'] = float(expit(pars[1]))
            if name.startswith(('UF ', 'UFK')):
                ex['lambda'] = float(expit(pars[3]))
            if name.startswith('FK'):
                ex['rho2_scale'] = float(np.exp(pars[2]))
            if name.startswith('UFK'):
                ex['rho2_scale'] = float(np.exp(pars[4]))
            if name.startswith(('U1', 'U2', 'UF', 'V')):
                ex['beta0'], ex['beta_logA'] = float(pars[0]), float(pars[1])
            if name.startswith('V1'):
                ex['vq_A1'], ex['vq_A2plus'] = float(pars[-2]), float(pars[-1])
            if lapse:
                ex['lapse_rate'] = float(expit(th[-1]))
            rows.append(dict(spec=spec, dataset=ds, model=name + (' + lapse' if lapse else ''), kind=kind,
                             lapse=lapse, k=k, n=d['n'], LL=ll, AIC=2 * k - 2 * ll,
                             BIC=k * np.log(d['n']) - 2 * ll, McFaddenR2=1 - (-ll) / null,
                             cv_test_LL=cv, theta=' '.join(f'{v:.4f}' for v in th), **ex))
            print(spec, ds, rows[-1]['model'], f"LL={ll:.1f}", flush=True)
            pd.DataFrame(rows).to_csv(OUT / f'wu_mn_models_{spec}_{ds}.csv', index=False)
    return rows


def vq_lr(T):
    out = []
    for (spec, ds), g in T.groupby(['spec', 'dataset']):
        a = g[g.model.str.startswith('V0')].iloc[0]
        b = g[g.model.str.startswith('V1')].iloc[0]
        lr = 2 * (b.LL - a.LL)
        out.append(dict(spec=spec, dataset=ds, LR_df2=lr, p=chi2.sf(lr, 2),
                        vq_A1=b.vq_A1, vq_A2plus=b.vq_A2plus))
        for base, full, lab in [('FL', 'FA', 'free per-A-bin scale beyond future-path temp. (lambda)'),
                                ('U1', 'U2', 'free per-A-bin tau beyond tau(A) (UCB)'),
                                ('UF0', 'UF ', 'V_q term (lambda) beyond rho^2-only future-path temp. (UCB base)')]:
            gb = g[(g.model.str.startswith(base)) & (~g.lapse)].iloc[0]
            gf = g[(g.model.str.startswith(full)) & (~g.lapse)].iloc[0]
            lr = 2 * (gf.LL - gb.LL)
            out.append(dict(spec=spec, dataset=ds, test=lab, LR=lr, df=int(gf.k - gb.k),
                            p=chi2.sf(lr, max(int(gf.k - gb.k), 1))))
    return pd.DataFrame(out)


def main():
    spec, ds = sys.argv[1], sys.argv[2]
    rows = run(spec, ds)
    pd.DataFrame(rows).to_csv(OUT / f'wu_mn_models_{spec}_{ds}.csv', index=False)


def report():
    T = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob('wu_mn_models_*_*.csv'))], ignore_index=True)
    T['tau_A_exponent'] = np.where(T.model.str.startswith('U1 '), T.theta.str.split().str[3].astype(float), np.nan)
    T.to_csv(OUT / 'wu_mn_models.csv', index=False)
    lines = ['Wu, multinomial choice over all cells (softmax family)',
             'Specs: wu = Wu specification (Wu et al. 2018: rewards (y-50)/100, m0=0, sf2=1, noise variance 1e-4); priorvar1 = prior-variance-1 specification (m0=25, sf2=1, noise sd 5); priorvar300 = prior-variance-300 specification (m0=50, sf2=300, noise sd 1)',
             'Future-path temperature: tau * sqrt(mean_x sigma^2(x) + (A-1) lambda V_q + A kappa rho^2).',
             'Lapse: P = e/|D| + (1-e) P_model. CV = subject-level 5-fold test LL. Free models use 10 A-bins.', '']
    with pd.option_context('display.width', 260, 'display.max_colwidth', 55,
                           'display.float_format', lambda v: f'{v:.3f}'):
        for (spec, ds), g in T.groupby(['spec', 'dataset'], sort=False):
            g = g.copy()
            g['dAIC'] = g.AIC - g.AIC.min()
            g['dBIC'] = g.BIC - g.BIC.min()
            g['dCV'] = g.cv_test_LL - g.cv_test_LL.max()
            cols = ['model', 'k', 'LL', 'AIC', 'dAIC', 'dBIC', 'cv_test_LL', 'dCV', 'McFaddenR2'] + \
                   [c for c in ['lambda', 'rho2_scale', 'beta0', 'beta_logA', 'tau_A_exponent', 'vq_A1', 'vq_A2plus', 'lapse_rate']
                    if c in g and g[c].notna().any()]
            lines += [f'[{spec} | {ds} | n={g.n.iloc[0]}]', g[cols].to_string(index=False, na_rep=''), '']
        lines += ['[nested likelihood-ratio tests]', vq_lr(T).to_string(index=False)]
    out = '\n'.join(lines)
    (OUT / 'wu_mn_models.txt').write_text(out, encoding='utf-8')
    print(out)


def run_all():
    import subprocess
    procs = [subprocess.Popen([sys.executable, __file__, s, d]) for s in ('wu', 'priorvar1', 'priorvar300')
             for d in ('Wu1D', 'Wu2D')]
    codes = [p.wait() for p in procs]
    if any(codes):
        sys.exit(max(codes))
    report()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        run_all()
    elif sys.argv[1] == 'report':
        report()
    else:
        main()
