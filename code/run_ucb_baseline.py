import json
import sys
from pathlib import Path
from multiprocessing import Pool

import ucb_horizon

import numpy as np
import pandas as pd

from config import (RESULTS, WU_PARAMS_1D, WU_PARAMS_2D, WU1D_CSV, WU2D_CSV, WU_LSCALE_1D, WU_LSCALE_2D, WU_GRIDSIDE_1D, WU_GRIDSIDE_2D, RHO_WU_PRIORVAR1)
from wu_envs import load_envs, ENV1D, ENV2D
from gp_utils import grid, posterior

from config import N_WORKERS

OUT = RESULTS
N_REP = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEED = 20261030

AGENTS = {
    'Wu1D': [('GP-UCB, own params (Wu exp. 1 best)', dict(kernel='RBF', local=False)),
             ('GP*-UCB localized, own params', dict(kernel='LRBF', local=True)),
             ('GP-UCB, own params pooled over horizons', dict(kernel='RBF', local=False, pool=True)),
             ('GP-UCB, median params (0.554, 0.506, 0.015)', dict(ls=0.554, beta=0.506, tau=0.015, local=False)),
             ('random', dict(kind='random')), ('random unsampled', dict(kind='novel'))],
    'Wu2D': [('GP*-UCB localized, own params (Wu exp. 2 best)', dict(kernel='LRBF', local=True)),
             ('GP*-UCB localized, own params pooled over horizons', dict(kernel='LRBF', local=True, pool=True)),
             ('GP-UCB, own params', dict(kernel='RBF', local=False)),
             ('GP*-UCB localized, median params (0.766, 0.488, 0.087)', dict(ls=0.766, beta=0.488, tau=0.087, local=True)),
             ('random', dict(kind='random')), ('random unsampled', dict(kind='novel'))],
}


def load_params():
    out = {}
    for label, f in [('Wu1D', WU_PARAMS_1D), ('Wu2D', WU_PARAMS_2D)]:
        d = pd.read_csv(f)
        d = d[d.acq == 'UCB']
        g = d.groupby(['kernel', 'participant', 'horizon'])[['lambda', 'beta', 'tau']].median()
        out[label] = {k: v.to_dict() for k, v in g.iterrows()}
        gp = d.groupby(['kernel', 'participant'])[['lambda', 'beta', 'tau']].median()
        out[label].update({(k[0], k[1], 'pooled'): v.to_dict() for k, v in gp.iterrows()})
    return out
REF = {'Wu1D': (1, WU_GRIDSIDE_1D, WU_LSCALE_1D), 'Wu2D': (2, WU_GRIDSIDE_2D, WU_LSCALE_2D)}


def coords(cell, ndim, side):
    return np.array([cell]) if ndim == 1 else np.array([cell % side, cell // side])


def ref_posterior(Xc, z, Xg, ls):
    return posterior(Xc, np.asarray(z, float), Xg, ls, 25.0, 1.0, RHO_WU_PRIORVAR1 ** 2)


def wu_posterior(Xc, raw, Xg, ls):
    return posterior(Xc, (np.asarray(raw, float) - 50.0) / 100.0, Xg, ls, 0.0, 1.0, 1e-4)


def ref_cols(mu, cv, ch, env, suffix):
    star = int(np.argmax(mu))
    return {f'non_greedy{suffix}': int(ch != star), f'regret_inc{suffix}': int(env[ch] < env[star]),
            f'greedy_is_opt{suffix}': int(star == int(np.argmax(env))), f'dval{suffix}': float(env[ch] - env[star]),
            f'gap{suffix}': gap_of(mu, cv)}


def gap_of(mu, cov):
    star = int(np.argmax(mu))
    order = np.argsort(mu)[::-1]
    alt = int(order[1]) if order[0] == star else int(order[0])
    Sig = max(float(cov[star, star] + cov[alt, alt] - 2 * cov[star, alt]), 1e-8)
    return float(mu[star] - mu[alt]) / np.sqrt(Sig)


def agent_choice(spec, Xc, z, Xg, hist, prev, ndim, rng, D):
    kind = spec.get('kind')
    if kind == 'random':
        return int(rng.integers(D))
    if kind == 'novel':
        cand = np.setdiff1d(np.arange(D), hist)
        return int(rng.choice(cand)) if len(cand) else int(rng.integers(D))
    yn = (np.asarray(z, float) - 50.0) / 100.0
    mu, cov = posterior(Xc, yn, Xg, spec['ls'], 0.0, 1.0, 0.0001)
    u = mu + spec['beta'] * np.sqrt(np.abs(np.diag(cov)))
    u = np.clip(np.where(u <= 0, 0.0001, u), None, 100.0)
    if spec['local']:
        dist = np.abs(Xg - Xg[prev]).sum(axis=1)
        u = u * np.where(dist == 0, 1.0, 1.0 / np.where(dist == 0, 1.0, dist))
    v = u / spec['tau']
    p = np.exp(v - v.max())
    p /= p.sum()
    return int(rng.choice(D, p=p))


def simulate(task):
    label, sid, pidx, r, env, T, first, a, b, seed, P = task
    ndim, side, ls_ref = REF[label]
    Xg = grid(ndim, side)
    D = len(Xg)
    rng = np.random.default_rng(seed)
    xg = int(np.argmax(env))
    out = []
    for ai, (aname, spec) in enumerate(AGENTS[label]):
        if 'kernel' in spec:
            pp = P.get((spec['kernel'], pidx, 'pooled' if spec.get('pool') else T - 1))
            if pp is None:
                continue
            spec = dict(ls=pp['lambda'], beta=pp['beta'], tau=pp['tau'], local=spec['local'])
        for rep in range(N_REP):
            cells = [first]
            raw = [env[first] * 100 + rng.standard_normal()]
            for t in range(1, T):
                Xc = Xg[cells].astype(float)
                z = [a + b * v for v in raw]
                mu, cv = ref_posterior(Xc, z, Xg, ls_ref)
                star = int(np.argmax(mu))
                gap = gap_of(mu, cv)
                ch = agent_choice(spec, Xc, raw, Xg, cells, cells[-1], ndim, rng, D)
                muw, cvw = wu_posterior(Xc, raw, Xg, ls_ref)
                out.append(dict(dataset=label, agent=aname, subject=sid, round=r, rep=rep, t=t + 1, T=T, A=T - t,
                                non_greedy=int(ch != star), regret_inc=int(env[ch] < env[star]),
                                found_opt=int(ch == xg), greedy_is_opt=int(star == xg),
                                dval=float(env[ch] - env[star]), novel=int(ch not in cells),
                                repeat=int(ch == cells[-1]), gap=gap, **ref_cols(muw, cvw, ch, env, '_wu')))
                cells.append(ch)
                raw.append(env[ch] * 100 + rng.standard_normal())
    return pd.DataFrame(out)


def human(task):
    label, sid, r, env, cells, z, rawh = task
    ndim, side, ls_ref = REF[label]
    Xg = grid(ndim, side)
    xg = int(np.argmax(env))
    out = []
    for t in range(1, len(cells)):
        mu, cv = ref_posterior(Xg[cells[:t]].astype(float), z[:t], Xg, ls_ref)
        star = int(np.argmax(mu))
        ch = cells[t]
        muw, cvw = wu_posterior(Xg[cells[:t]].astype(float), rawh[:t], Xg, ls_ref)
        out.append(dict(dataset=label, agent='HUMAN', subject=sid, round=r, rep=0, t=t + 1, T=len(cells), A=len(cells) - t,
                        non_greedy=int(ch != star), regret_inc=int(env[ch] < env[star]), found_opt=int(ch == xg),
                        greedy_is_opt=int(star == xg), dval=float(env[ch] - env[star]),
                        novel=int(ch not in cells[:t]), repeat=int(ch == cells[t - 1]), gap=gap_of(mu, cv),
                        **ref_cols(muw, cvw, ch, env, '_wu')))
    return pd.DataFrame(out)


def tasks_for(path, ndim, side, envp, label, P):
    envs = load_envs(envp, ndim, side)
    df = pd.read_csv(path)
    sims, hums = [], []
    rng = np.random.default_rng(SEED + ndim)
    for i, row in df.iterrows():
        sh = json.loads(row['searchHistory']); order = json.loads(row['envOrder'])
        pidx = int(row['id']) if 'id' in row.index else i + 1
        sid = f'{label}_{pidx}'
        zk, rk = ('ycollectScaled', 'ycollect') if ndim == 1 else ('zcollectScaled', 'zcollect')
        for r in range(len(sh[zk])):
            if r >= len(order):
                continue
            env = envs.get(int(row['kernel']), {}).get(int(order[r]))
            if env is None:
                continue
            xs = sh['xcollect'][r]; ys = sh['ycollect'][r] if ndim == 2 else None
            cells = [int(xs[k]) if ndim == 1 else int(ys[k]) * side + int(xs[k]) for k in range(len(sh[zk][r]))]
            zs = np.array(sh[zk][r], float); raw = np.array(sh[rk][r], float)
            b, a = np.polyfit(raw, zs, 1) if np.ptp(raw) > 0 else (0.75, 5.0)
            hums.append((label, sid, r, env, cells, zs, raw))
            sims.append((label, sid, pidx, r, env, len(cells), cells[0], a, b, int(rng.integers(1 << 31)), P))
    return sims, hums


def as_ref(R, suffix):
    if not suffix:
        return R
    R = R.copy()
    for c in ('non_greedy', 'regret_inc', 'greedy_is_opt', 'dval', 'gap'):
        R[c] = R[c + suffix]
    return R


def summarize(R):
    ng = R[R.non_greedy == 1]
    g = ng.groupby(['dataset', 'agent'], sort=False).agg(
        n_ng=('regret_inc', 'size'), regret_inc=('regret_inc', 'mean'), found_opt=('found_opt', 'mean'),
        dval=('dval', 'mean'), novel=('novel', 'mean'), repeat=('repeat', 'mean'), greedy_is_opt=('greedy_is_opt', 'mean'))
    a = R.groupby(['dataset', 'agent'], sort=False).agg(p_ng=('non_greedy', 'mean'), found_opt_any=('found_opt', 'mean'))
    return a.join(g)


def by_A(R):
    ng = R[R.non_greedy == 1].copy()
    ng['Abin'] = pd.cut(ng.A, [0, 1, 3, 1000], labels=['A=1', 'A=2-3', 'A>=4'])
    return ng.groupby(['dataset', 'agent', 'Abin'], sort=False, observed=True).agg(
        n=('regret_inc', 'size'), regret_inc=('regret_inc', 'mean'), found_opt=('found_opt', 'mean'))


def main():
    P = load_params()
    s1, h1 = tasks_for(WU1D_CSV, 1, WU_GRIDSIDE_1D, ENV1D, 'Wu1D', P['Wu1D'])
    s2, h2 = tasks_for(WU2D_CSV, 2, WU_GRIDSIDE_2D, ENV2D, 'Wu2D', P['Wu2D'])
    with Pool(N_WORKERS, maxtasksperchild=10) as pool:
        H = pd.concat(pool.imap(human, h1 + h2, chunksize=8), ignore_index=True)
        S = pd.concat(pool.imap(simulate, s1 + s2, chunksize=4), ignore_index=True)
    R = pd.concat([H, S], ignore_index=True)
    R.to_csv(OUT / 'ucb_baseline_pertrial.csv.gz', index=False, compression='gzip')
    with pd.option_context('display.width', 250, 'display.max_colwidth', 70):
        lines = [f'Optimum-direction baseline. Agents simulated on the participants\' own environments, horizons and '
                 f'initial tiles ({N_REP} replications per round). Agents learn from unscaled rewards normalized (y-50)/100 as in Wu et al. code; greedy references: Wu specification and prior-variance-1 specification '
                 f'(same as for humans); true rewards on the participant-visible scale.', '']
        for suffix, lab in (('_wu', 'greedy reference = GP of Wu et al. (2018) on unscaled rewards (y-50)/100, prior mean 0, variance 1, noise variance 1e-4'),
                            ('', 'greedy reference = prior-variance-1 specification (prior mean 25 on the participant-visible scale, prior variance 1, noise SD 5)')):
            Rs = as_ref(R, suffix)
            lines += [f'=== {lab} ===', '[summary: among non-greedy choices]', summarize(Rs).to_string(float_format=lambda v: f'{v:.3f}'), '',
                      '[by remaining horizon, among non-greedy choices]', by_A(Rs).to_string(float_format=lambda v: f'{v:.3f}'), '']
    out = '\n'.join(lines)
    (OUT / 'ucb_baseline.txt').write_text(out, encoding='utf-8')
    print(out)
    ucb_horizon.main()


if __name__ == '__main__':
    main()
