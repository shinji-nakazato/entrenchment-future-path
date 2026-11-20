import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (RESULTS, WU1D_CSV, WU2D_CSV, WU_LSCALE_1D, WU_LSCALE_2D, WU_GRIDSIDE_1D, WU_GRIDSIDE_2D,
                    WILSON_CSV, RHO_WILSON, WILSON_FORCED_TRIALS)
from wu_envs import load_envs, ENV1D, ENV2D
from gp_utils import SPECS, grid, posterior, rbf_kernel

OUT = RESULTS


def wu_rounds(path, ndim, side, envp, label):
    envs = load_envs(envp, ndim, side)
    df = pd.read_csv(path)
    out = []
    for i, row in df.iterrows():
        sh = json.loads(row['searchHistory']); order = json.loads(row['envOrder'])
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
            out.append(dict(label=label, subject=f'{label}_{i}', round=r, env=env, cells=cells, z=zs, raw=raw, a=a, b=b))
    return out


def wu_section():
    lines = ['', '[1] Wu: prior mean relative to true values, and the lock-in condition in the observed persistent rounds']
    specs = {
        'prior-variance-1 specification: m0=25 on visible scale, sf2=1, noise SD 5': dict(scale='visible', **SPECS['priorvar1']),
        'prior-variance-300 specification: m0=50 on visible scale, sf2=300, noise SD 1': dict(scale='visible', **SPECS['priorvar300']),
        'Wu specification: (raw-50)/100, m0=0, sf2=1, noise var 1e-4': dict(scale='wu', m0=0.0, sf2=1.0, noise2=1e-4),
    }
    for path, ndim, side, envp, label, ls in [(WU1D_CSV, 1, WU_GRIDSIDE_1D, ENV1D, 'Wu1D', WU_LSCALE_1D),
                                              (WU2D_CSV, 2, WU_GRIDSIDE_2D, ENV2D, 'Wu2D', WU_LSCALE_2D)]:
        R = wu_rounds(path, ndim, side, envp, label)
        Xg = grid(ndim, side)
        for sname, sp in specs.items():
            def val(rd, f):
                return rd['a'] + rd['b'] * 100 * f if sp['scale'] == 'visible' else (100 * f - 50) / 100
            above_all, above_ng, above_lock, cond_lock, cond_all_ng = [], [], [], [], []
            for rd in R:
                env, cells = rd['env'], rd['cells']
                ys = rd['z'] if sp['scale'] == 'visible' else (rd['raw'] - 50) / 100
                above_all.append(np.mean(val(rd, env) > sp['m0']))
                T = len(cells)
                last3 = cells[-3:]
                xg = int(np.argmax(env))
                if len(set(last3)) == 1 and last3[0] != xg:
                    x = last3[0]
                    above_lock.append(val(rd, env[x]) > sp['m0'])
                    k0 = T - 3
                    while k0 > 0 and cells[k0 - 1] == x:
                        k0 -= 1
                    mu, _ = posterior(Xg[cells[:k0 + 1]].astype(float), ys[:k0 + 1], Xg, ls, sp['m0'], sp['sf2'], sp['noise2'])
                    cond_lock.append(mu[xg] < val(rd, env[x]))
                for t in range(1, T):
                    mu, _ = posterior(Xg[cells[:t]].astype(float), ys[:t], Xg, ls, sp['m0'], sp['sf2'], sp['noise2'])
                    star = int(np.argmax(mu))
                    ch = cells[t]
                    if ch != star and ch != xg:
                        above_ng.append(val(rd, env[ch]) > sp['m0'])
                        cond_all_ng.append(mu[xg] < val(rd, env[ch]))
            lines.append(f'  {label} | {sname}')
            lines.append(f'    share of cells with f(x) above prior mean: all cells {np.mean(above_all):.3f}; '
                         f'non-greedy chosen (x != x*_g) {np.mean(above_ng):.3f}; locked cells {np.mean(above_lock):.3f} '
                         f'(rounds locked on a non-optimal cell: {len(above_lock)})')
            lines.append(f'    lock-in condition mu_t(x*_g) < f(x): at non-greedy choices {np.mean(cond_all_ng):.3f}; '
                         f'at the start of the locked run {np.mean(cond_lock):.3f}')
    return lines


def wilson_section():
    df = pd.read_csv(WILSON_CSV).dropna(subset=['m1', 'm2'])
    df = df[df.gameLength == 10]
    F = WILSON_FORCED_TRIALS
    rows = []
    for _, g in df.iterrows():
        c = [int(g[f'c{t}']) for t in range(1, 11)]
        r = [float(g[f'r{t}']) for t in range(1, 11)]
        m = {1: float(g['m1']), 2: float(g['m2'])}
        best = 1 if m[1] > m[2] else 2
        worse = 3 - best
        nb = sum(1 for t in range(F) if c[t] == best)
        if nb == 0 or nb == F:
            continue
        mu_b = np.mean([r[t] for t in range(F) if c[t] == best])
        mu_w = np.mean([r[t] for t in range(F) if c[t] == worse])
        free = c[F:]
        rows.append(dict(subjectID=g['subjectID'], hot=int(mu_b < m[worse]), greedy_worse=int(mu_w > mu_b),
                         n_best_forced=nb, locked_worse=int(all(x == worse for x in free[-3:])),
                         share_worse=np.mean([x == worse for x in free]),
                         never_best=int(all(x == worse for x in free)), gap=m[best] - m[worse]))
    W = pd.DataFrame(rows)
    lines = ['', f'[2] Wilson H6 games (n={len(W):,}): flat prior; the frozen estimate of the better arm after the forced trials']
    lines.append(f'  share of games where the better arm\'s forced-trial mean is below the worse arm\'s true mean '
                 f'(lock-in condition): {W.hot.mean():.3f}')
    lines.append(f'  share where the worse arm is posterior-greedy at t=5: {W.greedy_worse.mean():.3f}')
    for k, g in W.groupby('hot'):
        lines.append(f'  lock-in condition = {k}: n={len(g):,}; last 3 free choices on the worse arm = {g.locked_worse.mean():.3f}; '
                     f'never chooses the better arm in 6 free choices = {g.never_best.mean():.3f}; share of worse-arm choices = {g.share_worse.mean():.3f}')
    for k, g in W.groupby('greedy_worse'):
        lines.append(f'  worse arm greedy at t=5 = {k}: n={len(g):,}; locked on worse = {g.locked_worse.mean():.3f}; never best = {g.never_best.mean():.3f}')
    W['gap_bin'] = pd.cut(W.gap, [0, 5, 9, 13, 21, 31])
    t = W.groupby('gap_bin', observed=True).agg(n=('hot', 'size'), hot=('hot', 'mean'), locked_worse=('locked_worse', 'mean'),
                                                never_best=('never_best', 'mean'))
    lines += ['  by true mean difference:', '  ' + t.to_string(float_format=lambda v: f'{v:.3f}').replace('\n', '\n  ')]
    return lines


def main():
    lines = ['Prior dependence of lock-in', '']
    lines += wu_section()
    lines += wilson_section()
    out = '\n'.join(lines)
    (OUT / 'prior_lockin.txt').write_text(out, encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
