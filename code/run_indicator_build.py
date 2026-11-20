import json
import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

from config import (RESULTS, WU1D_CSV, WU2D_CSV, WU_LSCALE_1D, WU_LSCALE_2D, WU_GRIDSIDE_1D,
                    WU_GRIDSIDE_2D, WILSON_CSV, RHO_WILSON, WILSON_FORCED_TRIALS)
from wu_envs import load_envs, ENV1D, ENV2D
from gp_utils import SPECS, grid, posterior, observations

from config import N_WORKERS

OUT = RESULTS
TIE = 1e-9


def wu_rows(args):
    ridx, row, ndim, side, ls, label, spec, envs = args
    p = SPECS[spec]
    Xg = grid(ndim, side)
    try:
        sh = json.loads(row['searchHistory'])
        envOrder = json.loads(row['envOrder'])
    except Exception:
        return []
    kidx = int(row['kernel'])
    zkey = 'zcollectScaled' if ndim == 2 else 'ycollectScaled'
    out = []
    for r in range(len(sh[zkey])):
        if r >= len(envOrder):
            continue
        env = envs.get(kidx, {}).get(int(envOrder[r]))
        if env is None:
            continue
        zs = observations(sh, ndim, r, spec)
        xs = sh['xcollect'][r]
        ys = sh['ycollect'][r] if ndim == 2 else None
        cells = [int(xs[i]) if ndim == 1 else int(ys[i]) * side + int(xs[i]) for i in range(len(zs))]
        T = len(zs)
        post = {}
        for t in range(1, T + 1):
            X = (np.array(xs[:t], float).reshape(-1, 1) if ndim == 1
                 else np.stack([xs[:t], ys[:t]], axis=1).astype(float))
            try:
                post[t] = posterior(X, np.array(zs[:t], float), Xg, ls, p['m0'], p['sf2'], p['noise2'])
            except np.linalg.LinAlgError:
                post[t] = None
        xg = int(np.argmax(env))
        fmax = float(env[xg])
        for t in range(1, T):
            if post[t] is None or post[t + 1] is None:
                continue
            mu, cov = post[t]
            mu1, _ = post[t + 1]
            ch = cells[t]
            prev = cells[t - 1]
            hist = cells[:t]
            star = int(np.argmax(mu))
            n_top = int(np.sum(mu >= mu[star] - TIE))
            order = np.argsort(mu)[::-1]
            alt = int(order[1]) if order[0] == star else int(order[0])
            Sig = float(cov[star, star] + cov[alt, alt] - 2 * cov[star, alt])
            D = len(mu)
            Vq = float(((mu - mu.mean()) ** 2).mean() + np.trace(cov) / D - cov.sum() / D ** 2)
            nxt = cells[t + 1] if t + 1 < T else -1
            out.append(dict(
                dataset=label, spec=spec, subject=f'{label}_{row["id"] if "id" in row.index else ridx}', round=r,
                t=t + 1, T=T, A=T - t, chosen=ch, prev=prev, greedy=star, n_top=n_top,
                xg=xg, next=nxt,
                non_greedy=int(ch != star), ng_tieaware=int(mu[ch] < mu[star] - TIE),
                is_prev=int(ch == prev), n_samp=int(hist.count(ch)),
                greedy_sampled=int(star in hist), greedy_is_prev=int(star == prev),
                xg_sampled=int(xg in hist),
                f_chosen=float(env[ch]), f_greedy=float(env[star]), f_max=fmax,
                dmu=float(mu[star] - mu[alt]), Sigma=max(Sig, 1e-8), V_q=Vq,
                dmu_chosen=float(mu[star] - mu[ch]),
                mu_ch=float(mu[ch]), mu_xg=float(mu[xg]), s2_ch=float(cov[ch, ch]),
                c_xg_ch=float(cov[xg, ch]), noise2=p['noise2'],
                mu1_ch=float(mu1[ch]), mu1_xg=float(mu1[xg]), mu1_greedy=float(mu1[star]),
                top1=int(np.argmax(mu1)),
            ))
    return out


def build_wu():
    parts = []
    for spec in SPECS:
        for path, ndim, side, ls, label, envp in [
            (WU1D_CSV, 1, WU_GRIDSIDE_1D, WU_LSCALE_1D, 'Wu1D', ENV1D),
            (WU2D_CSV, 2, WU_GRIDSIDE_2D, WU_LSCALE_2D, 'Wu2D', ENV2D),
        ]:
            envs = load_envs(envp, ndim, side)
            df = pd.read_csv(path)
            tasks = [(i, row, ndim, side, ls, label, spec, envs) for i, row in df.iterrows()]
            with Pool(N_WORKERS) as pool:
                res = pool.map(wu_rows, tasks, chunksize=2)
            parts.append(pd.DataFrame([r for rr in res for r in rr]))
    return pd.concat(parts, ignore_index=True)


def build_wilson():
    df = pd.read_csv(WILSON_CSV).dropna(subset=['m1', 'm2'])
    rho2 = RHO_WILSON ** 2
    F = WILSON_FORCED_TRIALS
    rows = []
    for _, g in df.iterrows():
        H = int(g['gameLength'])
        if H not in (5, 10):
            continue
        c = [int(g[f'c{t}']) for t in range(1, H + 1)]
        r = [float(g[f'r{t}']) for t in range(1, H + 1)]
        m = {1: float(g['m1']), 2: float(g['m2'])}
        xg = 1 if m[1] > m[2] else 2
        n = {1: 0, 2: 0}; s = {1: 0.0, 2: 0.0}
        for t in range(F):
            n[c[t]] += 1; s[c[t]] += r[t]
        for t in range(F, H):
            if n[1] == 0 or n[2] == 0:
                n[c[t]] += 1; s[c[t]] += r[t]
                continue
            mu = {a: s[a] / n[a] for a in (1, 2)}
            v = {a: rho2 / n[a] for a in (1, 2)}
            star = 1 if mu[1] >= mu[2] else 2
            ch = c[t]
            n1 = dict(n); s1 = dict(s)
            n1[ch] += 1; s1[ch] += r[t]
            mu1 = {a: s1[a] / n1[a] for a in (1, 2)}
            nf = [n[1], n[2]]
            rows.append(dict(
                dataset='Wilson', spec='flat', subject=g['subjectID'], round=int(g['game']),
                t=t + 1, T=H, H=H - F, A=H - t, chosen=ch, prev=c[t - 1], greedy=star,
                n_top=1 + int(mu[1] == mu[2]), xg=xg, next=c[t + 1] if t + 1 < H else -1,
                non_greedy=int(ch != star), ng_tieaware=int(mu[ch] < mu[star]),
                is_prev=int(ch == c[t - 1]), n_samp=n[ch], greedy_sampled=1,
                greedy_is_prev=int(star == c[t - 1]), xg_sampled=1,
                f_chosen=m[ch], f_greedy=m[star], f_max=m[xg],
                dmu=abs(mu[1] - mu[2]), Sigma=v[1] + v[2], V_q=(v[1] + v[2] + (mu[1] - mu[2]) ** 2) / 4.0,
                dmu_chosen=mu[star] - mu[ch], mu_ch=mu[ch], mu_xg=mu[xg], s2_ch=v[ch],
                c_xg_ch=v[ch] if ch == xg else 0.0, noise2=rho2,
                mu1_ch=mu1[ch], mu1_xg=mu1[xg], mu1_greedy=mu1[star],
                top1=1 if mu1[1] >= mu1[2] else 2,
                info=('equal' if nf[0] == nf[1] else 'unequal'),
            ))
            n = n1; s = s1
    return pd.DataFrame(rows)


def main():
    W = build_wu()
    W.to_csv(OUT / 'indicator_wu_pertrial.csv', index=False)
    L = build_wilson()
    L.to_csv(OUT / 'indicator_wilson_pertrial.csv', index=False)
    lines = []
    for (spec, ds), g in W.groupby(['spec', 'dataset']):
        lines.append(f'{spec:>10} {ds}: n={len(g):,}, non-greedy={g.non_greedy.mean():.4f}')
    lines.append(f'Wilson: n={len(L):,} (t=5: {(L.t == 5).sum():,}), non-greedy={L.non_greedy.mean():.4f}')
    out = '\n'.join(lines)
    (OUT / 'indicator_build.txt').write_text(out, encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
