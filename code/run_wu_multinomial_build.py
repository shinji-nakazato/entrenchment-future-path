import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (RESULTS, WU1D_CSV, WU2D_CSV, WU_LSCALE_1D, WU_LSCALE_2D,
                    WU_GRIDSIDE_1D, WU_GRIDSIDE_2D)
from gp_utils import SPECS, grid, posterior, observations

OUT = RESULTS


def build(path, ndim, side, ls, label, spec):
    df = pd.read_csv(path)
    Xg = grid(ndim, side)
    p = SPECS[spec]
    MU, S2, CH, A, T, SUBJ, VQ, RND, TT = [], [], [], [], [], [], [], [], []
    for ridx, row in df.iterrows():
        try:
            sh = json.loads(row['searchHistory'])
        except Exception:
            continue
        sid = f"{label}_{row['id']}" if 'id' in df.columns else f'{label}_{ridx}'
        zkey = 'zcollectScaled' if ndim == 2 else 'ycollectScaled'
        for r_idx in range(len(sh[zkey])):
            zs = observations(sh, ndim, r_idx, spec)
            xs = sh['xcollect'][r_idx]
            ys = sh['ycollect'][r_idx] if ndim == 2 else None
            n_t = len(zs)
            for t in range(1, n_t):
                if ndim == 1:
                    X = np.array(xs[:t], float).reshape(-1, 1)
                    chosen = int(xs[t])
                else:
                    X = np.stack([xs[:t], ys[:t]], axis=1).astype(float)
                    chosen = int(ys[t]) * side + int(xs[t])
                y = np.array(zs[:t], float)
                try:
                    mu, cov = posterior(X, y, Xg, ls, p['m0'], p['sf2'], p['noise2'])
                except np.linalg.LinAlgError:
                    continue
                D = len(mu)
                V_mu = float(((mu - mu.mean()) ** 2).mean())
                V_sig = float(np.trace(cov) / D - cov.sum() / (D * D))
                MU.append(mu.astype(np.float32))
                S2.append(np.clip(np.diag(cov), 1e-10, None).astype(np.float32))
                CH.append(chosen)
                A.append(n_t - t)
                T.append(n_t)
                TT.append(t + 1)
                SUBJ.append(sid)
                VQ.append(V_mu + V_sig)
                RND.append(r_idx)
    np.savez_compressed(OUT / f'wu_mn_{spec}_{label}.npz', mu=np.array(MU), s2=np.array(S2),
                        chosen=np.array(CH), A=np.array(A, float), T=np.array(T), t=np.array(TT),
                        subject=np.array(SUBJ), Vq=np.array(VQ), round=np.array(RND),
                        noise2=p['noise2'])
    return len(CH), len(set(SUBJ))


def main():
    lines = []
    for spec in SPECS:
        for args in [(WU1D_CSV, 1, WU_GRIDSIDE_1D, WU_LSCALE_1D, 'Wu1D'),
                     (WU2D_CSV, 2, WU_GRIDSIDE_2D, WU_LSCALE_2D, 'Wu2D')]:
            n, ns = build(*args, spec)
            lines.append(f'{spec} {args[4]}: n={n}, subjects={ns}')
    out = '\n'.join(lines)
    (OUT / 'wu_mn_build.txt').write_text(out, encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
