import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

import wilson_choice as m1
from wilson_choice import fit, as_dict, build_first_free, subset
from config import N_WORKERS
from run_recovery import FAM_NOLAPSE, FAM_NOLAPSE_NAMES, FAM_LAPSE, _refit_all, confusion, inversion, pair_stats, ftab, G

from config import RESULTS

OUT = RESULTS
SEED = 20261027
N_SIM = int(sys.argv[1]) if len(sys.argv) > 1 else 50
SHRINKS = [float(sys.argv[2])] if len(sys.argv) > 2 else [1.0, 0.5]
NPROC = N_WORKERS
FAMS = {'nolapse': FAM_NOLAPSE, 'lapse': FAM_LAPSE}


def _init(D, tag):
    G['D'] = D
    G['fam'] = FAMS[tag]


def _subj_fit(task):
    tag, j, th0, sid = task
    fam = FAMS[tag]
    name, kind, fn, x0 = fam[j]
    D, y, subj = G['D'], G['y'], G['subj']
    m = subj == sid
    th, ll = fit(fn, th0, subset(D, m), y[m])
    return tag, j, sid, th, ll


def _init_subj(D, y, subj):
    G['D'], G['y'], G['subj'] = D, y, subj


def main(SHRINK):
    SUF = '' if SHRINK == 1.0 else f'_s{SHRINK:g}'
    R = build_first_free()
    D = as_dict(R)
    y = R['left'].to_numpy(float)
    subj = R['subjectID'].to_numpy()
    uniq = np.unique(subj)
    lines = []
    for tag, short in [('nolapse', lambda s: FAM_NOLAPSE_NAMES[s]), ('lapse', lambda s: s.split(' ')[0])]:
        fam = FAMS[tag]
        pooled = [fit(fn, x0, D, y)[0] for _, _, fn, x0 in fam]
        tasks = [(tag, j, pooled[j], s) for j in range(len(fam)) for s in uniq]
        t0 = time.time()
        with Pool(NPROC, initializer=_init_subj, initargs=(D, y, subj)) as pool:
            res = pool.map(_subj_fit, tasks, chunksize=2)
        par = {(j, s): th for _, j, s, th, _ in res}
        pd.DataFrame([dict(model=short(fam[j][0]), k=len(fam[j][3]), subjectID=s, LL=ll,
                           n=int((subj == s).sum()), theta=' '.join(f'{v:.4f}' for v in th))
                      for _, j, s, th, ll in res]).to_csv(OUT / f'subject_fits_{tag}.csv', index=False)
        rng = np.random.default_rng(SEED)
        sim_tasks, prow = [], []
        for j, (name, kind, fn, x0) in enumerate(fam):
            Th = np.array([par[(j, s)] for s in uniq])
            lo, hi = np.percentile(Th, 10, axis=0), np.percentile(Th, 90, axis=0)
            Th = np.clip(Th, lo, hi)
            Th = pooled[j] + SHRINK * (Th - pooled[j])
            p = np.empty(len(y))
            for i, s in enumerate(uniq):
                m = subj == s
                p[m] = fn(Th[i], subset(D, m))
            prow.append(dict(model=short(name), mean_p_ng=float(np.mean(np.where(D['greedy_left'] == 1, 1 - p, p))),
                             obs_p_ng=float(R.non_greedy.mean())))
            for k in range(N_SIM):
                yy = (rng.random(len(p)) < p).astype(float)
                sim_tasks.append((j, k, yy, pooled))
        with Pool(NPROC, initializer=_init, initargs=(D, tag)) as pool:
            out = pool.map(_refit_all, sim_tasks, chunksize=2)
        Rm = pd.DataFrame([r for rr in out for r in rr])
        names = [short(m[0]) for m in fam]
        Rm['gen_name'] = Rm.gen.map(dict(enumerate(names)))
        Rm['fit_name'] = Rm.fit.map(dict(enumerate(names)))
        Rm.to_csv(OUT / f'recovery_hetero_{tag}{SUF}.csv', index=False)
        lines += [f'=== Heterogeneous-generator model recovery, family {tag}: per-subject MLE '
                  f'(clipped to 10-90th pct, shrink {SHRINK:g} toward pooled), {N_SIM} sims per generator, pooled refit '
                  f'({(time.time() - t0) / 60:.1f} min) ===',
                  ftab(pd.DataFrame(prow)), '']
        for crit in ('AIC', 'BIC'):
            C = confusion(Rm, crit, names)
            lines += [f'[confusion, {crit}; rows = generator]', ftab(C), '',
                      f'[inversion, {crit}]', ftab(inversion(C)), '']
        real_aic = {}
        for (name, kind, fn, x0), th in zip(fam, pooled):
            real_aic[short(name)] = 2 * len(x0) + 2 * m1.nll(fn(th, D), y)
        pairs = [('FP calib', 'softmax tau_H')] if tag == 'nolapse' else [("C1'+e", 'F2u+e'), ('C1+e', 'F2u+e')]
        for a, b in pairs:
            obs = real_aic[a] - real_aic[b]
            lines += [f'[pair dAIC = AIC({a}) - AIC({b}); observed {obs:+.2f}]',
                      ftab(pair_stats(Rm, a, b, names, obs), '{:.2f}'), '']
        (OUT / f'recovery_hetero{SUF}.txt').write_text('\n'.join(lines), encoding='utf-8')
        print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    for sh in SHRINKS:
        main(sh)
