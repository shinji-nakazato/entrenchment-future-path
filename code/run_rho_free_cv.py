import sys
from pathlib import Path

import numpy as np

from wilson_choice import build_first_free, as_dict, fit, nll, subset, SEED
from run_assumptions import models

from config import RESULTS

OUT = RESULTS


def main():
    R = build_first_free()
    D = as_dict(R)
    y = R['left'].to_numpy(float)
    subj = R.subjectID.to_numpy()
    rng = np.random.default_rng(SEED)
    uniq = np.unique(subj)
    fold = {s: i % 5 for i, s in enumerate(rng.permutation(uniq))}
    folds = np.array([fold[s] for s in subj])
    lines = ['F2u+e with rho free (all games): pooled fit, BIC, and 5-fold subject-level CV (same folds as wilson_models)']
    for name, fn, x0 in models(True):
        if name not in ('F2u+e: FP q=U_D + A_H + B (rho = 8)', '+ rho free'):
            continue
        best = None
        for st in (np.array(x0), np.array(x0) + 0.4, np.array(x0) - 0.4):
            th, ll = fit(fn, st, D, y)
            if best is None or ll > best[1]:
                best = (th, ll)
        th, ll = best
        cv = 0.0
        for f in range(5):
            tr, te = folds != f, folds == f
            th_f, _ = fit(fn, th, subset(D, tr), y[tr])
            cv += -nll(fn(th_f, subset(D, te)), y[te])
        k = len(x0)
        lines.append(f'  {name}: k={k} LL={ll:.3f} AIC={2*k-2*ll:.2f} BIC={k*np.log(len(y))-2*ll:.2f} CV={cv:.2f}')
    out = '\n'.join(lines)
    (OUT / 'rho_free_cv.txt').write_text(out, encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
