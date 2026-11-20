from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import RESULTS

OUT = RESULTS

REFS = (('_wu', 'greedy reference = GP of Wu et al. (2018)'),
        ('', 'greedy reference = prior-variance-1 specification (prior mean 25 on the participant-visible scale, prior variance 1, noise SD 5)'))


def section(R):
    lines = ['', '[horizon dependence: logit(y ~ log A), subject-clustered; regret_inc and found_opt among non-greedy choices]']
    for (ds, ag), g in R.groupby(['dataset', 'agent'], sort=False):
        res = []
        for y, sub in [('non_greedy', g), ('regret_inc', g[g.non_greedy == 1]), ('found_opt', g[g.non_greedy == 1])]:
            X = sm.add_constant(np.log(sub.A.to_numpy(float)))
            m = sm.GLM(sub[y].to_numpy(float), X, family=sm.families.Binomial()).fit(
                cov_type='cluster', cov_kwds={'groups': pd.factorize(sub.subject)[0]})
            res.append(f'{y} {m.params[1]:+.3f} (z={m.tvalues[1]:+.1f})')
        lines.append(f'  {ds} {ag:58s} ' + '  '.join(res))
    lines += ['', '[logit(non_greedy ~ log A + standardized gap), subject-clustered (Appendix I)]']
    for (ds, ag), g in R.groupby(['dataset', 'agent'], sort=False):
        X = sm.add_constant(np.column_stack([np.log(g.A.to_numpy(float)), g.gap.to_numpy(float)]))
        m = sm.GLM(g.non_greedy.to_numpy(float), X, family=sm.families.Binomial()).fit(
            cov_type='cluster', cov_kwds={'groups': pd.factorize(g.subject)[0]})
        lines.append(f'  {ds} {ag:58s} log A {m.params[1]:+.3f} (z={m.tvalues[1]:+.1f})  '
                     f'gap {m.params[2]:+.3f} (z={m.tvalues[2]:+.1f})')
    return lines


def main():
    R0 = pd.read_csv(OUT / 'ucb_baseline_pertrial.csv.gz')
    lines = []
    for suffix, lab in REFS:
        R = R0.copy()
        if suffix:
            for c in ('non_greedy', 'regret_inc', 'greedy_is_opt', 'dval', 'gap'):
                R[c] = R[c + suffix]
        lines += ['', f'=== horizon logits, {lab} ==='] + section(R)
    out = '\n'.join(lines)
    with open(OUT / 'ucb_baseline.txt', 'a', encoding='utf-8') as fh:
        fh.write(out + '\n')
    print(out)


if __name__ == '__main__':
    main()
