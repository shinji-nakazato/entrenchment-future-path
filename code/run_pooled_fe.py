import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import RESULTS


def gof(formula, data):
    m = smf.logit(formula, data=data).fit(disp=0)
    return len(data), m.llnull, m.llf, m.prsquared


def main():
    d = pd.read_csv(RESULTS / 'all_pertrial.csv')
    d['dmu_std'] = d['dmu'] / np.sqrt(d['Sigma'].clip(lower=1e-8))
    d['A_long'] = (d['A'] == 6).astype(float)

    lines = ['Pooled logistic regression diagnostics (Appendix F; Wu: prior-variance-1 specification)', '']

    lines.append('=== (1) McFadden pseudo-R^2: pooled vs within-task ===')
    lines.append(f'{"sample":<16}{"n":>8}{"logL_null":>12}{"logL_full":>12}{"McFadden":>10}')
    rows = [
        ('pooled',        'non_greedy ~ log_A + log_D + dmu_std', d),
        ('within-Wilson', 'non_greedy ~ A_long + dmu_std',        d[d.dataset == 'Wilson']),
        ('within-Wu1D',   'non_greedy ~ log_A + dmu_std',         d[d.dataset == 'Wu1D']),
        ('within-Wu2D',   'non_greedy ~ log_A + dmu_std',         d[d.dataset == 'Wu2D']),
    ]
    for name, f, sub in rows:
        n, ln, lf, r2 = gof(f, sub)
        lines.append(f'{name:<16}{n:>8,}{ln:>12.1f}{lf:>12.1f}{r2:>10.3f}')

    lines.append('')
    lines.append('=== (2) Task fixed-effects logit: non_greedy ~ log_A + dmu_std + C(dataset) ===')
    lines.append('(log|D| is collinear with dataset and absorbed; identifies WITHIN-task channels)')
    fe = smf.glm('non_greedy ~ log_A + dmu_std + C(dataset)',
                 data=d, family=sm.families.Binomial()).fit(
                 cov_type='cluster', cov_kwds={'groups': d['subject_uid'].values})
    for nm in fe.params.index:
        lines.append(f'  {nm:<22}{fe.params[nm]:+.4f}  (SE {fe.bse[nm]:.4f}, p {fe.pvalues[nm]:.2e})')

    lines.append('')
    lines.append('=== (3) Collinearity diagnostics (pooled design) ===')
    X = sm.add_constant(d[['log_A', 'log_D', 'dmu_std']].to_numpy())
    names = ['const', 'log_A', 'log_D', 'dmu_std']
    for i, nm in enumerate(names):
        lines.append(f'  VIF[{nm:<8}] = {variance_inflation_factor(X, i):.2f}')
    cond = np.linalg.cond(X)
    lines.append(f'  condition number = {cond:.1f}')
    lines.append(f'  corr(log_D, dmu_std) = {d[["log_D","dmu_std"]].corr().iloc[0,1]:.3f}')

    lines.append('')
    lines.append('=== (4) Trial-level correlation of V_q and the standardized gap within the Wu data ===')
    for ds in ('Wu1D', 'Wu2D'):
        s = d[d.dataset == ds]
        lines.append(f'  {ds}: corr(V_q, dmu_std) = {np.corrcoef(s.V_q, s.dmu_std)[0, 1]:.3f}  (n={len(s):,})')

    out = '\n'.join(lines)
    (RESULTS / 'pooled_fe.txt').write_text(out, encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
