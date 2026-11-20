from pathlib import Path

import numpy as np
import pandas as pd

from config import RESULTS
from model_selection import vb_bms, bor_pxp

OUT = RESULTS


def main():
    lines = []
    for tag in ('lapse',):
        S = pd.read_csv(OUT / f'subject_fits_{tag}.csv')
        S['AIC'] = 2 * S.k - 2 * S.LL
        S['BIC'] = S.k * np.log(S.n) - 2 * S.LL
        models = list(dict.fromkeys(S.model))
        tab = S.groupby('model', sort=False).agg(k=('k', 'first'), sumLL=('LL', 'sum'),
                                                 sumAIC=('AIC', 'sum'), sumBIC=('BIC', 'sum'))
        for crit in ('AIC', 'BIC'):
            W = S.pivot(index='subjectID', columns='model', values=crit)[models]
            tab[f'n_best_{crit}'] = W.idxmin(axis=1).value_counts().reindex(models).fillna(0).astype(int)
            Lm = (-W / 2).to_numpy()
            a, Er, xp = vb_bms(Lm)
            bor, pxp = bor_pxp(Lm, a, xp)
            tab[f'Er_{crit}'] = Er
            tab[f'XP_{crit}'] = xp
            tab[f'PXP_{crit}'] = pxp
            tab.attrs[f'BOR_{crit}'] = bor
        tab['dSumAIC'] = tab.sumAIC - tab.sumAIC.min()
        tab['dSumBIC'] = tab.sumBIC - tab.sumBIC.min()
        with pd.option_context('display.width', 250, 'display.float_format', lambda v: f'{v:.3f}'):
            lines += [f'=== Subject-level model comparison, family {tag} '
                      f'({S.subjectID.nunique()} subjects, per-subject MLE; random-effects BMS with '
                      f'log evidence = -IC/2) ===',
                      tab.to_string(), f'BOR (AIC) = {tab.attrs["BOR_AIC"]:.3g}, '
                                       f'BOR (BIC) = {tab.attrs["BOR_BIC"]:.3g}', '']
    out = '\n'.join(lines)
    (OUT / 'subject_level.txt').write_text(out, encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
