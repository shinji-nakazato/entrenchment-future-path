import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold', 'pdf.fonttype': 42})

from config import RESULTS

OUT = RESULTS
FIGS = OUT / 'figs'
FIGS.mkdir(parents=True, exist_ok=True)
SPEC = sys.argv[1] if len(sys.argv) > 1 else 'wu'
STYLE = {'Wu1D': ('tab:orange', 'o', '1D'), 'Wu2D': ('tab:purple', 's', '2D')}
TSTYLE = {6: '--', 11: '-', 21: '--', 41: '-'}


def load(spec):
    d = pd.read_csv(OUT / 'indicator_wu_pertrial.csv', usecols=['dataset', 'spec', 'subject', 'T', 'A', 'non_greedy', 'dmu', 'V_q'])
    return d[d.spec == spec].copy()


def gap_bins(R):
    rows = []
    for ds in ['Wu1D', 'Wu2D']:
        sub = R[(R.dataset == ds) & (R.A >= 2)].copy()
        sub['bin'] = pd.qcut(sub['dmu'], 8, duplicates='drop')
        g = (sub.groupby('bin', observed=True)
             .agg(x=('dmu', 'mean'), emp=('non_greedy', 'mean'), n=('non_greedy', 'size')).reset_index())
        g['dataset'] = ds
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def main():
    R = load(SPEC)
    G = gap_bins(R)
    byT = R.groupby(['dataset', 'T', 'A']).agg(V_q=('V_q', 'mean'), n=('V_q', 'size')).reset_index()
    byA = R.groupby(['dataset', 'A']).agg(ng=('non_greedy', 'mean'), V_q=('V_q', 'mean'), n=('non_greedy', 'size')).reset_index()

    fig, ax = plt.subplots(1, 2, figsize=(5.15, 2.5))
    for ds, (c, mk, lab) in STYLE.items():
        g = G[G.dataset == ds]
        ci = 1.96 * np.sqrt(g.emp * (1 - g.emp) / g.n)
        ax[0].errorbar(g.x, g.emp, yerr=ci, fmt='-', color=c, marker=mk, capsize=1.5, markersize=3.5, linewidth=1.2, label=lab)
        for T in sorted(byT[byT.dataset == ds]['T'].unique()):
            s = byT[(byT.dataset == ds) & (byT['T'] == T)].sort_values('A')
            ax[1].plot(s.A, s.V_q, color=c, linestyle=TSTYLE.get(T, '-'), marker=mk, markersize=2.5, linewidth=1.0, alpha=0.85, label=f'{lab}, T={T}')
    ax[0].set_xlabel(r'Posterior-mean gap $\Delta\mu_{t-1}(x^\star_t, x)$')
    ax[0].set_ylabel('P(non-greedy)')
    ax[0].set_ylim(0, 1.0)
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel('$A$ (remaining horizon)')
    ax[1].set_ylabel(r'$V_q$ ($q = U_D$)')
    ax[1].legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    stem = 'fig_wu_appendix' if SPEC == 'wu' else f'fig_wu_appendix_{SPEC}'
    for ext in ('pdf', 'png'):
        fig.savefig(FIGS / f'{stem}.{ext}', dpi=200)
    plt.close(fig)

    v = byA.pivot(index='A', columns='dataset', values='V_q').dropna()
    ratio = v['Wu2D'] / v['Wu1D']
    lines = [f'Wu appendix figure data (GP specification: {SPEC})', '',
             '[left: non-greedy rate by 8 within-dataset quantile bins of the posterior mean gap, A >= 2]',
             G[['dataset', 'x', 'emp', 'n']].to_string(index=False, float_format=lambda x: f'{x:.4f}'), '',
             '[non-greedy rate and mean V_q by remaining horizon A]',
             byA.to_string(index=False, float_format=lambda x: f'{x:.4f}'), '',
             '[ratio of mean V_q, 2D / 1D, over common A]',
             '  ' + '  '.join(f'A={a}:{r:.2f}' for a, r in ratio.items()),
             f'  mean {ratio.mean():.2f} (range {ratio.min():.2f} to {ratio.max():.2f})']
    txt = '\n'.join(lines)
    (OUT / f'{stem}_data.txt').write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
