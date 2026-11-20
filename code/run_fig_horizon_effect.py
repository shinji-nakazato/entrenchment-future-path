import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import wilson_choice as m1
from wilson_choice import build_first_free, as_dict, fit
from run_valuefree import fpr, vfree, f2ur, vfree_all
from run_info_condition import cell_labels

from config import RESULTS

OUT = RESULTS
FIGS = OUT / 'figs'
FIGS.mkdir(parents=True, exist_ok=True)
N_BOOT = 2000
SEED = 20261102

LAP = {n.split(' ', 1)[0]: (fn, x0) for n, _, fn, x0 in m1.LAPSE}
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dc'
MODELS = [('future-path (rho free)', '#2a78d6', '-', 'o'),
          ('value-free random exploration (epsilon_H)', '#eb6834', '--', 's'),
          ('horizon-specific noise (Wilson et al.)', '#1baf7a', ':', '^')]
LABELS = ['Future-path model (estimated $\\rho$)', 'Value-free random exploration', 'Horizon-specific softmax']
CELLS = [('equal prior observations', 'Equal prior\nobservations'),
         ('unequal, non-greedy to more-observed arm', 'Non-greedy option\nmore observed'),
         ('unequal, non-greedy to less-observed arm', 'Non-greedy option\nless observed')]


def ng_pred(fn, th, D):
    pl = fn(th, D)
    return np.where(D['greedy_left'] == 1, 1 - pl, pl)


def boot(subj, stat, rng):
    uniq = np.unique(subj)
    idx = {s: np.flatnonzero(subj == s) for s in uniq}
    out = []
    for _ in range(N_BOOT):
        ss = rng.choice(uniq, len(uniq), replace=True)
        out.append(stat(np.concatenate([idx[s] for s in ss])))
    return np.percentile(np.array(out), [2.5, 97.5], axis=0)


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    rng = np.random.default_rng(SEED)
    R = build_first_free()
    R['cell'] = cell_labels(R)
    E = R[R.dI == 0].reset_index(drop=True)
    DE, yE = as_dict(E), E['left'].to_numpy(float)
    DA, yA = as_dict(R), R['left'].to_numpy(float)

    eq = [(fpr, [-0.74, -0.26, 0.33, -1.57]),
          (vfree, [0.99, -0.33, -1.67, -0.45]),
          LAP['C0+e']]
    al = [(f2ur, [-1.0, -1.44, 5.27, -0.39, 1.3, -1.46]),
          (vfree_all, [1.26, -1.44, 3.84, -0.34, -1.57, -0.36]),
          LAP['C1+e']]
    rows = []
    for (name, *_), (fn, x0) in zip(MODELS, eq):
        th, ll = fit(fn, x0, DE, yE)
        E[name] = ng_pred(fn, th, DE)
        rows.append(dict(set='equal', model=name, k=len(x0), LL=ll, AIC=2 * len(x0) - 2 * ll))
    for (name, *_), (fn, x0) in zip(MODELS, al):
        th, ll = fit(fn, x0, DA, yA)
        R[name] = ng_pred(fn, th, DA)
        rows.append(dict(set='all', model=name, k=len(x0), LL=ll, AIC=2 * len(x0) - 2 * ll))
    fits = pd.DataFrame(rows)

    E['gq'] = pd.qcut(E.dmu, 5, labels=False)
    cols = ['non_greedy'] + [m[0] for m in MODELS]
    eff = lambda d: np.array([d[(d.gq == q) & (d.H == 6)][c].mean() - d[(d.gq == q) & (d.H == 1)][c].mean()
                              for q in range(5) for c in cols]).reshape(5, len(cols))
    left = pd.DataFrame(eff(E), columns=cols)
    subjE = E.subjectID.to_numpy()
    ciL = boot(subjE, lambda ix: eff(E.iloc[ix])[:, 0], rng)
    left['lo'], left['hi'] = ciL
    edges = pd.qcut(E.dmu, 5, retbins=True)[1]
    left['bin'] = [f'{edges[i]:.1f}\N{EN DASH}{edges[i + 1]:.1f}' for i in range(5)]
    left['n_H1'] = [int(((E.gq == q) & (E.H == 1)).sum()) for q in range(5)]
    left['n_H6'] = [int(((E.gq == q) & (E.H == 6)).sum()) for q in range(5)]

    keys = [(c, h) for c, _ in CELLS for h in (1, 6)]
    cellstat = lambda d: np.array([[d[(d.cell == c) & (d.H == h)][col].mean() for col in cols] for c, h in keys])
    right = pd.DataFrame(cellstat(R), columns=cols)
    ciR = boot(R.subjectID.to_numpy(), lambda ix: cellstat(R.iloc[ix])[:, 0], rng)
    right['lo'], right['hi'] = ciR
    right['cell'] = [c for c, _ in keys]
    right['H'] = [h for _, h in keys]
    right['n'] = [int(((R.cell == c) & (R.H == h)).sum()) for c, h in keys]

    fig = plt.figure(figsize=(7.2, 3.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[2.4, 0.35, 1, 1, 1], wspace=0.15)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(5)
    for (name, col, ls, mk), lab in zip(MODELS, LABELS):
        ax.plot(x, left[name], color=col, ls=ls, lw=2, marker=mk, ms=4.5, label=lab, zorder=2)
    ax.errorbar(x, left.non_greedy, yerr=[left.non_greedy - left.lo, left.hi - left.non_greedy], fmt='o',
                color=INK, ms=6, capsize=0, lw=1.2, mec='white', mew=1, label='Observed (95% CI)', zorder=3)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(x, left['bin'])
    ax.set_xlabel('Posterior-mean gap (quintile range, points)', fontsize=8, color=INK)
    ax.set_ylabel('Horizon effect ($H = 6$ minus $H = 1$)', fontsize=8, color=INK)
    ax.set_title('A  Games with equal prior observations', fontsize=9, loc='left', color=INK)
    style(ax)
    ax.tick_params(axis='x', labelsize=6.5)

    axs = [fig.add_subplot(gs[0, 2 + i]) for i in range(3)]
    for i, (a, (c, title)) in enumerate(zip(axs, CELLS)):
        sub = right[right.cell == c]
        xh = np.array([0, 1])
        for (name, col, ls, mk) in MODELS:
            a.plot(xh, sub[name], color=col, ls=ls, lw=2, marker=mk, ms=4.5, zorder=2)
        a.errorbar(xh, sub.non_greedy, yerr=[sub.non_greedy - sub.lo, sub.hi - sub.non_greedy], fmt='o',
                   color=INK, ms=6, capsize=0, lw=1.2, mec='white', mew=1, zorder=3)
        a.set_xticks(xh, ['$H = 1$', '$H = 6$'])
        a.set_xlim(-0.4, 1.4)
        a.set_ylim(0.05, 0.45)
        a.set_title(title, fontsize=7.5, color=INK)
        style(a)
        if i:
            a.tick_params(labelleft=False)
        else:
            a.set_ylabel('P(non-greedy)', fontsize=8, color=INK)
    axs[0].annotate('B  All games, by information condition', xy=(0, 1.22), xycoords='axes fraction',
                    fontsize=9, color=INK)
    h, l = ax.get_legend_handles_labels()
    order = [3, 0, 1, 2]
    fig.legend([h[k] for k in order], [l[k] for k in order], loc='lower center', ncol=2, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.30)
    for ext in ('pdf', 'png'):
        fig.savefig(FIGS / f'fig_horizon_effect.{ext}', dpi=300)
    plt.close(fig)

    fmt = lambda v: f'{v:.3f}'
    txt = '\n'.join(['Figure data (Wilson first free choice; pooled MLE; observed CI = subject bootstrap, '
                     f'{N_BOOT} resamples)', '', '[model fits]', fits.to_string(index=False, float_format=lambda v: f'{v:.1f}'), '',
                     '[A: horizon effect by posterior-gap quintile, equal games]', left.to_string(index=False, float_format=fmt), '',
                     '[B: P(non-greedy) by information cell x H, all games]', right.to_string(index=False, float_format=fmt)])
    (OUT / 'fig_horizon_effect_data.txt').write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
