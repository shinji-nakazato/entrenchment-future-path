from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import RESULTS

OUT = RESULTS
FIGS = OUT / 'figs'
FIGS.mkdir(parents=True, exist_ok=True)


def load():
    w = pd.read_csv(RESULTS / 'all_pertrial.csv', usecols=['dataset', 'A', 'non_greedy', 'Dsize'])
    w = w[w.dataset == 'Wilson']
    k = pd.read_csv(OUT / 'indicator_wu_pertrial.csv', usecols=['dataset', 'spec', 'A', 'non_greedy'])
    k = k[k.spec == 'wu'].drop(columns='spec')
    k['Dsize'] = k.dataset.map({'Wu1D': 30, 'Wu2D': 121})
    k['dataset'] = k.dataset.map({'Wu1D': 'Wu 1D', 'Wu2D': 'Wu 2D'})
    return pd.concat([w, k], ignore_index=True)


def main():
    df = load()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {'Wilson': 'tab:blue', 'Wu 1D': 'tab:orange', 'Wu 2D': 'tab:purple'}
    for d in ('Wilson', 'Wu 1D', 'Wu 2D'):
        sub = df[df.dataset == d]
        agg = sub.groupby('A').non_greedy.mean().reset_index()
        ax[0].plot(agg['A'], agg['non_greedy'], 'o-', color=colors[d], label=f'{d} (|D|={int(sub.Dsize.iloc[0])})')
    ax[0].set_xlabel('$A$ (remaining horizon)')
    ax[0].set_ylabel('Empirical P(non-greedy)')
    ax[0].set_ylim(0, 1.0)
    ax[0].legend()
    aok = df[df.A >= 2].groupby(['dataset', 'Dsize']).non_greedy.agg(['mean', 'size']).reset_index().sort_values('Dsize')
    ax[1].plot(aok['Dsize'], aok['mean'], 'o-', color='black', markersize=10, linewidth=2)
    for _, r in aok.iterrows():
        ax[1].annotate(f"{r['dataset']}\n{r['mean']:.3f}", (r['Dsize'], r['mean']), xytext=(8, -8), textcoords='offset points', fontsize=9)
    ax[1].set_xlabel(r'$|D|$ (number of options)')
    ax[1].set_ylabel(r'Mean P(non-greedy) at $A \geq 2$')
    ax[1].set_ylim(0, 1.05)
    ax[1].grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIGS / f'fig_cross_dataset.{ext}', dpi=200)
    plt.close(fig)
    txt = 'Cross-dataset figure data (Wu: GP specification of Wu et al. 2018)\n\n' + aok.to_string(index=False, float_format=lambda x: f'{x:.3f}')
    (OUT / 'fig_cross_dataset_data.txt').write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
