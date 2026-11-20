import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

warnings.filterwarnings('ignore')
from config import RESULTS

OUT = RESULTS


def logit(y, X, groups):
    X = sm.add_constant(X, has_constant='add')
    m = sm.GLM(y, X, family=sm.families.Binomial()).fit(
        cov_type='cluster', cov_kwds={'groups': pd.factorize(groups)[0]})
    return m


def coef_line(m, names):
    return '  '.join(f'{n}={m.params[n]:+.3f} (z={m.tvalues[n]:+.1f})' for n in names)


def prep(W):
    W = W.copy()
    W['logA'] = np.log(W.A)
    W['gap'] = W.dmu / np.sqrt(W.Sigma)
    ng = W.non_greedy == 1
    W['cat'] = np.where(W.is_prev == 1, 'repeat', np.where(W.n_samp > 0, 'revisit', 'novel'))
    for c in ('repeat', 'revisit', 'novel'):
        W[f'ng_{c}'] = (ng & (W.cat == c)).astype(int)
    W['ng_switch'] = (ng & (W.is_prev == 0)).astype(int)
    W['persist_subopt'] = ((W.is_prev == 1) & (W.f_chosen < W.f_max)).astype(int)
    W['has_next'] = (W.next >= 0).astype(int)
    W['next_same'] = ((W.next == W.chosen) & (W.next >= 0)).astype(int)
    W['next_greedy_old'] = ((W.next == W.greedy) & (W.next >= 0)).astype(int)
    W['b_def'] = (W.mu1_xg <= W.mu1_ch).astype(int)
    W['b_overtake'] = (W.mu1_ch >= W.mu1_greedy).astype(int)
    W['b_top'] = (W.top1 == W.chosen).astype(int)
    W['b_gvsg'] = (W.mu1_xg <= W.mu1_greedy).astype(int)
    W['delta'] = W.mu_ch - W.mu_xg
    G = W.s2_ch - W.c_xg_ch
    W['kb'] = np.where(G > 0, W.delta * np.sqrt(W.s2_ch + W.noise2) / G.where(G > 0), np.nan)
    W['Pb'] = norm.cdf(W.kb)
    return W


def share(s):
    return f'{s.mean():.3f}'


def section_categories(W, label):
    L = [f'[{label}] composition (n={len(W):,})']
    ng = W[W.non_greedy == 1]
    gr = W[W.non_greedy == 0]
    a2 = W[W.A >= 2]
    L.append(f'  A >= 2 (n={len(a2):,}): P(non-greedy) = {a2.non_greedy.mean():.3f}; mean V_q = {a2.V_q.mean():.4g}; '
             f'mean V_q / noise variance = {(a2.V_q / a2.noise2).mean():.4g}')
    L.append(f'  P(non-greedy) = {share(W.non_greedy)}; tie-aware = {share(W.ng_tieaware)}; '
             f'tied top = {share(W.n_top > 1)}')
    L.append(f'  among non-greedy (n={len(ng):,}): repeat prev = {share(ng.cat == "repeat")}, '
             f'revisit = {share(ng.cat == "revisit")}, novel = {share(ng.cat == "novel")}')
    L.append(f'  among greedy     (n={len(gr):,}): repeat prev = {share(gr.cat == "repeat")}, '
             f'revisit = {share(gr.cat == "revisit")}, novel = {share(gr.cat == "novel")}')
    L.append(f'  greedy cell already sampled = {share(W.greedy_sampled)}; greedy cell = previous choice = '
             f'{share(W.greedy_is_prev)}')
    L.append(f'  P(repeat previous) all trials = {share(W.is_prev)}; P(chosen never sampled) = {share(W.n_samp == 0)}')
    L.append(f'  non-greedy & chosen = global optimum x*_g: {share(ng.chosen == ng.xg)}; '
             f'greedy & greedy = x*_g: {share(gr.greedy == gr.xg)}')
    return L


def section_horizon(W, label, controls=('gap',)):
    L = [f'[{label}] horizon signature: logit(outcome ~ log A + {" + ".join(controls)}), all trials, '
         f'subject-clustered']
    X = W[['logA', *controls]]
    for y in ['non_greedy', 'ng_repeat', 'ng_revisit', 'ng_novel', 'ng_switch', 'is_prev', 'persist_subopt']:
        if W[y].sum() < 20:
            L.append(f'  {y:15s}: too few events ({W[y].sum()})')
            continue
        m = logit(W[y], X, W.subject)
        L.append(f'  {y:15s} rate={W[y].mean():.3f}  ' + coef_line(m, ['logA', *controls]))
    ng = W[W.non_greedy == 1]
    L.append('  conditional on non-greedy:')
    for y, v in [('P(repeat | ng)', (ng.cat == 'repeat').astype(int)),
                 ('P(novel | ng)', (ng.cat == 'novel').astype(int))]:
        if v.sum() < 20:
            L.append(f'    {y}: too few events ({v.sum()})')
            continue
        m = logit(v, ng[['logA', *controls]], ng.subject)
        L.append(f'    {y:15s} rate={v.mean():.3f}  ' + coef_line(m, ['logA']))
    return L


def section_rates_by_A(W, label, bins):
    W = W.assign(Ab=pd.cut(W.A, bins))
    g = W.groupby('Ab', observed=True).agg(n=('A', 'size'), ng=('non_greedy', 'mean'),
                                          ng_repeat=('ng_repeat', 'mean'), ng_revisit=('ng_revisit', 'mean'),
                                          ng_novel=('ng_novel', 'mean'), is_prev=('is_prev', 'mean'),
                                          persist_subopt=('persist_subopt', 'mean'))
    return [f'[{label}] rates by A bin', g.to_string(float_format=lambda v: f'{v:.3f}')]


def section_condb(W, label):
    ng = W[(W.non_greedy == 1)].copy()
    L = [f'[{label}] condition (b) given (a): n non-greedy = {len(ng):,}']
    L.append(f'  Def. 1(b) mu_t(x*_g) <= mu_t(x_t): all = {share(ng.b_def)}; '
             f'excluding x_t = x*_g = {share(ng[ng.chosen != ng.xg].b_def)} (n={int((ng.chosen != ng.xg).sum()):,})')
    sub = ng[ng.chosen != ng.xg]
    pos = sub[sub.delta > 0]; neg = sub[sub.delta <= 0]
    L.append(f'  share with delta_(t-1) = mu(x_t) - mu(x*_g) > 0 (Proposition 5 regime): {share(sub.delta > 0)}')
    L.append(f'    (b) | delta > 0 = {share(pos.b_def)} (n={len(pos):,});  (b) | delta <= 0 = {share(neg.b_def)} (n={len(neg):,})')
    ok = sub.Pb.notna()
    L.append(f'  Proposition 5 prediction Phi(kappa_b): mean predicted = {sub.Pb[ok].mean():.3f}, '
             f'observed = {sub.b_def[ok].mean():.3f} (n={int(ok.sum()):,}; G<=0 excluded: {int((~ok).sum()):,})')
    if ok.sum() > 100:
        s = sub[ok].assign(bin=pd.qcut(sub.Pb[ok], 10, duplicates='drop'))
        c = s.groupby('bin', observed=True).agg(pred=('Pb', 'mean'), obs=('b_def', 'mean'), n=('Pb', 'size'))
        L.append('  calibration by decile of predicted P(b):')
        L.append('    pred: ' + ' '.join(f'{v:.2f}' for v in c.pred))
        L.append('    obs : ' + ' '.join(f'{v:.2f}' for v in c.obs))
    L.append(f'  by category (excluding x*_g): ' + ', '.join(
        f'{k}={share(g.b_def)} (n={len(g):,})' for k, g in sub.groupby('cat')))
    L.append(f'  variants (all non-greedy): x_t overtakes old greedy mu_t(x_t) >= mu_t(x*_t) = {share(ng.b_overtake)}; '
             f'x_t becomes posterior top = {share(ng.b_top)}; mu_t(x*_g) <= mu_t(x*_t) = {share(ng.b_gvsg)}')
    return L


def section_next(W, label):
    S = W[W.has_next == 1]
    ng = S[S.non_greedy == 1]
    gr = S[S.non_greedy == 0]
    L = [f'[{label}] what follows a choice (A >= 2, next choice observed)']
    L.append(f'  after greedy (n={len(gr):,}): repeat same = {share(gr.next_same)}')
    L.append(f'  after non-greedy (n={len(ng):,}): repeat same = {share(ng.next_same)}, '
             f'return to old greedy = {share(ng.next_greedy_old)}, other = '
             f'{1 - ng.next_same.mean() - ng.next_greedy_old.mean():.3f}')
    so = ng[ng.chosen != ng.xg]
    for k, g in so.groupby('b_def'):
        L.append(f'    x_t != x*_g, (b) = {k}: n={len(g):,}, repeat same = {share(g.next_same)}, '
                 f'return to old greedy = {share(g.next_greedy_old)}')
    for k, g in so.groupby('b_top'):
        L.append(f'    x_t != x*_g, x_t is posterior top after r_t = {k}: n={len(g):,}, repeat same = {share(g.next_same)}')
    return L


def wilson_t5(L5):
    L5 = L5.copy()
    L5['H6'] = (L5.H == 6).astype(int)
    out = ['[Wilson t=5] first free choice; previous choice = 4th forced trial (n=19,200)']
    for y in ['non_greedy', 'ng_repeat', 'ng_switch']:
        r1, r6 = L5[L5.H == 1][y].mean(), L5[L5.H == 6][y].mean()
        m = logit(L5[y], L5[['H6', 'gap']], L5.subject)
        out.append(f'  {y:11s}: H1={r1:.3f}  H6={r6:.3f}  diff={r6 - r1:+.3f}   logit ' + coef_line(m, ['H6']))
    ng = L5[L5.non_greedy == 1]
    for h in (1, 6):
        g = ng[ng.H == h]
        out.append(f'  H={h}: P(non-greedy arm = 4th forced arm | ng) = {share(g.is_prev)} (n={len(g):,})')
    for info in ('equal', 'unequal'):
        s = L5[L5['info'] == info]
        out.append(f'  info={info}: ' + ', '.join(
            f'{y} H1={s[s.H == 1][y].mean():.3f} H6={s[s.H == 6][y].mean():.3f}' for y in ['ng_repeat', 'ng_switch']))
    h6 = L5[(L5.H == 6) & (L5.non_greedy == 1)]
    out.append(f'  H6, after non-greedy at t=5: next choice repeats it = {share(h6.next_same)} (n={len(h6):,}); '
               f'x_t != x*_g & (b): {share(h6[(h6.chosen != h6.xg)].b_def)}')
    gr6 = L5[(L5.H == 6) & (L5.non_greedy == 0)]
    out.append(f'  H6, after greedy at t=5: next choice repeats it = {share(gr6.next_same)}')
    return out


def main():
    W = prep(pd.read_csv(OUT / 'indicator_wu_pertrial.csv'))
    L = prep(pd.read_csv(OUT / 'indicator_wilson_pertrial.csv'))
    lines = ['Decomposition of the non-greedy indicator', '']
    for spec in ('wu', 'priorvar1', 'priorvar300'):
        for ds in ('Wu1D', 'Wu2D'):
            S = W[(W.spec == spec) & (W.dataset == ds)]
            lab = f'{ds}, {spec}'
            lines += section_categories(S, lab) + ['']
            lines += section_horizon(S, lab) + ['']
            if spec in ('wu', 'priorvar1'):
                lines += section_rates_by_A(S, lab, [0, 1, 2, 4, 8, 16, 40]) + ['']
            lines += section_condb(S, lab) + ['']
            lines += section_next(S, lab) + ['']
    lines += section_categories(L, 'Wilson, all free choices') + ['']
    lines += section_condb(L, 'Wilson, all free choices') + ['']
    lines += section_condb(L[L.t == 5], 'Wilson, t=5') + ['']
    lines += wilson_t5(L[L.t == 5]) + ['']
    lines += section_next(L[L.H == 6], 'Wilson H6, all free choices') + ['']
    out = '\n'.join(lines)
    (OUT / 'indicator.txt').write_text(out, encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
