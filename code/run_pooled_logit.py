import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

from config import RESULTS, WU2D_CSV, RHO_WILSON, RHO_WU_PRIORVAR1


def load_combined() -> pd.DataFrame:
    W  = pd.read_csv(RESULTS / 'wilson_pertrial.csv')
    K1 = pd.read_csv(RESULTS / 'wu1d_pertrial_priorvar1.csv')
    K2 = pd.read_csv(RESULTS / 'wu2d_pertrial_priorvar1.csv')

    W  = W.rename(columns={'subjectID': 'subject_raw'})
    W['dataset']     = 'Wilson'
    W['Dsize']       = 2
    W['subject_uid'] = 'W_' + W['subject_raw'].astype(str)

    K1['dataset']     = 'Wu1D'
    K1['Dsize']       = 30
    K1['subject_uid'] = 'K1_' + K1['subject'].astype(str)

    K2['dataset'] = 'Wu2D'
    K2['Dsize']   = 121
    raw2 = pd.read_csv(WU2D_CSV)
    subj = []
    for ridx, row in raw2.iterrows():
        try:
            sh = json.loads(row['searchHistory'])
        except Exception:
            continue
        rounds = sh['zcollectScaled']
        subj.extend([f'K2_{ridx}'] * sum(max(len(zs) - 1, 0) for zs in rounds))
    if len(subj) >= len(K2):
        K2['subject_uid'] = subj[:len(K2)]
    else:
        pad = [subj[-1]] * (len(K2) - len(subj))
        K2['subject_uid'] = subj + pad

    cols = ['dataset', 'Dsize', 'subject_uid', 'A',
            'dmu', 'Sigma', 'V_q', 'kappa_v2', 'p_nongreedy_v2', 'non_greedy']
    All = pd.concat([W[cols], K1[cols], K2[cols]], ignore_index=True)
    All['log_A'] = np.log(All['A'].clip(lower=1))
    All['log_D'] = np.log(All['Dsize'])
    return All


def cluster_logit(formula: str, data: pd.DataFrame, cluster='subject_uid'):
    model = smf.glm(formula, data=data, family=sm.families.Binomial())
    return model.fit(cov_type='cluster',
                     cov_kwds={'groups': data[cluster].values})


def write_table2(df: pd.DataFrame) -> str:
    aok = (df[df.A >= 2]
           .groupby(['dataset', 'Dsize'])
           .agg(n=('non_greedy', 'size'),
                non_greedy_rate=('non_greedy', 'mean'),
                V_q=('V_q', 'mean'))
           .reset_index().sort_values('Dsize'))
    wilson_rate = aok.loc[aok.dataset == 'Wilson', 'non_greedy_rate'].iloc[0]
    aok['ratio_vs_Wilson'] = aok['non_greedy_rate'] / wilson_rate
    aok.to_csv(RESULTS / 'cross_dataset_Aok.csv', index=False)
    return aok.to_string(index=False)


def write_table3(df: pd.DataFrame) -> str:
    df = df.copy()
    df['dmu_std'] = df['dmu'] / np.sqrt(df['Sigma'].clip(lower=1e-8))
    m_logit = smf.logit('non_greedy ~ log_A + log_D + dmu_std', data=df).fit(disp=0)
    res     = cluster_logit('non_greedy ~ log_A + log_D + dmu_std', df)
    txt = [
        f'n = {len(df):,}, pseudo R^2 = {m_logit.prsquared:.4f}',
        '',
        str(res.summary().tables[1]),
    ]
    return '\n'.join(txt)


def write_table4(df: pd.DataFrame) -> str:
    df = df.copy()
    df['dmu_std_Sigma'] = df['dmu'] / np.sqrt(df['Sigma'].clip(lower=1e-8))
    rho = {'Wilson': RHO_WILSON, 'Wu1D': RHO_WU_PRIORVAR1, 'Wu2D': RHO_WU_PRIORVAR1}
    df['rho_task']    = df['dataset'].map(rho)
    df['dmu_std_rho'] = df['dmu'] / df['rho_task']
    df['dmu_std_task'] = df.groupby('dataset')['dmu'].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0))

    sections = []
    for label, var in [
        ('Delta_mu (no standardisation)',                 'dmu'),
        ('Delta_mu / sqrt(Sigma_{t-1})  (Table F.1 main)',  'dmu_std_Sigma'),
        ('Delta_mu / rho  (task-specific noise SD)',      'dmu_std_rho'),
        ('within-task z-score',                           'dmu_std_task'),
    ]:
        res = cluster_logit(f'non_greedy ~ log_A + log_D + {var}', df)
        sections.append(f'--- {label} ---\n' + str(res.summary().tables[1]))
    return '\n\n'.join(sections)


def main():
    df = load_combined()
    df.to_csv(RESULTS / 'all_pertrial.csv', index=False)
    print(f'combined per-trial: {len(df):,} rows, {df.subject_uid.nunique()} subjects')

    table2 = write_table2(df)
    table3 = write_table3(df)
    table4 = write_table4(df)

    (RESULTS / 'pooled_logit.txt').write_text(table3, encoding='utf-8')
    (RESULTS / 'pooled_logit_robust.txt').write_text(table4, encoding='utf-8')

    print('\n=== Appendix D: non-greedy rate and mean V_q at A>=2 (Wu: prior-variance-1 specification) ===')
    print(table2)
    print('\n=== Table F.1: pooled logit (Delta_mu / sqrt Sigma; Wu: prior-variance-1 specification) ===')
    print(table3)
    print('\n=== Appendix F: pooled logit robustness ===')
    print(table4)


if __name__ == '__main__':
    main()
