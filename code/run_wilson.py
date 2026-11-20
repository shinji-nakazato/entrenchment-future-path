import numpy as np
import pandas as pd
from scipy.stats import norm

from config import (WILSON_CSV, RESULTS,
                    RHO_WILSON, GAMMA, WILSON_FORCED_TRIALS)
from gp_utils import S_factor, kappa_q


def reconstruct_wilson_trials(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in df.iterrows():
        H = int(g['gameLength'])
        if H not in (5, 10):
            continue
        choices = [g[f'c{t}'] for t in range(1, H + 1)]
        rewards = [g[f'r{t}'] for t in range(1, H + 1)]

        n1 = sum(1 for c in choices[:WILSON_FORCED_TRIALS] if c == 1)
        n2 = WILSON_FORCED_TRIALS - n1
        sum1 = sum(rewards[t] for t in range(WILSON_FORCED_TRIALS) if choices[t] == 1)
        sum2 = sum(rewards[t] for t in range(WILSON_FORCED_TRIALS) if choices[t] == 2)

        for t in range(WILSON_FORCED_TRIALS, H):
            A = H - t
            if n1 == 0 or n2 == 0:
                c = choices[t]; r = rewards[t]
                if c == 1: n1 += 1; sum1 += r
                else:      n2 += 1; sum2 += r
                continue

            mu1, mu2 = sum1 / n1, sum2 / n2
            s1, s2 = RHO_WILSON ** 2 / n1, RHO_WILSON ** 2 / n2

            if mu1 >= mu2:
                mu_star, mu_alt = mu1, mu2
                sig2_star, sig2_alt = s1, s2
                greedy = 1
            else:
                mu_star, mu_alt = mu2, mu1
                sig2_star, sig2_alt = s2, s1
                greedy = 2

            dmu = mu_star - mu_alt
            Sigma = sig2_star + sig2_alt
            V_q = (Sigma + dmu ** 2) / 4.0
            S = S_factor(GAMMA, A)

            k_v2    = kappa_q(dmu, Sigma, V_q,  S, RHO_WILSON ** 2)
            k_v1    = kappa_q(dmu, Sigma, 0.0,  S, RHO_WILSON ** 2)
            k_naive = dmu / np.sqrt(Sigma) if Sigma > 0 else np.nan

            non_greedy = int(choices[t] != greedy)
            rows.append(dict(
                subjectID=g['subjectID'], expt=g['expt_name'],
                game=g['game'], block=g['block'], gameLength=H,
                uc=g['uc'], t=t + 1, A=A,
                mu_star=mu_star, mu_alt=mu_alt,
                sig2_star=sig2_star, sig2_alt=sig2_alt,
                dmu=dmu, Sigma=Sigma, V_q=V_q,
                kappa_v2=k_v2, kappa_v1=k_v1, kappa_naive=k_naive,
                p_nongreedy_v2=1.0 - norm.cdf(k_v2),
                p_nongreedy_v1=1.0 - norm.cdf(k_v1),
                p_nongreedy_naive=1.0 - norm.cdf(k_naive),
                non_greedy=non_greedy,
            ))
            c = choices[t]; r = rewards[t]
            if c == 1: n1 += 1; sum1 += r
            else:      n2 += 1; sum2 += r
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(WILSON_CSV).dropna(subset=['m1', 'm2'])
    R = reconstruct_wilson_trials(df)
    R.to_csv(RESULTS / 'wilson_pertrial.csv', index=False)
    print(f'Wilson: {len(R)} free-choice trials, {R.subjectID.nunique()} subjects')


if __name__ == '__main__':
    main()
