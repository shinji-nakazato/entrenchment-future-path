# Reproduction package: Entrenchment in Human Exploration

Analysis code and data for the manuscript
*"Entrenchment in Human Exploration: Imagined Future Choices and Horizon-Dependent Random Exploration."*

The package reproduces the figures, the tables, and the in-text statistics of the
empirical assessment (Section 7) and of the appendices. It ships the analysis code and
only the raw subject-data files that the pipeline reads. The code contains no comments
or docstrings; all documentation is in this file.

## Quick start

Requires Python >= 3.10 and the packages in `requirements.txt`.

```bash
pip install -r requirements.txt
python code/make_all.py              # everything
python code/make_all.py --skip-slow  # everything except the five slow steps
```

The driver runs the analysis scripts in dependency order and writes numbers and tables
to `results/` and the figures to `results/figs/`. Without the slow steps the pipeline
takes about 25 minutes on a 20-core machine. The slow steps are the model and
parameter recovery (`run_recovery.py`, about 25 minutes), the recovery under
participant heterogeneity (`run_recovery_hetero.py`, about 12 minutes) with its
participant-level summary (`run_subject_level.py`), the probit error at every horizon
(`run_probit_error_horizons.py`, about 10 minutes), and the multinomial Wu models
(`run_wu_multinomial.py`, about 1 to 3 hours; its six model sets run in parallel).
Scripts that use several processes take the number of workers from
`config.N_WORKERS` (the number of CPU cores minus one); the results do not depend on
it. Any script can also be run on its own, e.g. `python code/run_competitors.py`, once
its inputs exist (see the dependency tiers below).

One generated file is not included in the repository because of its size:
`results/ucb_baseline_pertrial.csv.gz` (about 110 MB, the trial-level output of the
GP-UCB simulation). `run_ucb_baseline.py` writes it and then summarizes it into
`results/ucb_baseline.txt`, which is included.

## Repository layout

```
.
├── code/
│   ├── config.py               paths, task constants, number of worker processes
│   ├── gp_utils.py             GP posterior, V_q, kappa^q, GP prior specifications
│   ├── wilson_choice.py        Wilson first-free-choice data and the choice-rule library
│   ├── model_selection.py      random-effects Bayesian model selection
│   ├── wu_envs.py              Wu reward environments (true value of every cell)
│   ├── ucb_horizon.py          horizon summaries for run_ucb_baseline.py
│   ├── assumptions_gap.py      gap-binned predictions for run_assumptions.py
│   ├── make_all.py             driver: runs every script in order, halts on the first failure
│   └── run_*.py                one script per analysis (see the map below)
├── data/                       raw subject data (MIT-licensed upstream)
│   ├── horizon_ddm/            Wilson Horizon Task, compiled by Feng et al. (2021)
│   └── gridsearch/             Wu et al. (2018) GP bandit: choice CSVs, environment files,
│                               and the published per-participant parameter estimates
├── results/                    generated outputs
│   └── figs/                   fig_horizon_effect.{pdf,png} (Figure 1),
│                               fig_cross_dataset.{pdf,png} (Figure D.1),
│                               fig_wu_appendix.{pdf,png} (Figure I.1)
├── requirements.txt
└── README.md
```

## Where each result comes from

| Paper | Script(s) | Output file(s) |
|---|---|---|
| Section 7.1, footnote: participant with accuracy 0.155; lapse 0.17 to 0.14 and AIC difference 28 to 33 without this participant | `run_equal_games_checks.py` | `equal_games_checks.txt` |
| Section 7.3, Table 3 | `run_competitors.py`, `run_info_condition.py` | `competitors.txt`, `info_condition.txt` |
| Section 7.3: participant-level model selection and reference distributions (horizon-specific softmax) | `run_equal_games_robust.py` | `equal_games_robust.txt` |
| Section 7.3: value-free random exploration (participant level, reference distributions) | `run_valuefree.py` | `valuefree.txt` |
| Section 7.3: Bayes-optimal values (slopes 1.000 and 0.998) | `run_competitors.py` | `competitors.txt` |
| Section 7.3: shape test of the horizon effect across gap quintiles | `run_shape_test.py` | `shape_test.txt` |
| Figure 1 | `run_fig_horizon_effect.py` | `figs/fig_horizon_effect.{pdf,png}`, `fig_horizon_effect_data.txt` |
| Section 7.4, Table 4; information bonus and decision noise by participant and horizon | `run_info_condition.py` | `info_condition.txt`, `info_condition_subject_params.csv` |
| Section 7.4, Table 5 | `run_rho_free_cv.py` (first row), `run_wilson_models.py`, `run_competitors.py` (value-free and Bayes-optimal rows) | `rho_free_cv.txt`, `wilson_models.txt`, `competitors.txt` |
| Section 7.4: normative information bonus of about 0.6 points | `run_competitors.py` | `competitors.txt` |
| Section 7.5.1, Table 6, footnote on the lapse component | `run_wu_multinomial_build.py`, `run_wu_multinomial.py` | `wu_mn_models.txt`, `wu_mn_models.csv` |
| Section 7.5.2: composition of non-greedy choices, horizon coefficients, Wilson repeat and switch | `run_indicator.py` | `indicator.txt` |
| Section 7.5.2: 2D non-greedy rate by A (0.820, 0.86 to 0.98) | `run_fig_wu_appendix.py` | `fig_wu_appendix_data.txt` |
| Section 7.5.3, Table 7 | `run_ucb_baseline.py` | `ucb_baseline.txt` |
| Section 7.6, Table 8 | `run_assumptions.py` (with `assumptions_gap.py`); gamma free: `run_equal_games_checks.py` | `assumptions.txt`, `equal_games_checks.txt` |
| Section 7.7: condition (b) | `run_indicator.py` | `indicator.txt` |
| Section 7.7: lock-in condition (D3) in the Wilson and Wu data | `run_prior_lockin.py` | `prior_lockin.txt` |
| Appendix B, Table B.1 | `run_probit_error_horizons.py` | `probit_error_horizons.txt` |
| Appendix D, Table D.1 (Wu specification; V_q in units of the observation-noise variance) | `run_indicator.py` | `indicator.txt` |
| Appendix D: mean V_q 1.21 and 2.04 under the prior-variance-1 specification | `run_pooled_logit.py` | `cross_dataset_Aok.csv` |
| Figure D.1 | `run_fig_cross_dataset.py` | `figs/fig_cross_dataset.{pdf,png}`, `fig_cross_dataset_data.txt` |
| Appendix E, Table E.1 | `run_calib_sensitivity.py` | `calib_sensitivity_grid.csv` |
| Appendix F, Table F.1 and robustness to the standardization of the gap | `run_pooled_logit.py` | `pooled_logit.txt`, `pooled_logit_robust.txt` |
| Appendix F: within-task pseudo R^2, task fixed effects, collinearity, correlation of V_q and the gap | `run_pooled_fe.py` | `pooled_fe.txt` |
| Appendix F: wild cluster bootstrap | `run_wild_bootstrap.py` | `wild_bootstrap.txt` |
| Appendix G, Table G.1 and model recovery without a lapse component | `run_recovery.py` | `recovery_model_lapse.txt`, `recovery_model_nolapse.txt` |
| Appendix G, Table G.2 and the recovery error of theta | `run_recovery.py` | `recovery_param.txt` |
| Appendix G, Table G.3 | `run_recovery_hetero.py` (all-game rows), `run_equal_games_robust.py` (rows for games with equal prior observations) | `recovery_hetero.txt`, `recovery_hetero_s0.5.txt`, `equal_games_robust.txt` |
| Appendix G, Table G.4 | `run_subject_level.py` | `subject_level.txt` |
| Appendix H, Table H.1 | `run_wilson_models.py` | `wilson_models.txt` |
| Appendix H, Table H.2 | `run_info_condition.py` | `info_condition.txt` |
| Appendix I: GP-UCB baseline of the horizon coefficient; human coefficients | `run_ucb_baseline.py`, `run_indicator.py` | `ucb_baseline.txt`, `indicator.txt` |
| Appendix I: ties in the posterior mean | `run_indicator.py` | `indicator.txt` |
| Appendix I, Table I.1 | `run_wu_multinomial.py` | `wu_mn_models.txt` |
| Appendix I, Table I.2 | `run_indicator.py` | `indicator.txt` |
| Appendix I, Table I.3 | `run_prior_lockin.py` | `prior_lockin.txt` |
| Figure I.1 | `run_fig_wu_appendix.py` | `figs/fig_wu_appendix.{pdf,png}`, `fig_wu_appendix_data.txt` |

Table 3 and Table 5 combine rows from several output files. The differences in AIC and
BIC in these tables are taken relative to the future-path model with estimated rho:
`FP q=U_D + B + lapse (rho free)` in `competitors.txt` for Table 3, and the
`+ rho free` row of `rho_free_cv.txt` for Table 5.

## Model codes in the output files

Wilson task (`wilson_choice.py`; all likelihoods are on the left/right choice). `B` is
the side bias, `A_H` a horizon-specific information bonus, `+e` a common lapse
component `e/2 + (1 - e) P`.

| Code in the outputs | Name in the paper | Tables |
|---|---|---|
| `softmax (mu)` | Softmax (posterior mean) | H.1 |
| `Thompson (tempered)` | Thompson sampling (tempered) | H.1 |
| `UCB-softmax` | GP-UCB softmax | H.1 |
| `future-path, q=U_D` | Future-path model (q = U_D) | H.1 |
| `future-path, calib. q` | Future-path model (calibrated q) | H.1 |
| `softmax (per-H tau)` | Horizon-specific softmax model | H.1 |
| `C0` / `C0+e` | Horizon-specific softmax model (+ side bias B) | H.1 / 5, G.1, G.4 |
| `C3` / `C3+e` | Horizon-specific Thompson sampling (+ side bias B) | H.1 / 5 |
| `C2` / `C2+e` | UCB (horizon-specific exploration bonus and temperature) | 5 |
| `C1` / `C1+e` | Baseline Wilson hybrid model | H.1 / 5, G.1, G.3, G.4, H.2 |
| `C1'` / `C1'+e` | Extended Wilson hybrid model (horizon x information-condition noise) | H.1 / 5, G.1, G.3, G.4, H.2 |
| `C4` / `C4+e` | Extended Gershman hybrid model (horizon-specific weights) | H.1 / 5, G.1, G.4 |
| `F1` / `F1+e` | Future-path model (calibrated q) + side bias B / future-path model (rho = 8, no info bonus) | H.1 / 5, G.1, G.4, H.2 |
| `F2` | Future-path model (calibrated q) + horizon-specific info bonus | H.1 |
| `F2u` / `F2u+e` | Future-path model (q = U_D) + horizon-specific info bonus / future-path model (rho = 8) + horizon-specific info bonus | H.1 / 5, G.1-G.4, H.2 |
| `F0+e` | Future-path model (rho = 8, no V_q) + horizon-specific info bonus | 5, G.1, G.4, H.2 |
| `L1`, `L2`, `L3` | Baseline Wilson hybrid model (probit link); future-path model + side bias B (logistic link); future-path model + info bonus (logistic link) | H.1 |
| `FP q=U_D + B + lapse (rho = 8)` / `(rho free)` | Future-path model (q = U_D, rho = 8) / (q = U_D, estimated rho) | 3 |
| `softmax tau_H + B + lapse_H` | Horizon-specific softmax model + horizon-specific lapse | 3 |
| `softmax one tau + B + lapse_H` | Value-free random exploration (softmax + horizon-specific lapse) | 3, 5 |
| `Bayes-optimal Q + ...` | Bayes-optimal value models | 3, 5 |

In `F1`, `F2`, and `future-path, calib. q` the weight lambda on V_q is estimated
("calibrated q"). With a lapse component (`F1+e`) the estimate reaches its upper bound
of 1, so the fit coincides with q = U_D; without a lapse component the estimates are
about 0.02 to 0.03.

Wu task (`run_wu_multinomial.py`; multinomial softmax over all cells):

| Code | Name in Table 6 and Table I.1 |
|---|---|
| `S0` | Softmax, constant softmax temperature |
| `S2` | Softmax, binned-A softmax temperatures |
| `U0` | GP-UCB (Wu et al., 2018) |
| `U1` | GP-UCB, beta(A) + power-law softmax temperature in A (`tau_A_exponent` is the power) |
| `U2` | GP-UCB, beta(A) + binned-A softmax temperatures |
| `FU` | Future-path softmax temperature (q = U_D) |
| `FL` | Future-path softmax temperature (estimated lambda) |
| `FA` | Future-path softmax temperature x binned-A multipliers |
| `V0`, `V1` | gap-decile temperature control, and the partial contribution of V_q within horizon (Section 7.5.1) |

Model and parameter recovery (`run_recovery.py`, `run_recovery_hetero.py`): the family
`nolapse` contains the rules of Table H.1 (top panel) without Thompson sampling, and the
family `lapse` contains the seven rules with a lapse component of Table G.1.

## GP prior specifications for the Wu task

`gp_utils.SPECS` defines the three specifications used in the paper:

| Name in the code | Name in the paper | Rewards | Prior mean | Prior variance | Noise |
|---|---|---|---|---|---|
| `wu` | Wu specification (primary) | unscaled, normalized as (y - 50)/100 | 0 | 1 | variance 1e-4 |
| `priorvar1` | prior-variance-1 specification | participant-visible scale | 25 | 1 | SD 5 |
| `priorvar300` | prior-variance-300 specification | participant-visible scale | 50 | 300 | SD 1 |

The RBF length-scales are the medians reported by Wu et al. (2018): 0.5 in 1D and 0.76
in 2D. The pooled regression of Appendix F uses the prior-variance-1 specification.

## Shared modules

### `config.py`

Defines all paths relative to the repository root, so the package runs from any
location, and creates `results/` and `results/figs/` on import. Task constants:

| Constant | Value | Meaning |
|---|---|---|
| `RHO_WILSON` | 8.0 | observation-noise SD of the Wilson task |
| `RHO_WU_PRIORVAR1` | 5.0 | observation-noise SD of the Wu task under the prior-variance-1 specification |
| `GAMMA` | 1.0 | discount factor (neither task has explicit discounting) |
| `WILSON_FORCED_TRIALS` | 4 | forced choices before the first free choice |
| `WU_LSCALE_1D`, `WU_LSCALE_2D` | 0.5, 0.76 | median fitted RBF length-scales reported by Wu et al. (2018) |
| `WU_GRIDSIDE_1D`, `WU_GRIDSIDE_2D` | 30, 11 | grid side; the 2D grid has 11 x 11 = 121 cells |
| `WU_PARAMS_1D`, `WU_PARAMS_2D` | paths | published per-participant parameter estimates of Wu et al. (2018) |
| `N_WORKERS` | CPU cores - 1 | worker processes for the parallel scripts |

### `gp_utils.py`

Implements the objects of Sections 3 and 4:

```
S(gamma, A) = sum_{i=0}^{A-1} gamma^{2i}                  (= A when gamma = 1)
V_mu(q)     = Var_q[ mu_{t-1}(X) ]
V_sigma(q)  = E_q[ sigma_{t-1}^2(X) ] - E_{q x q}[ sigma_{t-1}(X, Y) ]
V_q         = V_mu(q) + V_sigma(q)
kappa^q     = Delta_mu / sqrt( Delta_V + 2 (S - 1) V_q + 2 S rho^2 )
```

- `rbf_kernel`: squared-exponential kernel `exp(-||x - y||^2 / (2 l^2))`.
- `gp_posterior`: posterior mean and full covariance over the grid, under a zero-mean,
  unit-variance RBF prior (Cholesky-based).
- `V_q_uniform`: V_q under q = U_D, i.e. `V_mu = mean_x (mu(x) - mean mu)^2` and
  `V_sigma = trace(cov)/|D| - sum(cov)/|D|^2`.
- `kappa_q`: returns NaN if the implied variance is non-positive.
- `SPECS`, `observations`, `grid`, `posterior`: GP posterior with an explicit prior mean
  `m0`, prior variance `sf2` and noise variance `noise2` under the three
  specifications above. `observations(searchHistory, ndim, round, spec)` returns the
  visible (scaled) rewards for `priorvar1` and `priorvar300`, and the normalized
  unscaled rewards for `wu`.

### `wilson_choice.py`

Builds the 19,200 Wilson first free choices from the raw data with left/right labels
(`build_first_free`): for each game, the forced-trial counts `n1`, `n2`, the reward
difference `dR = mu_left - mu_right`, the information difference `dI` (+1 if the left
arm was observed fewer times, -1 if more, 0 in games with equal prior observations),
the uncertainty difference, `Delta_mu`, `Delta_V`, `V_q` under q = U_D, the horizon,
and the left/non-greedy choice. It also defines the choice-rule library listed above
(`REPLICATE`, `NEW`, `LAPSE`, `LINK`). `fit` maximizes the likelihood with L-BFGS-B
followed by Nelder-Mead from two starts.

### `model_selection.py`

Random-effects Bayesian model selection over participants (variational scheme of
Stephan et al., 2009) with log evidence approximated by `-AIC/2` or `-BIC/2`: expected
model frequencies, exceedance probabilities from 200,000 Dirichlet draws, the Bayesian
omnibus risk (BOR), and protected exceedance probabilities (Rigoux et al., 2014).

### `wu_envs.py`

`ENV1D`, `ENV2D` (paths of the environment files) and `load_envs(paths, ndim,
gridside)`, which reads the fixed reward environments of Wu et al. (2018) and returns,
for each kernel and environment index, the true reward of every cell. Used to locate
the true optimum.

## Common definitions

- **Non-greedy indicator** (the realization of condition (a)):
  `s_t = 1{x_t != x*_t}`, where `x*_t = argmax mu_{t-1}` is the posterior-greedy arm.
- **Comparison arm** `x`: the second-ranked alternative by posterior mean. In the Wilson
  task this is the other arm. Both `x*_t` and `x` are fixed from the time-(t-1)
  posterior before the choice is made.
- **Gap and standardized gap**: `Delta_mu = mu(x*_t) - mu(x)`,
  `Delta_V = sigma^2(x*_t) + sigma^2(x) - 2 sigma(x*_t, x)`, and
  `dmu_std = Delta_mu / sqrt(Delta_V)` (the regressor written as tilde Delta_mu in the
  paper).
- **Model specification**: `gamma = 1` and `q = U_D` throughout, unless a script
  states otherwise (Section 7.2).
- **Information condition of the Wilson first free choice**: games with equal prior
  observations (two forced observations of each arm) and games with unequal prior
  observations (one and three), split by whether the non-greedy choice goes to the
  more-observed or the less-observed arm.
- **Subject clusters** in the pooled analysis: G = 221 (60 Wilson + 81 Wu 1D + 80 Wu 2D).

## Scripts

The pipeline has three dependency tiers. Tier 0 reconstructs the posteriors from the
raw data and writes per-trial files. Tier 1 pools them. Tier 2 holds the remaining
analyses, each reading a Tier 0/1 output or the raw data directly, so their order is
irrelevant (it is fixed in `make_all.py` only for reproducibility).

### Tier 0: per-trial reconstruction

#### `run_wilson.py`

Reconstructs the posterior at every free choice of the Wilson Horizon Task (|D| = 2;
67,200 trials). With independent arms and a flat prior, `mu_a` is the sample mean of
arm a's observed rewards and `sigma_a^2 = rho^2 / n_a`, so
`Delta_V = sigma^2_* + sigma^2_alt` and, under q = U_D,
`V_q = (Delta_V + Delta_mu^2) / 4`. The remaining horizon A counts the current trial.
Used by the pooled regression (Appendix F) and the calibration grid (Appendix E).
Output: `wilson_pertrial.csv`.

#### `run_wu_pertrial_priorvar1.py`

For every free choice in the Wu 1D (|D| = 30) and 2D (|D| = 121) tasks, refits the GP
posterior over the full grid from the past observations under the prior-variance-1
specification, then computes V_q under q = U_D and kappa^q and records `s_t`. Each row
of the Wu CSVs is one participant session. In 2D the flat index of cell (x, y) is
`y * 11 + x`, matching the `expand.grid(0:10, 0:10)` convention of the original R code.
Used by the pooled regression (Appendix F). Outputs: `wu1d_pertrial_priorvar1.csv`,
`wu2d_pertrial_priorvar1.csv`.

#### `run_indicator_build.py`

Rebuilds every Wu free choice under the three GP specifications and every Wilson free
choice, keeping for each trial the chosen cell, the previous cell, the number of
earlier samples of the chosen cell, the posterior-greedy cell and whether it is tied,
the true optimum `x*_g` and the true values, the gap statistics, and the posterior
means before and after the reward at t. Outputs: `indicator_wu_pertrial.csv`,
`indicator_wilson_pertrial.csv`, `indicator_build.txt`.

#### `run_wu_multinomial_build.py`

Stores, for every Wu free choice under the three GP specifications, the posterior mean
and variance of all cells, the chosen cell, A, T, and V_q under q = U_D. Outputs:
`wu_mn_{wu,priorvar1,priorvar300}_{Wu1D,Wu2D}.npz`, `wu_mn_build.txt`.

### Tier 1: pooled analysis (Appendices D and F)

#### `run_pooled_logit.py`

Combines `wilson_pertrial.csv` and the two prior-variance-1 Wu files (96,120 trials) and
estimates the pooled logit `non_greedy ~ log A + log|D| + dmu_std` with
subject-clustered SEs (Table F.1), and the same model with the gap unstandardized,
divided by the task noise rho, and z-scored within task (Appendix F). Also writes the
non-greedy rate and mean V_q at A >= 2 by dataset under the prior-variance-1
specification (Appendix D). Outputs: `all_pertrial.csv`, `cross_dataset_Aok.csv`,
`pooled_logit.txt`, `pooled_logit_robust.txt`.

### Tier 2: Wilson task (Sections 7.3, 7.4, 7.6; Appendices E, G, H)

#### `run_competitors.py`

Games with equal prior observations and all games: the future-path model (q = U_D,
side bias, lapse; rho = 8 and rho free); a softmax with a temperature per horizon (the
choice-level form of a drift-diffusion model whose threshold depends on the horizon);
softmax or probit on Bayes-optimal finite-horizon values with one temperature (computed
by backward induction on the mean difference and the two sample counts, flat prior,
rho = 8, Gauss-Hermite quadrature with 31 nodes); value-free random exploration (a
softmax with one temperature and a lapse rate per horizon); and combinations with a
horizon-specific lapse. Reports LL, AIC, BIC, cross-validated LL, predicted rates,
parameters, the slope of the optimal value gap on the posterior-mean gap, and the
normative information bonus. Output: `competitors.txt`.

#### `run_info_condition.py`

Observed non-greedy rates by information condition and horizon (Table 4); the Wilson
et al. (2014) model fitted separately to every participant and horizon (information
bonus and decision noise, with sign and rank tests); predicted rates and prediction
errors by condition for the rules of `wilson_choice.py` (Table H.2); and the models
of Table 3 that are not in `competitors.txt` (horizon-specific probit, and the
future-path model without V_q). Outputs: `info_condition.txt`,
`info_condition_cells.csv`, `info_condition_subject_params.csv`.

#### `run_equal_games_robust.py`

Games with equal prior observations, future-path model (q = U_D) against the
horizon-specific softmax (both with side bias and lapse): participant-level AIC and BIC,
random-effects model selection, and the distribution of the pooled AIC difference over
50 data sets simulated from each model with per-participant parameters (clipped to the
10th to 90th percentiles and shrunk toward the pooled estimate by 0, 0.5, or 1).
Output: `equal_games_robust.txt`.

#### `run_equal_games_checks.py`

Games with equal prior observations: the future-path model with gamma free, the
Thompson rule with a temperature per horizon, both comparisons without the participant
with the lowest accuracy, and the accuracy of every participant (share of free choices
of the arm with the higher true mean). Output: `equal_games_checks.txt`.

#### `run_valuefree.py`

Comparison between the future-path model (rho = 8 and rho free) and value-free random
exploration, in games with equal prior observations and in all games (with an
information bonus in both): predicted horizon effect by gap quintile,
participant-level AIC and BIC with random-effects model selection, and the distribution
of the AIC difference over 50 simulated data sets per generator and heterogeneity
level. Output: `valuefree.txt`.

#### `run_shape_test.py`

Tests whether the horizon effect across the five gap quintiles of the games with equal
prior observations matches the profile predicted by each model: the future-path model
with rho free and with rho = 8, value-free random exploration, and the
horizon-specific softmax (all with side bias and lapse). Wald chi-square test with the
covariance of (observed - predicted) from a participant bootstrap that refits every
model in each resample (2,000 resamples by default; `python run_shape_test.py 500` for
fewer), and two shape contrasts (linear trend across quintiles and the hump). About 4
minutes on 20 cores. Output: `shape_test.txt`.

#### `run_fig_horizon_effect.py`

Figure 1. Panel A: games with equal prior observations, split into quintiles of the
posterior-mean gap; the observed horizon effect and the values predicted by the
future-path model (rho free), value-free random exploration, and the horizon-specific
softmax. Panel B: P(non-greedy) by information condition and horizon, observed and
predicted by the same models with an information bonus. Observed 95% intervals are
from a participant-level bootstrap (2,000 resamples). Outputs:
`figs/fig_horizon_effect.{pdf,png}`, `fig_horizon_effect_data.txt` (fits, plotted
values, trials per bin).

#### `run_wilson_models.py`

Fits the choice-rule library of `wilson_choice.py` to the 19,200 first free choices:
the rules without side bias (Table H.1, top panel), the rules with side bias (Table H.1,
middle panel), the link diagnostics (Table H.1, bottom panel), and the rules with a
lapse component (Table 5). Reports LL, AIC, BIC, the implied horizon effect, the
five-fold subject-level cross-validated LL, predicted rates by horizon and information
condition, and all parameter estimates. Outputs: `wilson_models.txt`,
`wilson_models.csv`, `wilson_models_cells.csv`, `wilson_models_params.csv`.

#### `run_rho_free_cv.py`

The future-path model with a horizon-specific information bonus, side bias, and lapse,
on all 19,200 first free choices with rho = 8 and with rho free: LL, AIC, BIC, and the
cross-validated LL on the folds of `run_wilson_models.py` (first two rows of Table 5).
Output: `rho_free_cv.txt`.

#### `run_assumptions.py`

Section 7.6: a concentrated future path `q_theta ∝ exp(theta mu)`; observation noise
with rho free, with a profile likelihood over rho; and the exact mixture choice
probability against the moment-matched probit. For |D| = 2 and q = U_D the exact
probability is `sum_k P(k) Phi((1 + k) m / sqrt((1 + k)^2 Delta_V + 2 A rho^2))`, where
`k` is the difference of two independent Binomial(A - 1, 1/2) counts. Also reports the
predictions of both rules by gap quintile and horizon (`assumptions_gap.py`) and the
share of mixture components with `1 + k <= 0`. Output: `assumptions.txt`.

#### `run_calib_sensitivity.py`

Appendix E, Table E.1: the calibration of `1 - Phi(kappa^q)` on the first free choice
(n = 19,200) over a 27-point grid of gamma in {0.7, 0.85, 1.0}, rho / rho_task in
{0.5, 1.0, 1.5}, and `q_theta(x) ∝ exp(theta mu_{t-1}(x))` with theta in
{0, 0.05, 0.1}. For |D| = 2 everything is closed-form:

```
w_theta      = logistic(theta * dmu)                    (weight on the greedy arm)
V_q(q_theta) = w_theta (1 - w_theta) (Delta_V + dmu^2)
p_nongreedy  = 1 - Phi( dmu / sqrt(Delta_V + 2 (S-1) V_q + 2 S rho^2) )
```

For each grid point the CSV reports mean V_q, the mean predicted rate, the observed
rate, the calibration slope (OLS of the binned empirical rate on the binned prediction
across deciles), and the Cox slope and intercept. Output: `calib_sensitivity_grid.csv`.

#### `run_recovery.py`

Appendix G. Model and parameter recovery on the design of the first free choice (the
19,200 covariate sets are kept and only choices are simulated).

- Model recovery: 100 data sets from each rule of the `nolapse` and `lapse` families,
  each refit by every rule of its family; confusion and inversion matrices under AIC
  and BIC (Table G.1), and the distribution of selected AIC differences.
- Parameter recovery: 200 parameter sets each for the future-path model with (tau,
  lambda), (tau, theta), (tau, theta, gamma) at t = 5, (tau, theta, gamma) on all free
  choices (simulating whole games from the true means), and the future-path model with
  an information bonus and lapse; correlations, bias, RMSE, median absolute error
  (Table G.2), and the median absolute error of theta for true theta <= 0.2.

Optional arguments: `python run_recovery.py [model|param|all] [N_SIM] [N_PAR]`.
Outputs: `recovery_model_{nolapse,lapse}.{txt,csv}`, `recovery_param.{txt,csv}`.

#### `run_recovery_hetero.py`

Appendix G, Table G.3. Fits every rule of both families to each participant separately,
then simulates 50 data sets per rule with per-participant parameters (clipped to the
10th to 90th percentiles; shrink 1 and 0.5 toward the pooled estimate) and refits the
pooled rules. Outputs: `recovery_hetero.txt`, `recovery_hetero_s0.5.txt`,
`recovery_hetero_{nolapse,lapse}{,_s0.5}.csv`, `subject_fits_{nolapse,lapse}.csv`.

#### `run_subject_level.py`

Appendix G, Table G.4. From `subject_fits_lapse.csv`: summed AIC and BIC over
participants, the number of participants best fit by each rule, and random-effects
model selection. Output: `subject_level.txt`.

### Tier 2: Wu task (Sections 7.5, 7.7; Appendices I)

#### `run_wu_multinomial.py`

Section 7.5.1, Table 6, Table I.1. Multinomial softmax over all cells of the Wu grid for
the three GP specifications: constant temperature, temperature per horizon bin, the
future-path temperature `tau sqrt(mean sigma^2 + (A - 1) lambda V_q + A kappa rho^2)`
and its variants, GP-UCB with beta depending on A, and lapse versions, with five-fold
subject-level cross-validation and nested likelihood-ratio tests. Without arguments it
runs the six specification x dataset combinations in parallel and then writes the
report; `python run_wu_multinomial.py {wu|priorvar1|priorvar300} {Wu1D|Wu2D}` runs one
combination and `python run_wu_multinomial.py report` only the report. Outputs:
`wu_mn_models_{spec}_{dataset}.csv`, `wu_mn_models.csv`, `wu_mn_models.txt`.

#### `run_indicator.py`

From the Tier 0 reconstructions, for each GP specification: the composition of
non-greedy choices (repeat of the previous cell, revisit of an earlier cell,
never-sampled cell) and its horizon logits (Section 7.5.2, Table I.2); non-greedy rates
by bins of A; condition (b) of Definition 1 computed directly, its rate when
`mu(x_t) > mu(x*_g)` before the choice, and its calibration against Proposition 5
(Section 7.7); the choice that follows a non-greedy choice; ties in the posterior mean
(Appendix I); mean V_q and V_q in units of the noise variance (Table D.1); and, for the
Wilson task, the same quantities at the first free choice. Output: `indicator.txt`.

#### `run_ucb_baseline.py`

Section 7.5.3, Table 7, Appendix I. Simulates the GP-UCB model of Wu et al. (2018) on
every participant's own environments, horizons, and initial tile (20 replications per
round), using the model definition of the published code (rewards normalized as
(y - 50)/100, prior mean 0, prior variance 1, noise variance 1e-4, UCB = mu + beta sigma
with values <= 0 set to 1e-4, softmax with temperature tau, and, for the localized
model, utilities multiplied by the inverse Manhattan distance to the previous choice)
and each participant's published parameter estimates (median over cross-validation
folds per participant and horizon). Random agents are included as baselines. For
agents and humans alike, the greedy reference is computed under the Wu specification
and under the prior-variance-1 specification, and for each reference the script
reports the share of non-greedy choices that go to cells with lower true values, the
share that find the optimum, both by remaining horizon, and horizon logits with and
without the standardized gap (`ucb_horizon.py`). Outputs: `ucb_baseline.txt`,
`ucb_baseline_pertrial.csv.gz` (not included in the repository; see Quick start).

#### `run_prior_lockin.py`

Section 7.7, Table I.3. For the Wu data, the share of cells whose true value lies above
the prior mean under each GP specification, and how often the lock-in condition
`mu_t(x*_g) < f(x)` holds at non-greedy choices and at the start of end-of-round
persistent runs. For the Wilson H = 6 games, how often the forced-trial mean of the
better arm lies below the true mean of the worse arm (the lock-in condition) and how
persistence on the worse arm depends on it. Output: `prior_lockin.txt`.

#### `run_fig_wu_appendix.py`

Figure I.1, under the Wu specification, from `indicator_wu_pertrial.csv`. Left:
non-greedy rate by 8 within-dataset quantile bins of the posterior-mean gap at A >= 2,
with 95% binomial intervals. Right: mean V_q by remaining horizon A, split by dataset
and horizon condition T. `python run_fig_wu_appendix.py priorvar1` draws the same
figure under the prior-variance-1 specification. Outputs:
`figs/fig_wu_appendix.{pdf,png}`, `fig_wu_appendix_data.txt` (bin rates and trial
counts, rates and V_q by A, the 2D/1D ratio of V_q).

### Tier 2: both tasks (Appendices B and D)

#### `run_fig_cross_dataset.py`

Figure D.1: non-greedy rate against A by dataset (left) and at A >= 2 against |D|
(right). The Wilson trials come from `all_pertrial.csv`; the Wu trials come from
`indicator_wu_pertrial.csv` under the Wu specification. Outputs:
`figs/fig_cross_dataset.{pdf,png}`, `fig_cross_dataset_data.txt`.

#### `run_probit_error_horizons.py`

Appendix B, Table B.1. Absolute difference between the moment-matched probit and the
exact future-path choice probability (tau = 1, no bias, q = U_D, gamma = 1) at every
remaining horizon. Wilson: every free choice of the H = 6 games, in closed form, by A
and information condition. Wu: Monte Carlo with 50,000 draws per trial on 30 random
trials per A under the Wu and prior-variance-1 specifications, using the pairwise
statistic of the greedy cell against the second-ranked cell; the run then repeats
A = 2 (where the error is largest) on every trial. Outputs:
`probit_error_horizons.txt`, `probit_error_wu_pertrial.csv`,
`probit_error_wu_A2_all.csv`. About 10 minutes.

## Reproducibility note

Running `make_all.py` on a clean checkout reproduces the figures and every output file.
All random numbers are drawn from seeded generators in the main process or from
per-task seeds, so the parallel scripts give the same results for any number of
workers.

## Data sources

Both datasets are redistributed under their original authors' MIT licences; the licence
notices are retained in each subdirectory (`data/*/LICENSE`). Only the files the
pipeline reads are included.

- **Wilson Horizon Task** (Wilson et al., 2014; Somerville et al., 2017), as compiled by
  Feng, Wang, Zarnescu and Wilson (2021), *The dynamics of explore-exploit decisions
  reveal a signal-to-noise mechanism for random exploration*, Scientific Reports,
  https://doi.org/10.1038/s41598-021-82530-8.
  Upstream: https://github.com/sffeng/horizon_ddm.
  File used: `data/horizon_ddm/allHorizonData_cut.csv` (60 participants, 160 games per
  horizon condition; 19,200 first free choices and 67,200 free choices in total).
- **Wu et al. GP bandit**: Wu, Schulz, Speekenbrink, Nelson and Meder (2018),
  *Generalization guides human exploration in vast decision spaces*, Nature Human
  Behaviour. Upstream: https://github.com/charleywu/gridsearch.
  Files used: `data/gridsearch/analysis{1D,2D}/ExperimentData/experimentData{1D,2D}.csv`
  (subject choices; 9,720 trials in 1D and 19,200 in 2D),
  `data/gridsearch/experiment{1D,2D}/kernel{1,2}.json` (the fixed reward environments,
  used to locate the true optimum), and
  `data/gridsearch/analysis{1D,2D}/modelResults/paramEstimates.csv` (the published
  cross-validated parameter estimates of all models, of which `run_ucb_baseline.py`
  uses the UCB rows).

## Licence

The analysis code in `code/` and the outputs in `results/` are released under the MIT
Licence (see `LICENSE`). The data in `data/` is MIT-licensed by its respective upstream
authors (see the `LICENSE` file in each subdirectory).
