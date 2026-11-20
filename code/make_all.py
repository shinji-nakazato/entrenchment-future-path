import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = [
    ('run_wilson.py', False),
    ('run_wu_pertrial_priorvar1.py', False),
    ('run_indicator_build.py', False),
    ('run_wu_multinomial_build.py', False),

    ('run_pooled_logit.py', False),

    ('run_pooled_fe.py', False),
    ('run_wild_bootstrap.py', False),
    ('run_calib_sensitivity.py', False),

    ('run_wilson_models.py', False),
    ('run_info_condition.py', False),
    ('run_equal_games_robust.py', False),
    ('run_equal_games_checks.py', False),
    ('run_competitors.py', False),
    ('run_valuefree.py', False),
    ('run_assumptions.py', False),
    ('run_rho_free_cv.py', False),
    ('run_indicator.py', False),
    ('run_prior_lockin.py', False),
    ('run_ucb_baseline.py', False),
    ('run_fig_horizon_effect.py', False),
    ('run_shape_test.py', False),
    ('run_fig_cross_dataset.py', False),
    ('run_fig_wu_appendix.py', False),

    ('run_recovery.py', True),
    ('run_recovery_hetero.py', True),
    ('run_subject_level.py', True),
    ('run_probit_error_horizons.py', True),
    ('run_wu_multinomial.py', True),
]


def main():
    skip_slow = '--skip-slow' in sys.argv[1:]
    for step, slow in STEPS:
        if slow and skip_slow:
            print(f'\n[make_all] skipping {step} (--skip-slow)')
            continue
        print(f'\n{"=" * 78}\n>>> {step}\n{"=" * 78}', flush=True)
        t0 = time.time()
        result = subprocess.run([sys.executable, str(HERE / step)], cwd=str(HERE))
        if result.returncode != 0:
            print(f'\n[make_all] {step} exited with status {result.returncode}; halting.')
            sys.exit(result.returncode)
        print(f'[make_all] {step} done in {(time.time() - t0) / 60:.1f} min', flush=True)
    print('\n[make_all] pipeline complete; all steps succeeded.')


if __name__ == '__main__':
    main()
