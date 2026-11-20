import os
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / 'data'
RESULTS = ROOT / 'results'
FIGS    = RESULTS / 'figs'

WILSON_CSV = DATA / 'horizon_ddm' / 'allHorizonData_cut.csv'
WU1D_CSV  = DATA / 'gridsearch' / 'analysis1D' / 'ExperimentData' / 'experimentData1D.csv'
WU2D_CSV  = DATA / 'gridsearch' / 'analysis2D' / 'ExperimentData' / 'experimentData2D.csv'
WU_PARAMS_1D = DATA / 'gridsearch' / 'analysis1D' / 'modelResults' / 'paramEstimates.csv'
WU_PARAMS_2D = DATA / 'gridsearch' / 'analysis2D' / 'modelResults' / 'paramEstimates.csv'

RHO_WILSON   = 8.0
RHO_WU_PRIORVAR1      = 5.0
GAMMA        = 1.0
WILSON_FORCED_TRIALS = 4
WU_LSCALE_1D = 0.5
WU_LSCALE_2D = 0.76
WU_GRIDSIDE_1D = 30
WU_GRIDSIDE_2D = 11

N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
