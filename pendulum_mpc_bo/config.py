"""
All experiment settings in one place.

The values are the ones used for the thesis results.
"""
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT_DIR / "data"       # cached results shipped with the repo
RESULTS_DIR = ROOT_DIR / "results"    # everything you generate (git-ignored)

# ── Seeds ───────────────────────────────────────────────────
SEED = 1234

# ── Search space (log10 scale for every tuned weight) ───────
PARAM_MIN = -2   # 10^-2 = 0.01
PARAM_MAX = 3    # 10^3  = 1000

# MPC weights that are not tuned keep these values
DEFAULT_WEIGHTS = {'q1': 1, 'q2': 1, 'q3': 1, 'q4': 1, 'R': 1}
DEFAULT_HORIZON = 50

# ── BO cost functions ───────────────────────────────────────
# time:     time to reach the goal
# effort:   sum of squared normalised inputs  Σ (u / F_max)²
# combined: (1 - α) · effort + α · time
#
# Every run that does not end upright with the cart centred gets
# `penalty` added. `epsilon` is the EI exploration offset and
# `sobol_seed` seeds the Sobol initial points (2D / 5D).
ALPHA = 0.263

OBJECTIVES = {
    'time':     {'max_sim_time': 15, 'penalty': 2,  'epsilon': 0.3, 'sobol_seed': 7},
    'effort':   {'max_sim_time': 40, 'penalty': 70, 'epsilon': 1.4, 'sobol_seed': 10},
    'combined': {'max_sim_time': 30, 'penalty': 50, 'epsilon': 1.0, 'sobol_seed': 4},
}

# ── BO setup per dimension ──────────────────────────────────
# `params` lists the tuned weights per objective.
# 1D starts from an evenly spaced grid, 2D / 5D from Sobol points.
DIMENSIONS = {
    1: {
        'params': {'time': ['q3'], 'effort': ['q3'], 'combined': ['q3']},
        'n_init': 5, 'n_iter': 25, 'init': 'grid',
    },
    2: {
        'params': {'time': ['q1', 'q3'], 'effort': ['q3', 'R'], 'combined': ['q3', 'R']},
        'n_init': 10, 'n_iter': 65, 'init': 'sobol',
    },
    5: {
        'params': {obj: ['q1', 'q2', 'q3', 'q4', 'R'] for obj in OBJECTIVES},
        'n_init': 50, 'n_iter': 200, 'init': 'sobol',
    },
}


def tuned_params(dim, objective):
    return DIMENSIONS[dim]['params'][objective]


def ground_truth_file(params, objective, horizon=DEFAULT_HORIZON):
    """Cache file of the dense-grid ("real") cost, e.g. 2d_q3_R_combined.pkl"""
    name = f"{len(params)}d_{'_'.join(params)}_{objective}"
    if horizon != DEFAULT_HORIZON:
        name += f"_N{horizon}"
    return DATA_DIR / "ground_truth" / f"{name}.pkl"
