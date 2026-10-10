"""
BO cost of one MPC episode for the three objectives.
"""
import numpy as np

from . import config
from .simulator import mpc_loop, is_upright, dt, F_max


def evaluate(weights, objective, horizon=config.DEFAULT_HORIZON, alpha=config.ALPHA):
    """
    Run one closed-loop episode and compute its BO cost (lower is better).

    Parameters
    ----------
    weights   : dict – MPC weights to change, e.g. {'q3': 12.5};
                missing weights use config.DEFAULT_WEIGHTS
    objective : 'time' | 'effort' | 'combined'
    horizon   : MPC prediction horizon N
    alpha     : weight of the time term in the combined objective

    Returns
    -------
    cost         : float
    success      : bool  – ended upright with the cart centred
    time_to_goal : float – time term (without penalty)
    effort       : float – effort term (without penalty)
    """
    w = {**config.DEFAULT_WEIGHTS, **weights}
    Q = np.array([w['q1'], w['q2'], w['q3'], w['q4']])

    data = mpc_loop(Q, w['R'], horizon, config.OBJECTIVES[objective]['max_sim_time'])
    t     = data['_time']
    h     = data['_x', 'x']
    theta = data['_x', 'theta']
    u     = data['_u', 'F']

    # The episode stops after 50 steps (2 s) in the goal, so subtract them
    time_to_goal = t[-1][0] + dt - 2
    effort       = np.sum((u / F_max)**2)
    success      = bool(is_upright(theta[-1, 0], h[-1, 0]))

    return bo_cost(objective, time_to_goal, effort, success, alpha), success, time_to_goal, effort


def bo_cost(objective, time_to_goal, effort, success, alpha=config.ALPHA):
    """Cost of an episode from its time and effort terms, plus the failure penalty."""
    if objective == 'time':
        cost = time_to_goal
    elif objective == 'effort':
        cost = effort
    else:
        cost = (1 - alpha) * effort + alpha * time_to_goal

    if not success:
        cost += config.OBJECTIVES[objective]['penalty']
    return cost


def decode(x_log, params):
    """log10 search-space point → {param: real value}"""
    return {name: 10 ** float(v) for name, v in zip(params, x_log)}
