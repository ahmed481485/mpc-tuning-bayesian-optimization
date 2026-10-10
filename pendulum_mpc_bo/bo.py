"""
Bayesian optimisation of MPC weights: Matérn-GP surrogate + LogEI.

Internally the BO maximises the negated cost, so `y_train = -cost`.
"""
import random

import numpy as np
import torch
import gpytorch
from botorch import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from gpytorch.kernels import MaternKernel, ScaleKernel
from torch.quasirandom import SobolEngine

from . import config
from .objectives import evaluate, decode

MATERN_NU = 1.5


def set_seeds(seed=config.SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def search_bounds(n_dims):
    return torch.tensor([[config.PARAM_MIN] * n_dims,
                         [config.PARAM_MAX] * n_dims], dtype=torch.float64)


def fit_gp(x_train, y_train):
    """Fit a GP with a Matérn kernel on normalised inputs / standardised outputs."""
    model = SingleTaskGP(
        x_train, y_train,
        input_transform=Normalize(d=x_train.shape[-1]),
        outcome_transform=Standardize(m=1),
        covar_module=ScaleKernel(MaternKernel(nu=MATERN_NU)),
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def get_next_sample(x_train, y_train, epsilon, bounds):
    """Fit a GP, then maximise LogEI to propose the next candidate."""
    model = fit_gp(x_train, y_train)

    adjusted_best = y_train.max() + epsilon

    EI = LogExpectedImprovement(model=model, best_f=adjusted_best)
    candidates, _ = optimize_acqf(
        acq_function=EI,
        q=1,
        bounds=bounds,
        num_restarts=300,
        raw_samples=1000,
        options={"batch_limit": 5, "maxiter": 200},
    )
    return candidates


def initial_points(n_dims, n_init, init, sobol_seed):
    """Evenly spaced grid (1D) or scrambled Sobol points, in log10 space."""
    if init == 'grid':
        return torch.linspace(config.PARAM_MIN, config.PARAM_MAX, n_init).unsqueeze(-1)

    torch.manual_seed(sobol_seed)
    samples = SobolEngine(dimension=n_dims, scramble=True).draw(n_init)
    torch.manual_seed(config.SEED)
    return samples * (config.PARAM_MAX - config.PARAM_MIN) + config.PARAM_MIN


def make_cost_function(params, objective):
    """Wrap `evaluate` for the BO: log10 tensor → (-cost tensor (1, 1), success)."""
    def func(x):
        cost, success, _, _ = evaluate(decode(x.reshape(-1), params), objective)
        return torch.tensor(-cost).reshape(1, 1), success
    return func


def _log(prefix, idx, x, y, params):
    weights = "  ".join(f"{p}={10 ** v:>9.4f}" for p, v in zip(params, x.reshape(-1).tolist()))
    print(f"{prefix} {idx:>3}  {weights}  cost={-y.item():>8.3f}")


def run_bo(params, objective, n_init, n_iter, init, sobol_seed):
    """
    Run initial evaluations + BO iterations.

    Returns
    -------
    x_train : (n_init + n_iter, d) tensor, log10 weights
    y_train : (n_init + n_iter, 1) tensor, negated cost
    """
    set_seeds()
    n_dims  = len(params)
    epsilon = config.OBJECTIVES[objective]['epsilon']
    func    = make_cost_function(params, objective)
    bounds  = search_bounds(n_dims)

    print("____ Calculating cost – initial points ____")
    x_train = initial_points(n_dims, n_init, init, sobol_seed)
    y_list = []
    for i, x in enumerate(x_train):
        y, _ = func(x)
        y_list.append(y)
        _log("Run", i, x, y, params)
    y_train = torch.cat(y_list)

    print("____ BO Run ____")
    for i in range(n_iter):
        candidate = get_next_sample(x_train, y_train, epsilon, bounds)
        y, _ = func(candidate)

        x_train = torch.cat([x_train, candidate])
        y_train = torch.cat([y_train, y])
        _log("BO ", i + 1, candidate, y, params)

    return x_train, y_train
