"""Tuning MPC cost weights for pendulum swing-up with Bayesian optimisation."""
import torch

# All computations (grids, GP, BO) run in double precision
torch.set_default_dtype(torch.float64)
