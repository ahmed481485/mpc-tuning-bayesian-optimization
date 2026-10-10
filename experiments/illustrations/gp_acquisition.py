"""
Illustration: GP posterior of a toy 1D function and its acquisition function.

    python experiments/illustrations/gp_acquisition.py
"""
import torch
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.means import ConstantMean
from torch.distributions import Normal
import matplotlib.pyplot as plt

from pendulum_mpc_bo.plotting import use_style, save_figure

# =========================================================
# USER SETTINGS
# =========================================================
lengthscale  = 1.0
n_samples    = 5
res          = 500
mean_const   = 2.0
outputscale  = 1.0

color_real   = '#333333'
color_gp     = '#6a5acd'
color_acq    = "#100F185E"  # acquisition curve

# Domain
x_min, x_max = 0.0, 5.0


# ---- True underlying function ----
def true_function(x):
    return torch.sin(1.5 * x) + 0.3 * x + 1.5


# ---- Observations ----
X_train_points = [0.5, 1, 3, 3.9078]

# ---- Noise ----
use_noise                 = False
noise_var                 = 0.05
add_noise_to_observations = True

# ---- Bayesian Optimization settings ----
acquisition_type = 'EI'    # 'EI', 'UCB', or 'PI'
maximize         = False   # True -> maximize objective, False -> minimize
xi_explore       = 0.01    # exploration offset for EI / PI
kappa_ucb        = 2.0     # exploration weight for UCB
# =========================================================


def gp_predict(X_train, Y_train, X_test, mean_module, covar_module, noise_var=0.0):
    mu_test = mean_module(X_test).detach()
    K_ss    = covar_module(X_test).evaluate().detach()

    if X_train is None or X_train.numel() == 0:
        return mu_test, K_ss

    mu_train = mean_module(X_train).detach()
    K_ff = covar_module(X_train).evaluate().detach()
    K_fs = covar_module(X_train, X_test).evaluate().detach()

    n = X_train.shape[0]
    K_ff_noisy = K_ff + noise_var * torch.eye(n, dtype=torch.double) \
                      + 1e-8 * torch.eye(n, dtype=torch.double)

    L = torch.linalg.cholesky(K_ff_noisy)
    alpha = torch.cholesky_solve((Y_train - mu_train).unsqueeze(-1), L)
    v     = torch.cholesky_solve(K_fs, L)

    mean_post = mu_test + (K_fs.T @ alpha).squeeze(-1)
    cov_post  = K_ss - K_fs.T @ v
    return mean_post, cov_post


def acquisition(mean, std, Y_train, acq_type='EI', maximize=True, xi=0.01, kappa=2.0):
    """Evaluate an acquisition function over the test grid.

    Returns a tensor the same shape as `mean`; BO picks the argmax.
    For minimization, the formulas are re-expressed so the argmax
    always corresponds to the next query.
    """
    std_safe = std.clamp(min=1e-9)
    standard_normal = Normal(torch.tensor(0.0, dtype=torch.double),
                             torch.tensor(1.0, dtype=torch.double))

    if maximize:
        f_best = Y_train.max()
        improvement = mean - f_best - xi
    else:
        f_best = Y_train.min()
        improvement = f_best - mean - xi

    if acq_type == 'EI':
        z  = improvement / std_safe
        ei = improvement * standard_normal.cdf(z) \
             + std_safe * torch.exp(standard_normal.log_prob(z))
        # Zero out EI where predictive variance collapses
        return torch.where(std > 1e-9, ei, torch.zeros_like(ei))

    elif acq_type == 'PI':
        z = improvement / std_safe
        return standard_normal.cdf(z)

    elif acq_type == 'UCB':
        # Upper Confidence Bound (maximize) / negated Lower CB (minimize)
        if maximize:
            return mean + kappa * std
        else:
            return -(mean - kappa * std)

    else:
        raise ValueError(f"Unknown acquisition type: {acq_type}")


def main():
    use_style(linewidth=1)
    X_test = torch.linspace(x_min, x_max, res, dtype=torch.double).unsqueeze(-1)
    torch.manual_seed(41)

    X_train = torch.tensor(X_train_points, dtype=torch.double).unsqueeze(-1)
    Y_train = true_function(X_train.squeeze(-1))
    if add_noise_to_observations and use_noise:
        Y_train = Y_train + torch.randn_like(Y_train) * noise_var**0.5

    # ---- Build GP and compute posterior ----
    mean_module  = ConstantMean().double()
    covar_module = ScaleKernel(MaternKernel(1.5)).double()
    mean_module.constant                 = torch.tensor(mean_const, dtype=torch.double)
    covar_module.outputscale             = outputscale
    covar_module.base_kernel.lengthscale = lengthscale

    nv = noise_var if use_noise else 0.0
    mean, cov = gp_predict(X_train, Y_train, X_test, mean_module, covar_module, noise_var=nv)

    std  = torch.sqrt(torch.clamp(torch.diag(cov), min=0.0))
    x_np = X_test.squeeze().numpy()

    # ---- Acquisition + next candidate selection ----
    acq_vals = acquisition(mean, std, Y_train,
                           acq_type=acquisition_type, maximize=maximize,
                           xi=xi_explore, kappa=kappa_ucb)

    best_idx     = torch.argmax(acq_vals).item()
    x_next       = X_test[best_idx].item()
    acq_at_next  = acq_vals[best_idx].item()
    mean_at_next = mean[best_idx].item()
    std_at_next  = std[best_idx].item()

    # ---- Figure 1: GP posterior ----
    fig, ax = plt.subplots()

    # True function
    y_real = true_function(X_test.squeeze(-1))
    ax.plot(x_np, y_real.numpy(), color=color_real, linestyle='--', linewidth=1.2, label='Real function')

    # GP mean + 95% CI
    ax.fill_between(x_np,
                    (mean - 1.96 * std).numpy(),
                    (mean + 1.96 * std).numpy(),
                    color=color_gp, alpha=0.2, label=r'$95\%$ CI', linewidth=0)
    ax.plot(x_np, mean.numpy(), color=color_gp, lw=2, alpha=0.9, label='GP mean')

    # Realizations
    if n_samples > 0:
        L = torch.linalg.cholesky(cov + 1e-6 * torch.eye(res, dtype=torch.double))
        for s in range(n_samples):
            z = torch.randn(res, dtype=torch.double)
            sample = mean + L @ z
            ax.plot(x_np, sample.numpy(), color=color_gp, linestyle='-', alpha=0.35, lw=0.9)

    # Observations
    ax.scatter(X_train.squeeze(-1).numpy(), Y_train.numpy(), color='black', zorder=5, s=30, label='observations')

    ax.set_xlabel(r'$\xi$')
    ax.set_ylabel(r'$J(\xi)$')
    ax.grid(True)
    fig.tight_layout()
    save_figure(fig, "illustrations", "acquisition_gp", transparent=True)

    # ---- Figure 2: acquisition function ----
    acq_label = {'EI':  r'Expected Improvement',
                 'PI':  r'Probability of Improvement',
                 'UCB': r'Upper Confidence Bound'}[acquisition_type]

    fig2, ax2 = plt.subplots()
    ax2.plot(x_np, acq_vals.numpy(), color=color_acq, lw=1.5, alpha=0.9, label=acq_label)
    ax2.set_xlabel(r'$\xi$')
    ax2.set_ylabel(r"$\text{EI}(\boldsymbol{\xi})$")
    ax2.grid(True)
    fig2.tight_layout()
    save_figure(fig2, "illustrations", "acquisition_ei", transparent=True)

    # ---- Summary ----
    print("\n--- BO iteration summary ---")
    print(f"Acquisition:     {acquisition_type} ({'maximize' if maximize else 'minimize'})")
    print(f"Next candidate:  xi* = {x_next:.4f}")
    print(f"  posterior mean at xi*: {mean_at_next:.4f}")
    print(f"  posterior std  at xi*: {std_at_next:.4f}")
    print(f"  acquisition    at xi*: {acq_at_next:.6f}")
    plt.show()


if __name__ == "__main__":
    main()
