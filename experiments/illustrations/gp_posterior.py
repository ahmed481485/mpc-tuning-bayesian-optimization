"""
Illustration: GP prior / posterior with samples for a toy 1D function.

    python experiments/illustrations/gp_posterior.py
"""
import torch
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.means import ConstantMean
import matplotlib.pyplot as plt

from pendulum_mpc_bo.plotting import use_style, save_figure

# =========================================================
# USER SETTINGS
# =========================================================
plot_meanAndSTD = True

lengthscale  = 1.0
n_samples    = 5
res          = 500
mean_const   = 2.0
outputscale  = 1.0
color_real   = '#333333'
color_gp     = '#6a5acd'
# Domain
x_min, x_max = 0.0, 5.0


# ---- True underlying function ----
def true_function(x):
    return torch.sin(1.5 * x) + 0.3 * x + 1.5


# ---- Observations ----
use_observations = True  # <-- set to False for prior-only plot
X_train_points = [0.5, 1, 3, 3.9078]  # ignored if use_observations=False
# ---- Display ----
show_training_points = True   # <-- overlay training points on the plot (prior or posterior)
# ---- Noise ----
use_noise                 = False
noise_var                 = 0.05
add_noise_to_observations = True
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


def main():
    use_style()
    X_test = torch.linspace(x_min, x_max, res, dtype=torch.double).unsqueeze(-1)
    torch.manual_seed(41)
    if use_observations and len(X_train_points) > 0:
        X_train = torch.tensor(X_train_points, dtype=torch.double).unsqueeze(-1)
        Y_train = true_function(X_train.squeeze(-1))
        if add_noise_to_observations and use_noise:
            Y_train = Y_train + torch.randn_like(Y_train) * noise_var**0.5
    else:
        X_train, Y_train = None, None

    # Training points for *display only* (independent of conditioning)
    if show_training_points and len(X_train_points) > 0:
        X_train_display = torch.tensor(X_train_points, dtype=torch.double).unsqueeze(-1)
        Y_train_display = true_function(X_train_display.squeeze(-1))
        if add_noise_to_observations and use_noise:
            Y_train_display = Y_train_display + torch.randn_like(Y_train_display) * noise_var**0.5
    else:
        X_train_display, Y_train_display = None, None

    # ---- Build GP ----
    mean_module  = ConstantMean().double()
    covar_module = ScaleKernel(MaternKernel(1.5)).double()
    mean_module.constant                 = torch.tensor(mean_const, dtype=torch.double)
    covar_module.outputscale             = outputscale
    covar_module.base_kernel.lengthscale = lengthscale
    nv = noise_var if use_noise else 0.0
    mean, cov = gp_predict(X_train, Y_train, X_test, mean_module, covar_module, noise_var=nv)
    std  = torch.sqrt(torch.clamp(torch.diag(cov), min=0.0))
    x_np = X_test.squeeze().numpy()

    # ---- Plot ----
    fig, ax = plt.subplots()
    # True function
    y_real = true_function(X_test.squeeze(-1))
    ax.plot(x_np, y_real.numpy(), color=color_real, linestyle='--', linewidth=1.2, label='Real function')
    # GP mean + 95% CI
    if plot_meanAndSTD:
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
    # Observations / training points
    if show_training_points and X_train_display is not None and X_train_display.numel() > 0:
        ax.scatter(X_train_display.squeeze(-1).numpy(), Y_train_display.numpy(),
                   color='black', zorder=5, s=30, label='observations')
    ax.set_xlabel(r'$\xi$')
    ax.set_ylabel(r'$\mathrm{J}(\xi)$')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    save_figure(fig, "illustrations", "gp_posterior" if X_train is not None else "gp_prior", transparent=True)
    plt.show()


if __name__ == "__main__":
    main()
