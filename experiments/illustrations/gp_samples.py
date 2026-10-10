"""
Illustration: GP prior samples for different Matérn lengthscales.

    python experiments/illustrations/gp_samples.py
"""
import torch
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.means import ConstantMean
import matplotlib.pyplot as plt

from pendulum_mpc_bo.plotting import use_style, save_figure

lengthscales = [0.5, 1, 2]
colors = ['skyblue', 'orange', 'green']
linestyles = ['-', '-', '-']
res = 500


def main():
    use_style()
    X_test = torch.linspace(0, 5, res, dtype=torch.double).unsqueeze(-1)

    torch.manual_seed(41)
    z = torch.randn(res, dtype=torch.double)

    fig, ax = plt.subplots()

    for ls, color, ls_style in zip(lengthscales, colors, linestyles):
        mean_module = ConstantMean().double()
        covar_module = ScaleKernel(MaternKernel(1.5))

        mean_module.constant = torch.tensor(2.0, dtype=torch.double)
        covar_module.outputscale = 1.0
        covar_module.base_kernel.lengthscale = ls

        mean = mean_module(X_test).detach()
        covar = covar_module(X_test).evaluate().detach()

        L = torch.linalg.cholesky(covar + 1e-6 * torch.eye(res, dtype=torch.double))

        # Independent z for each lengthscale
        z = torch.randn(res, dtype=torch.double)
        sample = mean + L @ z

        ax.plot(X_test.squeeze().numpy(), sample.numpy(),
                color=color, linestyle=ls_style,
                alpha=0.9, label=r'$\ell = ' + str(ls) + r'$')

    ax.set_xlabel(r'$\xi$')
    ax.set_ylabel(r'$f(\xi)$')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    save_figure(fig, "illustrations", "gp_samples")
    plt.show()


if __name__ == "__main__":
    main()
