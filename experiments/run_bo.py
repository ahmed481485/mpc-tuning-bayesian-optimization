"""
Tune MPC cost weights with Bayesian optimisation.

    python experiments/run_bo.py --dim 1 --objective time
    python experiments/run_bo.py --dim 2 --objective combined
    python experiments/run_bo.py --dim 5 --objective effort

Plots: GP surrogate (1D) / GP-mean contour (2D) and the best-cost-so-far
curve. The ground-truth cost is overlaid when its cache file exists
(see ground_truth.py).
"""
import argparse
import pickle

import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata

from pendulum_mpc_bo import config
from pendulum_mpc_bo.bo import run_bo, fit_gp
from pendulum_mpc_bo.plotting import use_style, save_figure

# Per-objective axis limits / legend position of the 1D plot
YLIM_1D   = {'time': (1.8, 15.8), 'effort': (6, 75), 'combined': (8, 62)}
LEGEND_1D = {'time': 'best', 'effort': 'lower right', 'combined': 'lower right'}
# Marker size of the convergence plot per dimension
MARKERSIZE = {1: 4, 2: 2.5, 5: 1}


def label(name):
    """'q3' → '$q_{3}$', 'R' → '$R$'"""
    return rf"$q_{{{name[1:]}}}$" if name.startswith('q') else rf"${name}$"


def load_ground_truth(params, objective):
    path = config.ground_truth_file(params, objective)
    if not path.exists():
        print(f"No ground truth at {path} – plotting without it.")
        return None
    with open(path, 'rb') as f:
        print(f"Loaded from cache: {path}")
        return pickle.load(f)


def print_gp_hyperparameters(model):
    """GP hyperparameters in normalised and original space (1D only)."""
    lengthscale = model.covar_module.base_kernel.lengthscale.item()
    outputscale = model.covar_module.outputscale.item()
    noise = model.likelihood.noise.item()
    mean = model.mean_module.constant.item()

    x_coeff = model.input_transform.coefficient.item()
    y_mean = model.outcome_transform.means.item()
    y_std = model.outcome_transform.stdvs.item()

    print("=== Normalized/Standardized Space ===")
    print(f"Lengthscale: {lengthscale:.4f}")
    print(f"Outputscale: {outputscale:.4f}")
    print(f"Noise:       {noise:.4f}")
    print(f"Mean:        {mean:.4f}")

    print("\n=== Original Space ===")
    print(f"Lengthscale: {lengthscale * x_coeff:.4f}")
    print(f"Outputscale: {outputscale * y_std**2:.4f}")
    print(f"Noise:       {noise * y_std**2:.4f}")
    print(f"Mean:        {mean * y_std + y_mean:.4f}")


def plot_1d(model, x_train, y_train, n_init, params, objective):
    """GP mean ± 2σ with the BO samples and the ground-truth cost."""
    real = load_ground_truth(params, objective)

    model.eval()
    x_plot_log = torch.linspace(config.PARAM_MIN, config.PARAM_MAX, 500).unsqueeze(-1)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.posterior(x_plot_log)
        mean = -posterior.mean.squeeze()
        std  =  posterior.variance.sqrt().squeeze()

    x_plot  = 10 ** x_plot_log.squeeze().numpy()
    x_real  = 10 ** x_train.squeeze().numpy()
    y_cost  = -y_train.squeeze().numpy()

    fig, ax = plt.subplots()
    if real is not None:
        ax.plot(10 ** real['x_real'], real['y_real'], color='#333333', linestyle='--', linewidth=1)

    ax.plot(x_plot, mean.numpy(), color='#6a5acd', label='GP mean', alpha=0.9)
    ax.fill_between(
        x_plot,
        (mean - 2 * std).numpy(),
        (mean + 2 * std).numpy(),
        color='#6a5acd', alpha=0.2, label=r'$95\%$ CI', linewidth=0,
    )
    ax.scatter(x_real[:n_init], y_cost[:n_init], color='#E67E22', label='Initial points', zorder=11, s=17)
    ax.scatter(x_real, y_cost, color='#1f3fbd', label='BO samples', zorder=9, s=17)

    ax.set_xscale('log')
    ax.set_xlabel(label(params[0]))
    ax.set_ylabel("Cost")
    ax.set_ylim(*YLIM_1D[objective])
    ax.legend(loc=LEGEND_1D[objective])
    ax.grid(True, which='major')
    fig.tight_layout()
    save_figure(fig, "bo", f"1d_{objective}_gp")


def plot_2d(model, x_train, y_train, n_init, params, objective):
    """Contour of the ground-truth cost and of the GP mean with the BO samples (same colour scale)."""
    real = load_ground_truth(params, objective)

    n_grid = 100
    grid = torch.linspace(config.PARAM_MIN, config.PARAM_MAX, n_grid)
    X1, X2 = torch.meshgrid(grid, grid, indexing='ij')
    x_plot = torch.stack([X1.flatten(), X2.flatten()], dim=1)
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        mean_np = -model.posterior(x_plot).mean.squeeze().reshape(n_grid, n_grid).numpy()

    X1_np, X2_np = X1.numpy(), X2.numpy()
    X1_plot, X2_plot = 10 ** X1_np, 10 ** X2_np
    x_train_np = 10 ** x_train.numpy()

    if real is not None:
        z_real_grid = griddata(real['x_real'], real['y_real'].flatten(), (X1_np, X2_np), method='linear')
        vmin = float(np.nanmin([np.nanmin(z_real_grid), np.nanmin(mean_np)]))
        vmax = float(np.nanmax([np.nanmax(z_real_grid), np.nanmax(mean_np)]))
    else:
        vmin, vmax = float(np.nanmin(mean_np)), float(np.nanmax(mean_np))
    levels = np.linspace(vmin, vmax, 50)

    def contour(z):
        fig, ax = plt.subplots()
        c = ax.contourf(X1_plot, X2_plot, z, levels=levels, cmap=cm.viridis, vmin=vmin, vmax=vmax)
        fig.colorbar(c, ax=ax, format="%.3f")
        ax.set_xlabel(label(params[0]))
        ax.set_ylabel(label(params[1]))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which='major')
        return fig, ax

    # Figure 1: ground-truth cost
    if real is not None:
        fig, _ = contour(z_real_grid)
        fig.tight_layout()
        save_figure(fig, "bo", f"2d_{objective}_real")

    # Figure 2: GP mean with BO samples
    fig, ax = contour(mean_np)
    best = torch.argmax(y_train).item()
    ax.scatter(x_train_np[:, 0], x_train_np[:, 1], c="#22211F", marker='o', s=50, label='iter')
    ax.scatter(x_train_np[:n_init, 0], x_train_np[:n_init, 1], c='#E67E22', marker='o', s=50, label='init', zorder=3)
    ax.scatter(x_train_np[best, 0], x_train_np[best, 1], c="#91290F", marker='o', s=50, label='best', zorder=4)
    fig.tight_layout()
    save_figure(fig, "bo", f"2d_{objective}_gp")


def plot_convergence(y_train, n_init, dim, objective):
    """Best cost so far over the BO iterations (initial points dropped)."""
    bo_best = np.minimum.accumulate(-y_train.squeeze().numpy())[n_init:]
    bo_iters = np.arange(1, len(bo_best) + 1)

    fig, ax = plt.subplots()
    ax.plot(bo_iters, bo_best, '-', color='#1f3fbd', alpha=0.8)
    ax.plot(bo_iters, bo_best, 's', color='#1f3fbd', markersize=MARKERSIZE[dim])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best cost so far')
    ax.grid(True, which='major', alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    save_figure(fig, "bo", f"{dim}d_{objective}_convergence")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dim', type=int, choices=sorted(config.DIMENSIONS), required=True)
    parser.add_argument('--objective', choices=list(config.OBJECTIVES), required=True)
    parser.add_argument('--n-init', type=int, help="number of initial points (default: thesis setting)")
    parser.add_argument('--n-iter', type=int, help="number of BO iterations (default: thesis setting)")
    args = parser.parse_args()

    setup  = config.DIMENSIONS[args.dim]
    params = config.tuned_params(args.dim, args.objective)
    n_init = args.n_init if args.n_init is not None else setup['n_init']
    n_iter = args.n_iter if args.n_iter is not None else setup['n_iter']
    print(f"Optimising {params} for objective '{args.objective}' ({n_init} initial points + {n_iter} BO iterations)")

    x_train, y_train = run_bo(params, args.objective, n_init, n_iter, setup['init'],
                              config.OBJECTIVES[args.objective]['sobol_seed'])

    # y_train stores the negated cost → argmax = best
    best = torch.argmax(y_train).item()
    print("\n" + "=" * 50)
    print("  Optimal parameters found")
    print("=" * 50)
    for name, v in zip(params, x_train[best].tolist()):
        print(f"  {name:>4} = {10 ** v:.4f}")
    print(f"  Cost = {-y_train[best].item():.4f}")
    print(f"  Found at evaluation {best + 1} of {len(y_train)}")
    print("=" * 50)

    use_style()
    model = fit_gp(x_train, y_train)
    if args.dim == 1:
        print_gp_hyperparameters(model)
        plot_1d(model, x_train, y_train, n_init, params, args.objective)
    elif args.dim == 2:
        plot_2d(model, x_train, y_train, n_init, params, args.objective)
    plot_convergence(y_train, n_init, args.dim, args.objective)
    plt.show()


if __name__ == "__main__":
    main()
