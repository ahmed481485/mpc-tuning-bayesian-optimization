"""
Evaluate the true BO cost on a dense log10 grid and cache it
(the "real" cost that the BO plots compare against).

    python experiments/ground_truth.py --dim 1 --objective time
    python experiments/ground_truth.py --params q3 --objective time --horizon 25
    python experiments/ground_truth.py --params q3 R --objective effort

The result is written to data/ground_truth/ and skipped if it already
exists (use --force to recompute). 1D results are also plotted.
"""
import argparse
import pickle
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

from pendulum_mpc_bo import config
from pendulum_mpc_bo.objectives import evaluate, decode
from pendulum_mpc_bo.plotting import use_style, save_figure

# Grid points per dimension (1D: 200, 2D: 31 × 31, 5D: 6^5 = 7776)
POINTS_PER_DIM = {1: 200, 2: 31, 5: 6}


def compute(params, objective, horizon, points):
    n_dims = len(params)
    axis = torch.linspace(config.PARAM_MIN, config.PARAM_MAX, points)
    grid = torch.meshgrid(*[axis] * n_dims, indexing='ij')
    x_real = torch.stack(grid, dim=-1).reshape(-1, n_dims)

    print(f"Computing {len(x_real)} points for {params}, objective '{objective}', N={horizon} …")
    t_start = time.time()

    y_real, x_pen, y_pen, x_suc, y_suc = [], [], [], [], []
    for i, x in enumerate(x_real):
        cost, success, _, _ = evaluate(decode(x, params), objective, horizon=horizon)
        print(f"  {i + 1:>4}/{len(x_real)}  x={np.round(x.tolist(), 3)}  cost={cost:>8.3f}")
        y_real.append(cost)
        if success:
            x_suc.append(x.tolist())
            y_suc.append(cost)
        else:
            x_pen.append(x.tolist())
            y_pen.append(cost)

    x_real = x_real.numpy()
    if n_dims == 1:   # 1D files store flat arrays
        x_real = x_real.squeeze(-1)
        x_pen, x_suc = np.ravel(x_pen), np.ravel(x_suc)
    print(f"Took {time.time() - t_start:.1f} sec")

    return {
        'x_real':           x_real,           'y_real':           np.array(y_real),
        'x_real_penalized': np.array(x_pen),  'y_real_penalized': np.array(y_pen),
        'x_real_success':   np.array(x_suc),  'y_real_success':   np.array(y_suc),
    }


def plot_1d(data, param, name):
    """Cost over the weight with the unsuccessful regions shaded."""
    x_real = 10 ** data['x_real']
    x_rp   = 10 ** data['x_real_penalized']

    fig, ax = plt.subplots()
    ax.plot(x_real, data['y_real'], color='black', alpha=0.8, label='Real cost', linestyle="solid")

    # Group neighbouring unsuccessful grid points into shaded spans
    if len(x_rp) > 0:
        log_step = (config.PARAM_MAX - config.PARAM_MIN) / (len(x_real) - 1)
        split_indices = np.where(np.diff(np.log10(x_rp)) > log_step * 1.5)[0] + 1
        for i, group in enumerate(np.split(x_rp, split_indices)):
            ax.axvspan(group[0], group[-1], color="#9e0e0e", alpha=0.2,
                       label="Unsuccessful" if i == 0 else None, linewidth=0)

    ax.set_xscale('log')
    ax.xaxis.set_minor_locator(LogLocator(subs='auto'))
    ax.set_xlabel(rf"$q_{{{param[1:]}}}$" if param.startswith('q') else rf"${param}$")
    ax.set_ylabel("Cost")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, which='major', alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "ground_truth", name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dim', type=int, choices=sorted(config.DIMENSIONS),
                       help="use the weights tuned by run_bo.py for this dimension")
    group.add_argument('--params', nargs='+', choices=list(config.DEFAULT_WEIGHTS),
                       help="weights to put on the grid, e.g. --params q3 R")
    parser.add_argument('--objective', choices=list(config.OBJECTIVES), required=True)
    parser.add_argument('--horizon', type=int, default=config.DEFAULT_HORIZON, help="MPC horizon N")
    parser.add_argument('--points', type=int, help="grid points per dimension (default: 200 / 31 / 6)")
    parser.add_argument('--force', action='store_true', help="recompute and overwrite the cached file")
    args = parser.parse_args()

    params = args.params or config.tuned_params(args.dim, args.objective)
    points = args.points or POINTS_PER_DIM.get(len(params), 6)
    path = config.ground_truth_file(params, args.objective, args.horizon)

    if path.exists() and not args.force:
        print(f"Already cached – skipping: {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = compute(params, args.objective, args.horizon, points)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved to cache: {path}")

    if len(params) == 1:
        use_style(linewidth=1)
        plot_1d(data, params[0], path.stem)
        plt.show()


if __name__ == "__main__":
    main()
