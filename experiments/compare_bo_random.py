"""
Compare BO against random search when tuning all five weights (5D).

    python experiments/compare_bo_random.py --objective time
    python experiments/compare_bo_random.py --from-cache data/bo_vs_random/5d_time.pkl

Each of the `--runs` statistical runs evaluates 250 points: BO (50 Sobol
points + 200 BO iterations) vs. the same 50 initial points + 200
uniformly random points. Plots mean ± std of the best cost so far.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from pendulum_mpc_bo import config
from pendulum_mpc_bo.bo import run_bo, make_cost_function
from pendulum_mpc_bo.plotting import use_style, save_figure

DIM = 5


def compute(objective, n_runs, n_points):
    setup  = config.DIMENSIONS[DIM]
    params = config.tuned_params(DIM, objective)
    n_init = setup['n_init']
    sobol_seed = config.OBJECTIVES[objective]['sobol_seed']

    all_bo_x, all_bo, all_random_x, all_random = [], [], [], []

    print("──── Calculating BO Runs ────")
    for i in range(n_runs):
        print(f"  {i + 1}/{n_runs} BO run begin \n")
        x_train, y_train = run_bo(params, objective, n_init, n_points - n_init, setup['init'], sobol_seed + i)
        all_bo_x.append(x_train.tolist())
        all_bo.append((-y_train.squeeze()).tolist())

    print("\n──── Calculating Random Runs ────")
    func = make_cost_function(params, objective)
    for i in range(n_runs):
        print(f"  {i + 1}/{n_runs} Random run begin \n")
        torch.manual_seed(i)
        y_random = all_bo[i][:n_init]   # start from the same initial points as BO run i
        x_random = torch.rand(n_points - n_init, DIM, dtype=torch.float64)
        x_random = x_random * (config.PARAM_MAX - config.PARAM_MIN) + config.PARAM_MIN
        for index, x in enumerate(x_random):
            y, _ = func(x)
            print(f"Run Nr. {index + 1:>3} __ value: {-y.item():>4.3f}")
            y_random.append(-y.item())
        all_random_x.append(x_random.tolist())
        all_random.append(y_random)
        torch.manual_seed(config.SEED)

    return {
        'all_bo_x_lists': all_bo_x,
        'all_bo_lists': all_bo,
        'all_random_x_lists': all_random_x,
        'all_random_lists': all_random,
    }


def plot_comparison(all_bo, all_random, n_init, name):
    """Mean ± std of the best cost so far for BO vs random search."""
    bo_best     = np.minimum.accumulate(np.array(all_bo), axis=1)[:, n_init:]
    random_best = np.minimum.accumulate(np.array(all_random), axis=1)[:, n_init:]

    bo_mean,     bo_std     = bo_best.mean(axis=0),     bo_best.std(axis=0)
    random_mean, random_std = random_best.mean(axis=0), random_best.std(axis=0)
    bo_iters     = np.arange(1, bo_best.shape[1] + 1)
    random_iters = np.arange(1, random_best.shape[1] + 1)

    fig, ax = plt.subplots()
    ax.plot(random_iters, random_mean, color="tomato", label=r'Random (mean $\pm$ 1 std)')
    ax.fill_between(random_iters, random_mean - random_std, random_mean + random_std,
                    color="tomato", alpha=0.25, linewidth=0)
    ax.plot(bo_iters, bo_mean, color='steelblue', label=r'BO (mean $\pm$ 1 std)')
    ax.fill_between(bo_iters, bo_mean - bo_std, bo_mean + bo_std,
                    color='steelblue', alpha=0.25, linewidth=0)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best cost so far')
    ax.legend()
    ax.grid(True, which='major', alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    save_figure(fig, "bo_vs_random", name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--objective', choices=list(config.OBJECTIVES))
    group.add_argument('--from-cache', help="plot a saved result instead of computing")
    parser.add_argument('--runs', type=int, default=5, help="number of statistical runs")
    parser.add_argument('--points', type=int, default=250, help="evaluations per run")
    args = parser.parse_args()

    n_init = config.DIMENSIONS[DIM]['n_init']
    if args.from_cache:
        with open(args.from_cache, 'rb') as f:
            data = pickle.load(f)
        name = Path(args.from_cache).stem
    else:
        data = compute(args.objective, args.runs, args.points)
        directory = config.RESULTS_DIR / "bo_vs_random"
        directory.mkdir(parents=True, exist_ok=True)
        existing = [int(p.stem.rsplit("_", 1)[1]) for p in directory.glob(f"5d_{args.objective}_*.pkl")]
        path = directory / f"5d_{args.objective}_{max(existing, default=0) + 1}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved: {path}")
        name = f"5d_{args.objective}"

    use_style()
    plot_comparison(data['all_bo_lists'], data['all_random_lists'], n_init, name)
    plt.show()


if __name__ == "__main__":
    main()
