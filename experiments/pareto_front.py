"""
Pareto front between time to goal and control effort (1D, tuning q3).

    python experiments/pareto_front.py
    python experiments/pareto_front.py --from-cache data/pareto/1d_q3.pkl --alpha-index 6

For every q3 on a 200-point grid (denser around q3 = 10^0.2) one episode
is simulated; the combined cost (1 - α) · effort + α · time is then
evaluated for 20 values of α in [0, 1]. For each α the best grid point
gives one point of the Pareto front.
"""
import argparse
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

from pendulum_mpc_bo import config
from pendulum_mpc_bo.objectives import evaluate, bo_cost, decode
from pendulum_mpc_bo.plotting import use_style, save_figure

ALPHAS = np.linspace(0, 1, 20).tolist()
N_POINTS = 200
CENTER = 0.2   # log10 q3 where the grid is densest
WARP = 4       # higher = more concentration around CENTER


def q3_grid():
    """Nonlinear log10 grid: points concentrate around CENTER."""
    t = torch.linspace(0, 1, N_POINTS)
    left_frac = (CENTER - config.PARAM_MIN) / (config.PARAM_MAX - config.PARAM_MIN)
    mask = t < left_frac
    x_list = torch.empty_like(t)
    s = t[mask] / left_frac
    x_list[mask] = config.PARAM_MIN + (CENTER - config.PARAM_MIN) * (s ** (1 / WARP))
    s = (t[~mask] - left_frac) / (1 - left_frac)
    x_list[~mask] = CENTER + (config.PARAM_MAX - CENTER) * (s ** WARP)
    return x_list.unsqueeze(-1)


def compute():
    # The episode does not depend on α, so simulate each q3 once
    episodes = []
    for index, x in enumerate(q3_grid()):
        _, success, t, e = evaluate(decode(x, ['q3']), 'combined')
        print(f"run:{index:>3}, weight = {x.item():>8.3}, time = {t:>8.3}, effort = {e:>8.3}")
        episodes.append((t, e, success))

    return {
        'alphas': ALPHAS,
        'total_cost': [[bo_cost('combined', t, e, s, alpha) for t, e, s in episodes] for alpha in ALPHAS],
        'timeToGoal_cost': [[t for t, _, _ in episodes] for _ in ALPHAS],
        'minimizeInput_cost': [[e for _, e, _ in episodes] for _ in ALPHAS],
    }


def pareto_points(data):
    """For each α the time and effort of the grid point with the lowest combined cost."""
    best_idx = np.argmin(np.array(data['total_cost']), axis=1)
    rows = np.arange(len(data['alphas']))
    return (np.array(data['timeToGoal_cost'])[rows, best_idx],
            np.array(data['minimizeInput_cost'])[rows, best_idx])


def plot_front(data):
    alphas = data['alphas']
    time_pareto, input_pareto = pareto_points(data)

    # Pareto front coloured by α
    fig, ax = plt.subplots()
    sc = ax.scatter(time_pareto, input_pareto, c=alphas, cmap='viridis', s=40, zorder=3)
    fig.colorbar(sc, ax=ax).set_label(r'$\alpha$')
    ax.set_xlabel("Time to goal")
    ax.set_ylabel("Input cost")
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "pareto", "front")

    # Both objectives vs α
    fig, ax = plt.subplots()
    ax.plot(alphas, time_pareto, 'o-', label='Time to goal', color='steelblue')
    ax.plot(alphas, input_pareto, 's-', label='Input cost', color='tomato')
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Objective value")
    ax.legend()
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "pareto", "objectives_vs_alpha")


def plot_cost(data, alpha_index=None):
    """Combined cost over q3 for every α (or only for `alpha_index`)."""
    x = (10 ** q3_grid()).squeeze().tolist()
    alphas = data['alphas']

    fig, ax = plt.subplots()
    for i, y in enumerate(data['total_cost']):
        if alpha_index is None:
            ax.plot(x, y, label=fr"$\alpha$ = {alphas[i]:<3.2}" if i % 19 == 0 else None)
        elif i == alpha_index:
            ax.plot(x, y, label=fr"$\alpha$ = {alphas[i]:<3.2}")
    ax.grid(True, which="major", alpha=0.3)
    ax.set_xscale('log')
    ax.set_xlabel(r"$q_3$")
    ax.set_ylabel("Cost")
    ax.legend(ncol=2)
    fig.tight_layout()
    save_figure(fig, "pareto", "cost")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--from-cache', help="plot a saved result instead of computing")
    parser.add_argument('--alpha-index', type=int, help="only plot the cost curve of this α")
    args = parser.parse_args()

    if args.from_cache:
        with open(args.from_cache, 'rb') as f:
            data = pickle.load(f)
    else:
        data = compute()
        directory = config.RESULTS_DIR / "pareto"
        directory.mkdir(parents=True, exist_ok=True)
        existing = [int(p.stem.rsplit("_", 1)[1]) for p in directory.glob("1d_q3_*.pkl")]
        path = directory / f"1d_q3_{max(existing, default=0) + 1}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved: {path}")

    for alpha, t, e in zip(data['alphas'], *pareto_points(data)):
        print(f"alpha: {alpha:>5.3}__time_to_goal: {t:>5.3}___input: {e:>5.3}")

    use_style()
    plot_front(data)
    plot_cost(data, args.alpha_index)
    plt.show()


if __name__ == "__main__":
    main()
