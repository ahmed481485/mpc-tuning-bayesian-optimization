"""
Illustration: a generic Pareto front with dominated solutions.

    python experiments/illustrations/pareto_illustration.py
"""
import numpy as np
import matplotlib.pyplot as plt

from pendulum_mpc_bo.plotting import use_style, save_figure


def plot_pareto_illustration():
    # ----- Generate a feasible objective space and a Pareto front -----
    rng = np.random.default_rng(3)

    # Pareto front: convex curve from (0.1, 1.0) down to (1.0, 0.1)
    t = np.linspace(0, 1, 60)
    f1_front = 0.1 + 0.9 * t
    f2_front = 0.1 + 0.9 * (1 - t)**2.2

    # Feasible (dominated) cloud, kept above the front
    n_cloud = 220
    f1_cloud = rng.uniform(0.08, 1.15, n_cloud)
    f2_cloud = rng.uniform(0.08, 1.15, n_cloud)

    def front_value(f1):
        tt = np.clip((f1 - 0.1) / 0.9, 0, 1)
        return 0.1 + 0.9 * (1 - tt)**2.2

    mask = f2_cloud > front_value(f1_cloud) + 0.04
    f1_cloud = f1_cloud[mask]
    f2_cloud = f2_cloud[mask]

    # ----- Plot -----
    fig, ax = plt.subplots(figsize=(6, 5))

    # Dominated feasible points
    ax.scatter(f1_cloud, f2_cloud, s=18, color='0.65', alpha=0.7,
               edgecolors='none', label='Feasible solutions', zorder=2)

    # Pareto front curve
    ax.plot(f1_front, f2_front, color='tomato', linewidth=2.0,
            label='Pareto front', zorder=4)

    # A few highlighted Pareto points
    idx_pts = [8, 22, 38, 52]
    ax.scatter(f1_front[idx_pts], f2_front[idx_pts],
               s=45, color='tomato', edgecolors='white',
               linewidths=0.8, zorder=5)

    # Single dominated point highlighted
    dom_pt = (0.75, 0.78)
    ax.scatter(*dom_pt, s=45, color='steelblue', edgecolors='white',
               linewidths=0.8, zorder=5, label='Dominated solution')

    ax.set_xlabel(r'$J_1(\boldsymbol{\xi})$')
    ax.set_ylabel(r'$J_2(\boldsymbol{\xi})$')
    ax.set_xlim(0.0, 1.25)
    ax.set_ylim(0.0, 1.25)
    ax.grid(True, which='major', alpha=0.3)
    ax.legend()

    fig.tight_layout()
    save_figure(fig, "illustrations", "pareto_illustration")
    plt.close(fig)


if __name__ == "__main__":
    use_style()
    plot_pareto_illustration()
