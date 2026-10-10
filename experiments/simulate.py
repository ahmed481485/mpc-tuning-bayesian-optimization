"""
Simulate one swing-up with given MPC weights and plot θ, h and F.

    python experiments/simulate.py
    python experiments/simulate.py --q1 1 --q2 1 --q3 41.7 --q4 1 --R 1 --animate

The default weights are a tuned set from the thesis.
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from pendulum_mpc_bo.simulator import mpc_loop
from pendulum_mpc_bo.animation import run_animation
from pendulum_mpc_bo.plotting import use_style, save_figure


def plot_states(data):
    t = data['_time'][:, 0]

    fig, ax = plt.subplots(3, sharex=True, figsize=(6, 5), constrained_layout=True)
    ax[0].plot(t, data['_x', 'theta'][:, 0], color="black", alpha=0.8)
    ax[0].set_ylabel(r'$\theta$[rad]')
    ax[0].grid(True)

    ax[1].plot(t, data['_x', 'x'][:, 0], color="black", alpha=0.8)
    ax[1].set_ylabel(r'$h$[m]')
    ax[1].grid(True)

    ax[2].plot(t, data['_u', 'F'][:, 0], drawstyle='steps-post', color="black", alpha=0.8)
    ax[2].set_ylabel(r'$F$ [N]')
    ax[2].set_xlabel(r'$t$ [s]')
    ax[2].grid(True)
    fig.align_ylabels(ax)
    save_figure(fig, "simulate", "states")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--q1', type=float, default=70.4401)
    parser.add_argument('--q2', type=float, default=0.0205)
    parser.add_argument('--q3', type=float, default=41.6705)
    parser.add_argument('--q4', type=float, default=4.9869)
    parser.add_argument('--R', type=float, default=119.0782)
    parser.add_argument('--horizon', type=int, default=50, help="MPC horizon N")
    parser.add_argument('--time', type=float, default=30, help="maximum simulation time [s]")
    parser.add_argument('--animate', action='store_true', help="show the pendulum animation")
    parser.add_argument('--save-gif', action='store_true', help="save the animation as GIF")
    args = parser.parse_args()

    Q = np.array([args.q1, args.q2, args.q3, args.q4])
    data = mpc_loop(Q, args.R, args.horizon, args.time)

    print("Simulation complete!")
    print(f"Final cart position: {data['_x', 'x'][-1, 0]:.4f} m")
    print(f"Final pendulum angle: {data['_x', 'theta'][-1, 0] * 180 / np.pi:.4f} deg")

    use_style(linewidth=1)
    plot_states(data)
    if args.animate or args.save_gif:
        run_animation(data, save_gif=args.save_gif)
    plt.show()


if __name__ == "__main__":
    main()
