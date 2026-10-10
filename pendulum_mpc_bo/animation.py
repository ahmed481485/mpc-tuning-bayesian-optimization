"""
Animated replay of a closed-loop MPC episode.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from do_mpc.graphics import Graphics

from . import config
from .simulator import L


def run_animation(data, save_gif=False):
    """Show (and optionally save as GIF) the cart-pendulum animation next to the trajectories."""
    fig = plt.figure(figsize=(16, 9))
    gs  = fig.add_gridspec(5, 2)

    ax_anim = fig.add_subplot(gs[:, 0])
    ax_data = [fig.add_subplot(gs[i, 1]) for i in range(5)]

    fig.suptitle('Inverted Pendulum on Cart - MPC Control', fontsize=16)

    # Animation axes
    ax_anim.set_xlim([-2.5, 2.5])
    ax_anim.set_ylim([-1.2, 1.2])
    ax_anim.set_xlabel('Position [m]')
    ax_anim.set_ylabel('Height [m]')
    ax_anim.set_aspect('equal')
    ax_anim.grid(True)
    ax_anim.set_title('Animation')

    # Data plots
    graphics = Graphics(data)
    labels = [
        ('_x', 'x',         'Cart Position [m]'),
        ('_x', 'theta',     'Angle [rad]'),
        ('_x', 'x_dot',     'Cart Velocity [m/s]'),
        ('_x', 'theta_dot', 'Angular Velocity [rad/s]'),
        ('_u', 'F',         'Force [N]'),
    ]
    for ax, (cat, var, ylabel) in zip(ax_data, labels):
        graphics.add_line(cat, var, ax)
        ax.set_ylabel(ylabel)
        ax.grid(True)
    ax_data[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax_data[-1].set_xlabel('Time [s]')

    # Cart + pendulum drawing
    cart_width, cart_height = 0.3, 0.2
    cart = Rectangle((0, 0), cart_width, cart_height, fill=True, color='blue', zorder=2)
    ax_anim.add_patch(cart)
    ax_anim.plot([-2.5, 2.5], [0, 0], 'k-', linewidth=2)
    pendulum_line, = ax_anim.plot([], [], 'o-', linewidth=3, markersize=10, color='red', zorder=1)
    time_text = ax_anim.text(
        0.02, 0.95, '', transform=ax_anim.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    def update(t_now):
        xc = data['_x', 'x'][t_now, 0]
        th = data['_x', 'theta'][t_now, 0]
        cart.set_xy([xc - cart_width / 2, -cart_height / 2])
        pendulum_line.set_data([xc, xc + L * np.sin(th)], [0, L * np.cos(th)])
        time_text.set_text(f'Time: {data["_time"][t_now, 0]:.2f} s')
        graphics.plot_results(t_ind=t_now)
        graphics.reset_axes()
        return cart, pendulum_line, time_text

    anim = animation.FuncAnimation(
        fig, update, frames=len(data['_time']),
        interval=40, blit=False, repeat=True,
    )

    if save_gif:
        directory = config.RESULTS_DIR / "animations"
        directory.mkdir(parents=True, exist_ok=True)
        existing = [int(p.stem.split("_")[1]) for p in directory.glob("animation_*.gif")]
        filename = directory / f"animation_{max(existing, default=0) + 1}.gif"
        anim.save(filename, writer='pillow', fps=25)
        print(f"Saved: {filename}")

    plt.tight_layout()
    plt.show()
