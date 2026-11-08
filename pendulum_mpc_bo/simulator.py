"""
Closed-loop simulation: nonlinear MPC (do-mpc) swinging up an
inverted pendulum on a cart.
"""
import numpy as np
import do_mpc
from casadi import sin, cos

# ── Physical model (inverted pendulum on cart) ──────────────
M = 1.0       # cart mass [kg]
m = 0.1       # pendulum mass [kg]
L = 0.5       # pendulum length [m]
g = 9.81      # gravity [m/s²]
b = 0.5       # cart friction

# ── Simulation ──────────────────────────────────────────────
dt      = 0.04
F_max   = 6
F_min   = -6
init_x0 = [0, 0, 0.99 * np.pi, 0]   # [x, x_dot, theta, theta_dot], theta = 0 is upright

# ── Goal ────────────────────────────────────────────────────
success_window_theta = 0.01 * np.pi   # close to upright
success_window_h     = 0.2            # cart close to centre [m]
goal_steps           = 50             # steps inside the goal before early stop


def is_upright(theta, h):
    """Pendulum upright and cart centred."""
    theta_ok = np.abs(theta) <= success_window_theta or np.abs(theta - 2 * np.pi) <= success_window_theta
    cart_ok  = np.abs(h) <= success_window_h
    return theta_ok and cart_ok


def mpc_loop(Q_array, R, N_horizon, max_sim_time):

    np.random.seed(42)

    # ── Model ───────────────────────────────────────────────
    model = do_mpc.model.Model('continuous')

    x         = model.set_variable('_x', 'x')
    x_dot     = model.set_variable('_x', 'x_dot')
    theta     = model.set_variable('_x', 'theta')
    theta_dot = model.set_variable('_x', 'theta_dot')
    F         = model.set_variable('_u', 'F')

    den = M + m * sin(theta) ** 2

    x_ddot = (
        F + m * sin(theta) * (L * theta_dot**2 - g * cos(theta)) - b * x_dot
    ) / den

    theta_ddot = (
        -F * cos(theta)
        - m * L * theta_dot**2 * cos(theta) * sin(theta)
        + (M + m) * g * sin(theta)
        + b * x_dot * cos(theta)
    ) / (L * den)

    model.set_rhs('x', x_dot)
    model.set_rhs('x_dot', x_ddot)
    model.set_rhs('theta', theta_dot)
    model.set_rhs('theta_dot', theta_ddot)
    model.setup()

    # ── MPC controller ──────────────────────────────────────
    mpc = do_mpc.controller.MPC(model)
    mpc.set_param(
        n_horizon=N_horizon,
        n_robust=0,
        open_loop=0,
        t_step=dt,
        state_discretization='collocation',
        collocation_type='radau',
        collocation_deg=3,
        collocation_ni=1,
        store_full_solution=True,
        nlpsol_opts={'ipopt.linear_solver': 'mumps'},
    )
    mpc.settings.supress_ipopt_output()

    # Normalisation scales for the cost
    x_scale         = 2.0
    x_dot_scale     = 2.0
    theta_dot_scale = 10
    F_scale         = F_max

    # Stage & terminal cost (normalised)
    mterm = (
        Q_array[0] * (x / x_scale)**2
        + Q_array[1] * (x_dot / x_dot_scale)**2
        + Q_array[2] * ((1 - cos(theta)) / 2)**2
        + Q_array[3] * (theta_dot / theta_dot_scale)**2
    )
    lterm = mterm + R * (F / F_scale)**2

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(F=0.1)

    mpc.bounds['lower', '_u', 'F'] = F_min
    mpc.bounds['upper', '_u', 'F'] = F_max
    mpc.bounds['lower', '_x', 'x'] = -2.0
    mpc.bounds['upper', '_x', 'x'] =  2.0
    mpc.setup()

    # ── Estimator & Simulator ───────────────────────────────
    estimator = do_mpc.estimator.StateFeedback(model)

    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(integration_tool='idas', abstol=1e-8, reltol=1e-8, t_step=0.04)
    simulator.setup()

    # ── Initial conditions ──────────────────────────────────
    simulator.x0['x']         = init_x0[0]
    simulator.x0['x_dot']     = init_x0[1]
    simulator.x0['theta']     = init_x0[2]
    simulator.x0['theta_dot'] = init_x0[3]

    x0 = simulator.x0.cat.full()
    mpc.x0 = x0
    estimator.x0 = x0
    mpc.set_initial_guess()
    mpc.reset_history()

    # ── Closed-loop run ─────────────────────────────────────
    sim_steps = int(np.round(max_sim_time / dt))
    goal = 0

    for k in range(sim_steps):
        u0     = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0     = estimator.make_step(y_next)

        # Early stop: pendulum upright AND cart centred for 50 steps
        if is_upright(mpc.data['_x', 'theta'][-1, 0], mpc.data['_x', 'x'][-1, 0]):
            goal += 1
            if goal >= goal_steps:
                break
        else:
            goal = 0

    return mpc.data
