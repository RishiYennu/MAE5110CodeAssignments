# %% [markdown]
# Pendulum swing-up with value iteration
#
# From the repository root, run `uv run scripts/example_value_iteration.py`.
# Build a transition matrix, solve for a torque policy, and simulate the
# continuous pendulum using that policy.

# %% 
# Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from algorithms import build_transition_matrix, value_iteration
from integrators import rk4 as integrator
from models import pendulum as model

# %% 
# Parameters and grid
params = model.generate_params()
initial_state = np.array([-np.pi, 0.0])  # start hanging down, at rest
timestep = 0.01  # integration substep (s)
control_steps = 4  # hold each torque for 0.04 s
sim_time = 20.0  # s
discount = 0.99
SAVE_GIF = False  # Exporting every animation frame takes several seconds.

angle_grid = np.linspace(-np.pi, np.pi, 69)
velocity_grid = np.linspace(-10.0, 10.0, 121)
grid_points = np.stack(np.meshgrid(angle_grid, velocity_grid, indexing="ij"), axis=-1)
torque_limit = 7  # N m: enough to hold the pendulum at roughly pi/4
actions = np.linspace(-torque_limit, torque_limit, 3)
# The solver chooses the first tied action; prefer smaller torques, including zero.
actions = actions[np.argsort(np.abs(actions), kind="stable")]
points = grid_points.reshape(-1, 2)
lower = points.min(axis=0)
upper = points.max(axis=0)

# %% 
# Build the transition matrix
def step(state, torque):
    """Advance one control interval with constant torque, wrapping the angle."""
    step_params = params.copy()
    step_params["torque"] = torque
    for substep in range(control_steps):
        state = integrator(
            model.dynamics, substep * timestep, state, timestep, step_params
        )
    state[0] = (state[0] + np.pi) % (2 * np.pi) - np.pi
    return state


transition_matrix = build_transition_matrix(grid_points, actions, step)

# Reward and value iteration
# Reward depends only on the current state: 1 at upright equilibrium, 0 elsewhere.
upright = np.all(np.isclose(grid_points, [0.0, 0.0]), axis=-1)
reward = np.zeros_like(transition_matrix, dtype=float)
reward[upright] = 1.0  # the same state reward for every action

value, policy = value_iteration(transition_matrix, reward, discount=discount)

# %% 
# Simulate the policy on the continuous pendulum
if np.any(initial_state < lower) or np.any(initial_state > upper):
    raise ValueError("Choose an initial state inside the grid domain.")
time_traj = np.arange(round(sim_time / timestep) + 1) * timestep
state_traj = np.zeros((2, time_traj.size))
state_traj[:, 0] = initial_state
torque_traj = np.zeros(time_traj.size - 1)
simulation_params = params.copy()

for k, t in enumerate(time_traj[:-1]):
    state = state_traj[:, k]
    if k % control_steps == 0:
        distances_squared = np.sum((points - state) ** 2, axis=1)
        node = np.argmin(distances_squared)
        simulation_params["torque"] = actions[policy.flat[node]]

    torque_traj[k] = simulation_params["torque"]
    next_state = integrator(model.dynamics, t, state, timestep, simulation_params)
    next_state[0] = (next_state[0] + np.pi) % (2 * np.pi) - np.pi
    state_traj[:, k + 1] = next_state
    if np.any(next_state < lower) or np.any(next_state > upper):
        print("Simulation stopped: the state left the grid domain.")
        break

# %% 
# Lookup chooses a torque; the simulated state is never snapped onto the grid.
time_traj = time_traj[: k + 2]
state_traj = state_traj[:, : k + 2]
torque_traj = torque_traj[: k + 1]
print(
    f"Final angle: {state_traj[0, -1]:.4f} rad; "
    f"angular velocity: {state_traj[1, -1]:.4f} rad/s."
)

# %% 
# Plot the value, policy, and continuous trajectory
output = Path("output/value_iteration")
output.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
angle_ticks = np.arange(-2, 3) * np.pi / 2
angle_labels = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
velocity_ticks = np.arange(-3, 4) * np.pi
velocity_labels = [
    r"$-3\pi$",
    r"$-2\pi$",
    r"$-\pi$",
    "0",
    r"$\pi$",
    r"$2\pi$",
    r"$3\pi$",
]

value_plot = axes[0, 0].pcolormesh(
    angle_grid,
    velocity_grid,
    value.T,
    shading="nearest",
    vmin=0,
)
fig.colorbar(value_plot, ax=axes[0, 0], label="Discounted return")
axes[0, 0].set(title="Value function", xlabel="Angle (rad)", ylabel="Velocity (rad/s)")

policy_plot = axes[0, 1].pcolormesh(
    angle_grid,
    velocity_grid,
    actions[policy].T,
    shading="nearest",
    cmap="coolwarm",
    vmin=actions.min(),
    vmax=actions.max(),
)
fig.colorbar(policy_plot, ax=axes[0, 1], label="Torque (N m)")
# Break lines where the angle wraps across the edge of the plot.
plot_angles = state_traj[0].copy()
wraps = np.abs(np.diff(plot_angles)) > np.pi
plot_angles[np.flatnonzero(wraps) + 1] = np.nan
axes[0, 1].plot(plot_angles, state_traj[1], "k-", label="Continuous rollout")
axes[0, 1].plot(*initial_state, "ko")
axes[0, 1].set(
    title="Policy and trajectory", xlabel="Angle (rad)", ylabel="Velocity (rad/s)"
)
axes[0, 1].legend()
for phase_axis in axes[0]:
    phase_axis.set_xticks(angle_ticks, angle_labels)
    phase_axis.set_yticks(velocity_ticks, velocity_labels)
    phase_axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))

axes[1, 0].plot(time_traj, plot_angles, label="Angle (rad)")
axes[1, 0].plot(time_traj, state_traj[1], label="Velocity (rad/s)")
axes[1, 0].set(title="Swing-up and stabilization", xlabel="Time (s)")
trajectory_limits = axes[1, 0].get_ylim()
axes[1, 0].set_yticks(velocity_ticks, velocity_labels)
axes[1, 0].set_ylim(trajectory_limits)
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.25)

axes[1, 1].step(time_traj[:-1], torque_traj, where="post")
axes[1, 1].set(title="Applied torque", xlabel="Time (s)", ylabel="Torque (N m)")
axes[1, 1].grid(alpha=0.25)
fig.savefig(output / "pendulum.png", dpi=180)
print(f"Saved plots to {output / 'pendulum.png'}.")
fig  # noqa: B018 — display the figure in the notebook

# %% 
# Animate the pendulum, with zero angle pointing upward.
length = params["length"]
animation_fig, animation_axis = plt.subplots(figsize=(4, 4), layout="constrained")
animation_axis.set(
    xlim=(-1.2 * length, 1.2 * length),
    ylim=(-1.2 * length, 1.2 * length),
    xlabel="x (m)",
    ylabel="y (m)",
    aspect="equal",
)
animation_axis.grid(alpha=0.25)
animation_axis.plot(0, 0, "ko", zorder=3)  # fixed pivot
(rod,) = animation_axis.plot([], [], "o-", linewidth=3, markersize=10)


def draw_frame(index):
    angle = state_traj[0, index]
    rod.set_data([0, length * np.sin(angle)], [0, length * np.cos(angle)])
    animation_axis.set_title(f"Pendulum: t = {time_traj[index]:.2f} s")


fps = 25
frame_stride = max(1, round(1 / (fps * timestep)))
frame_indices = list(range(0, time_traj.size, frame_stride))
if frame_indices[-1] != time_traj.size - 1:
    frame_indices.append(time_traj.size - 1)
animation = FuncAnimation(
    animation_fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
)
if SAVE_GIF:
    animation.save(output / "pendulum.gif", writer=PillowWriter(fps=fps))
    print(f"Saved animation to {output / 'pendulum.gif'}.")
draw_frame(0)
plt.show()
# Display playback controls when this cell is run in a notebook.
plt.rcParams["animation.html"] = "jshtml"
animation  # noqa: B018 — display the animation in the notebook
