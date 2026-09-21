from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from assignment_2_balance_controller import (
    build_region_of_attraction,
    compute_ankle_torque,
    is_in_region_of_attraction,
)
from integrators.rk4_events import rk4_step
from models import inverted_pendulum_walker as model

# Fixed controls for this visualization example.
params = model.generate_params()

initial_state = np.array([0.0, 3.0])
timestep = 1e-3
sim_time = 12.0
balance_time = 3.0  # how long to keep watching once the walker has been caught

# Where the ankle controller can bring the walker to a standstill. Measuring this costs
# about a second and a half, and the walking loop below consults it every timestep.
region_of_attraction = build_region_of_attraction(params)

n_timesteps = round(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state
torque_traj = np.zeros(n_timesteps)
completed_steps = 0

# The two-state model does not track translation, so carry the stance foot alongside it:
# each impact moves the foot one step length down the slope.
foot_traj = np.zeros((2, n_timesteps))
step_displacement = (
    2
    * params["length"]
    * np.sin(params["angle_of_attack"])
    * np.array([np.cos(params["incline"]), -np.sin(params["incline"])])
)

# Simulation loop. The controller lives at this level, so the integrator is called one
# timestep at a time rather than handed the whole trajectory.
balance_start_step = None

for step, t in enumerate(time_traj[:-1]):
    state = state_traj[:, step]
    foot = foot_traj[:, step]

    # The event guard. Outside the region of attraction the ankle is left off and the
    # walker is a passive rimless wheel of two spokes; inside it, the ankle takes over
    # and the walker stops taking steps.
    if balance_start_step is None and is_in_region_of_attraction(
        state, region_of_attraction
    ):
        balance_start_step = step

    balancing = balance_start_step is not None
    params["ankle_torque"] = compute_ankle_torque(state, params) if balancing else 0.0
    torque_traj[step] = params["ankle_torque"]

    next_state = rk4_step(t, state, timestep, model, params)

    if not balancing and model.event_guard(state, next_state, params):
        next_state = model.event_dynamics(next_state, params)
        foot = foot + step_displacement
        completed_steps += 1

    state_traj[:, step + 1] = next_state
    foot_traj[:, step + 1] = foot

    if balancing and t - time_traj[balance_start_step] >= balance_time:
        break

time_traj = time_traj[: step + 2]
state_traj = state_traj[:, : step + 2]
foot_traj = foot_traj[:, : step + 2]
torque_traj = torque_traj[: step + 2]

if balance_start_step is None:
    print(f"Never reached the region of attraction in {sim_time} s.")
else:
    caught_state = state_traj[:, balance_start_step]
    print(
        f"Caught after {completed_steps} steps at t = "
        f"{time_traj[balance_start_step]:.3f} s, "
        f"angle {caught_state[0]:+.4f} rad, velocity {caught_state[1]:+.4f} rad/s."
    )
    print(
        f"Final state: angle {state_traj[0, -1]:+.2e} rad, "
        f"velocity {state_traj[1, -1]:+.2e} rad/s."
    )

fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")


def draw_frame(index):
    # The massless swing leg is repositioned instantaneously at each impact, and held
    # clear once the ankle controller has taken over and the walker stops stepping.
    balancing = balance_start_step is not None and index >= balance_start_step

    params["ankle_torque"] = torque_traj[index]
    model.visualize(
        state_traj[:, index],
        params,
        ax=ax,
        show_swing=not balancing,
        stance_position=foot_traj[:, index],
    )
    ax.set_title(f"t = {time_traj[index]:.2f} s" + ("  (balancing)" if balancing else ""))


# Simulate at a small timestep, but render only 25 frames per second.
fps = 25
frame_stride = round(1 / (fps * timestep))
frame_indices = list(range(0, time_traj.size, frame_stride))
if frame_indices[-1] != time_traj.size - 1:
    frame_indices.append(time_traj.size - 1)

animation = FuncAnimation(
    fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
)
output = Path("output/assignment_2")
output.mkdir(parents=True, exist_ok=True)
animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))

# To save an MP4 instead, install FFmpeg and use:
# animation.save(output / "walker.mp4", writer="ffmpeg", fps=fps)
print(f"Saved {output / 'walker.gif'} ({completed_steps} footstrikes).")
plt.show()
