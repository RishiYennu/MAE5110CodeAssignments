from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from models import inverted_pendulum_walker as model

# Fixed controls for this visualization example.
params = {
    "gravity": 9.81,  # m/s^2
    "length": 1.0,  # m
    "mass": 1.0,  # kg
    "incline": 0.06,  # rad
    "angle_of_attack": np.pi / 8,  # rad
    "ankle_torque": 0.0,  # N m
}

initial_state = np.array([0.0, 3.0])
timestep = 1e-4
sim_time = 3.0
desired_number_of_steps = 3

n_timesteps = round(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state
completed_steps = 0

# Simulation loop. Replace this Euler step with your own integrator as needed.
for step, t in enumerate(time_traj[:-1]):
    state = state_traj[:, step]
    next_state = state + timestep * model.dynamics(t, state, params)

    if model.event_guard(state, next_state, params):
        next_state = model.event_dynamics(next_state, params)
        completed_steps += 1

    state_traj[:, step + 1] = next_state
    if completed_steps == desired_number_of_steps:
        break

time_traj = time_traj[: step + 2]
state_traj = state_traj[:, : step + 2]

fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")


def draw_frame(index):
    # The massless swing leg is repositioned instantaneously at each impact.
    model.visualize(state_traj[:, index], params, ax=ax)
    ax.set_title(f"t = {time_traj[index]:.2f} s")


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