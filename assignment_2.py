from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from models import inverted_pendulum_walker as model

params = model.generate_params()

initial_state = np.array([0.0, 3.0])
initial_foot = np.array([0.0, 0.0])
timestep = 1e-4
sim_time = 3.0
desired_number_of_steps = 3


def advance_stance_foot(foot, params):
    """Move the stance foot one step length down the slope.

    The two-state model tracks only the leg angle, so translation has to be
    accumulated separately for the animation.
    """
    step_length = 2.0 * params["length"] * np.sin(params["angle_of_attack"])
    downhill = np.array([np.cos(params["incline"]), -np.sin(params["incline"])])
    return foot + step_length * downhill


n_timesteps = round(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
foot_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state
foot_traj[:, 0] = initial_foot
completed_steps = 0

# Simulation loop. Replace this Euler step with your own integrator as needed.
for step, t in enumerate(time_traj[:-1]):
    state = state_traj[:, step]
    foot = foot_traj[:, step]
    next_state = state + timestep * model.dynamics(t, state, params)
    next_foot = foot

    if model.event_guard(state, next_state, params):
        next_state = model.event_dynamics(next_state, params)
        next_foot = advance_stance_foot(foot, params)
        completed_steps += 1

    state_traj[:, step + 1] = next_state
    foot_traj[:, step + 1] = next_foot
    if completed_steps == desired_number_of_steps:
        break

time_traj = time_traj[: step + 2]
state_traj = state_traj[:, : step + 2]
foot_traj = foot_traj[:, : step + 2]

print(f"final stance foot: {foot_traj[:, -1]}")

# Fixed camera spanning the whole walk. Without this the view follows the
# stance foot and the walker appears to stay put while the ground slides.
margin = 2.15 * params["length"]
view_limits = (
    foot_traj[0].min() - margin,
    foot_traj[0].max() + margin,
    foot_traj[1].min() - margin,
    foot_traj[1].max() + margin,
)

fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")


def draw_frame(index):
    # The massless swing leg is repositioned instantaneously at each impact.
    model.visualize(
        state_traj[:, index],
        params,
        ax=ax,
        stance_position=foot_traj[:, index],
        view_limits=view_limits,
    )
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