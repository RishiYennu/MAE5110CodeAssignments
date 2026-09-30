from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from assignment_2_balance_controller import build_region_of_attraction
from assignment_2_lookup_table import simulate_walk
from models import inverted_pendulum_walker as model

# Fixed controls for this visualization example.
params = model.generate_params()

initial_state = np.array([0.0, 3.0])
timestep = 1e-3
sim_time = 12.0
balance_time = 3.0  # how long to keep watching once the walker has been caught

# Where the ankle controller can bring the walker to a standstill. Measuring this costs
# about a second and a half, and the walking loop consults it every timestep.
region_of_attraction = build_region_of_attraction(params)


def hold_angle_of_attack(section_velocity):
    # This script is the fixed-control example, so the same angle of attack every step.
    # The lookup table passes a policy here instead.

    return params["angle_of_attack"]


(
    time_traj,
    state_traj,
    foot_traj,
    torque_traj,
    angle_of_attack_traj,
    balance_start_step,
    completed_steps,
) = simulate_walk(
    initial_state,
    params,
    region_of_attraction,
    hold_angle_of_attack,
    timestep=timestep,
    sim_time=sim_time,
    balance_time=balance_time,
)

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

    # params is mutated by the simulation loop above, so replay the values each frame was
    # actually drawn under rather than whichever ones the loop happened to finish on.
    params["ankle_torque"] = torque_traj[index]
    params["angle_of_attack"] = angle_of_attack_traj[index]
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
