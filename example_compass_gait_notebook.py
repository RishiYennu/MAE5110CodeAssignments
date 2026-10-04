import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Compass-gait walker

    From the repository root, run `uv run scripts/example_compass_gait.py`.
    Edit that script to make shared changes, then regenerate this notebook using
    the command in assignments/assignment_3.md. Start with zero hip torque for passive
    downhill walking.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter

    from integrators import rk4 as integrator
    from models import compass_gait as model

    return FuncAnimation, Path, PillowWriter, integrator, model, np, plt


@app.cell
def _(model):
    params = model.generate_params()
    params["hip_torque"] = 0.0  # N m; passive downhill walking
    initial_state = model.generate_initial_condition()
    timestep = 0.002  # s
    sim_time = 6.0  # s
    SAVE_GIF = False  # Exporting every animation frame takes several seconds.
    return SAVE_GIF, initial_state, params, sim_time, timestep


@app.cell
def _(initial_state, integrator, model, np, params, sim_time, timestep):
    # Set up the trajectories, including the stance foot's world position.
    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((initial_state.size, n_timesteps))
    state_traj[:, 0] = initial_state
    foot_traj = np.zeros((2, n_timesteps))
    completed_steps = 0

    # Simulation loop
    for step, t in enumerate(time_traj[:-1]):
        state = state_traj[:, step]
        # A controller can update params["hip_torque"] here.

        next_state = integrator(model.dynamics, t, state, timestep, params)
        foot_traj[:, step + 1] = foot_traj[:, step]

        if model.event_guard(state, next_state, params):
            # Bisect this timestep to find when the swing foot touches down.
            lower, upper = 0.0, timestep
            for _ in range(20):
                middle = (lower + upper) / 2
                contact_state = integrator(model.dynamics, t, state, middle, params)
                if model.event_guard_value(contact_state, params) > 0:
                    lower = middle
                else:
                    upper = middle
            contact_time = upper
            contact_state = integrator(model.dynamics, t, state, contact_time, params)

            # Accept only a forward foothold at the actual contact time.
            if contact_state[0] > contact_state[1]:
                _, swing_foot = model.calculate_positions(contact_state, params)
                foot_traj[:, step + 1] += swing_foot
                next_state = model.event_dynamics(contact_state, params)
                completed_steps += 1

                # Finish the timestep using the new stance leg.
                next_state = integrator(
                    model.dynamics,
                    t + contact_time,
                    next_state,
                    timestep - contact_time,
                    params,
                )

        state_traj[:, step + 1] = next_state
        if model.ground_guard_value(next_state, params) >= 0:
            print("Simulation stopped: the walker fell over.")
            break

    time_traj = time_traj[: step + 2]
    state_traj = state_traj[:, : step + 2]
    foot_traj = foot_traj[:, : step + 2]
    return completed_steps, foot_traj, state_traj, time_traj


@app.cell
def _(Path, plt, state_traj, time_traj):
    output = Path("output/compass_gait")
    output.mkdir(parents=True, exist_ok=True)
    state_fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for angle, velocity, label in [(0, 2, "Stance leg"), (1, 3, "Swing leg")]:
        axes[0].plot(time_traj, state_traj[angle], label=label)
        axes[1].plot(state_traj[angle], state_traj[velocity], label=label)
    axes[0].set(xlabel="Time (s)", ylabel="Angle (rad)", title="Leg angles")
    axes[1].set(
        xlabel="Angle (rad)", ylabel="Angular velocity (rad/s)", title="Phase portrait"
    )
    for plot_axis in axes:
        plot_axis.legend()
        plot_axis.grid(alpha=0.25)
    state_fig.savefig(output / "states.png", dpi=180)
    state_fig  # noqa: B018 — display the figure in the notebook
    return (output,)


@app.cell
def _(
    FuncAnimation,
    PillowWriter,
    SAVE_GIF,
    completed_steps,
    foot_traj,
    model,
    np,
    output,
    params,
    plt,
    state_traj,
    time_traj,
    timestep,
):
    length = params["length"]
    view_limits = (
        np.min(foot_traj[0]) - 1.2 * length,
        np.max(foot_traj[0]) + 1.2 * length,
        np.min(foot_traj[1]) - 0.3 * length,
        np.max(foot_traj[1]) + 1.3 * length,
    )
    animation_fig, animation_axis = plt.subplots(figsize=(10, 4))


    def draw_frame(index):
        model.visualize(
            state_traj[:, index],
            params,
            ax=animation_axis,
            stance_position=foot_traj[:, index],
            view_limits=view_limits,
        )
        animation_axis.set_title(f"Compass gait: t = {time_traj[index]:.2f} s")


    fps = 25
    frame_stride = max(1, round(1 / (fps * timestep)))
    frame_indices = list(range(0, time_traj.size, frame_stride))
    if frame_indices[-1] != time_traj.size - 1:
        frame_indices.append(time_traj.size - 1)
    animation = FuncAnimation(
        animation_fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
    )
    if SAVE_GIF:
        animation.save(output / "compass_gait.gif", writer=PillowWriter(fps=fps))
        print(f"Saved animation to {output / 'compass_gait.gif'}.")
    print(f"Saved plots to {output} ({completed_steps} heelstrikes).")
    draw_frame(0)
    plt.show()
    # Display playback controls when this cell is run in a notebook.
    plt.rcParams["animation.html"] = "jshtml"
    animation  # noqa: B018 — display the animation in the notebook
    return


if __name__ == "__main__":
    app.run()
