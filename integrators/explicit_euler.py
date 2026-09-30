import numpy as np

def explicit_euler(initial_state, timestep, sim_time, model, params):

    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    for step, t in enumerate(time_traj[:-1]):
        state_traj[:, step + 1] = state_traj[:, step] + timestep * model.dynamics(
            t, state_traj[:, step], params
        )

    return time_traj, state_traj