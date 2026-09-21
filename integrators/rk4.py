import numpy as np

def rk4(initial_state, timestep, sim_time, model, params):

    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    has_reset = hasattr(model, "apply_reset")


    for step, t in enumerate(time_traj[:-1]):
        current_state = state_traj[:, step]

        k_1 = model.dynamics(t, current_state, params)
        k_2 = model.dynamics(t + timestep / 2, current_state + timestep / 2 * k_1, params)
        k_3 = model.dynamics(t + timestep / 2, current_state + timestep / 2 * k_2, params)
        k_4 = model.dynamics(t + timestep, current_state + timestep * k_3, params)

        next_state = state_traj[:, step + 1] = state_traj[:, step] + (timestep/6) * (k_1 + 2*k_2 + 2*k_3 + k_4)

        if has_reset:
            post_reset_state = model.apply_reset(next_state, params)
            next_state = post_reset_state

        state_traj[:, step+ 1] = next_state
    return time_traj, state_traj