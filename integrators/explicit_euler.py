import numpy as np


def run_integrator(sim_time, timestep, initial_state, params, model):
  n_timesteps = int(sim_time / timestep) + 1
  time_traj = np.arange(n_timesteps) * timestep
  state_traj = np.zeros((len(initial_state), n_timesteps))
  state_traj[:, 0] = initial_state

  # simulation loop
  for step, t in enumerate(time_traj[:-1]):
      if state_traj[0, step] <= 0:
         state_traj[1, step] = -state_traj[1, step]
         state_traj[0, step] = 0
      state_traj[:, step + 1] = state_traj[:, step] + timestep * model.dynamics(
          t, state_traj[:, step], params
    )
      
  return time_traj, state_traj