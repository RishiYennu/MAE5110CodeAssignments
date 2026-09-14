import numpy as np


def run_integrator(sim_time, timestep, initial_state, params, model):
  n_timesteps = int(sim_time / timestep) + 1
  time_traj = np.arange(n_timesteps) * timestep
  state_traj = np.zeros((2, n_timesteps))
  state_traj[:, 0] = initial_state

  # simulation loop
  for step, t in enumerate(time_traj[:-1]):
      k1 = model.dynamics(t, state_traj[:, step], params) 
      k2 = model.dynamics(t + (timestep/2.), state_traj[:, step] + k1 * (timestep/2.), params) 
      k3 = model.dynamics(t + (timestep/2.), state_traj[:, step] + k2 * (timestep/2.), params) 
      k4 = model.dynamics(t + timestep, state_traj[:, step] + k3 * timestep, params) 
      average_slope = 1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
      state_traj[:, step + 1] = state_traj[:, step] + timestep * average_slope
    
  return time_traj, state_traj