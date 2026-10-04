def rk4(dynamics, time, state, timestep, params):
    k1 = dynamics(time, state, params)
    k2 = dynamics(time + timestep / 2, state + k1 * timestep / 2, params)
    k3 = dynamics(time + timestep / 2, state + k2 * timestep / 2, params)
    k4 = dynamics(time + timestep, state + k3 * timestep, params)
    average_slope = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return state + timestep * average_slope
