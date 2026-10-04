def explicit_euler(dynamics, time, state, timestep, params):
    return state + timestep * dynamics(time, state, params)
