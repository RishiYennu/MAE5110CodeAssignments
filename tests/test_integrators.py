import inspect

import numpy as np
import pytest

import integrators


def test_constant_derivative():
    def dynamics(time, state, params):
        return params["velocity"]

    params = {"velocity": np.array([2.0, -3.0])}
    integrator_functions = inspect.getmembers(integrators, inspect.isfunction)
    assert integrator_functions, "No integrator functions exported"
    for name, integrator in integrator_functions:
        state = np.array([1.0, 4.0])
        result = integrator(dynamics, 2.0, state, 0.25, params)

        assert result == pytest.approx([1.5, 3.25]), name
        assert list(state) == [1.0, 4.0], name


def test_exponential_growth():
    def dynamics(time, state, params):
        return params["rate"] * state

    params = {"rate": 1.0}
    timestep = 0.01
    for name, integrator in inspect.getmembers(integrators, inspect.isfunction):
        state = np.array([1.0])
        for step in range(100):
            state = integrator(dynamics, step * timestep, state, timestep, params)

        # At t=1, the exact solution is exp(1). Allow Euler's first-order error.
        assert state == pytest.approx([np.e], rel=0.01, abs=0), name


def test_time_dependent_dynamics():
    def dynamics(time, state, params):
        return np.full_like(state, params["rate"] * time)

    params = {"rate": 2.0}
    timestep = 0.01
    for name, integrator in inspect.getmembers(integrators, inspect.isfunction):
        state = np.array([3.0])
        for step in range(100):
            time = 1.0 + step * timestep
            state = integrator(dynamics, time, state, timestep, params)

        # Integrating 2*t from t=1 to t=2 adds 3; Euler undershoots by 0.01.
        assert state == pytest.approx([6.0], rel=0, abs=0.02), name
