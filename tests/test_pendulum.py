import numpy as np

from models import pendulum as model
from integrators import rk4


def total_energy(state, params):
    kinetic, potential = model.calculate_energy(state, params)
    return kinetic + potential


def simulate(params):
    state = np.array([1.0, 0.0])  # not at equilibrium
    for step in range(100):
        state = rk4(model.dynamics, step * 0.01, state, 0.01, params)
    return state


def test_energy_conserved():
    params = model.generate_params()
    params["torque"] = 0.0
    params["damping_coeff"] = 0.0
    start = np.array([1.0, 0.0])
    end = simulate(params)
    assert np.isclose(total_energy(end, params) - total_energy(start, params), 0, atol=1e-6)


def test_damping_removes_energy():
    params = model.generate_params()
    params["torque"] = 0.0
    params["damping_coeff"] = 0.5
    start = np.array([1.0, 0.0])
    end = simulate(params)
    assert total_energy(end, params) < total_energy(start, params)


def test_torque_work_equals_energy_change():
    params = model.generate_params()
    params["torque"] = 0.5
    params["damping_coeff"] = 0.0
    start = np.array([1.0, 0.0])
    end = simulate(params)
    energy_change = total_energy(end, params) - total_energy(start, params)
    work = params["torque"] * (end[0] - start[0])  # W = torque * change in angle
    assert np.isclose(energy_change, work, atol=1e-6)