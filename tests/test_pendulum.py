import numpy as np

from integrators import rk4
from models import pendulum


def simulate(initial_state, params, timestep=0.01, steps=100):
    states = [np.asarray(initial_state, dtype=float)]
    for step in range(steps):
        states.append(
            rk4(
                pendulum.dynamics,
                step * timestep,
                states[-1],
                timestep,
                params,
            )
        )
    return np.asarray(states)


def test_energy_is_conserved_without_damping_or_torque():
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 0.0
    states = simulate([0.5, 0.0], params)

    kinetic, potential = pendulum.calculate_energy(states.T, params)
    total_energy = kinetic + potential

    assert np.isclose(total_energy[-1] - total_energy[0], 0.0, atol=1e-7)


def test_constant_torque_produces_expected_angular_velocity():
    params = pendulum.generate_params()
    params["gravity"] = 0.0
    params["damping_coeff"] = 0.0
    params["torque"] = 0.5
    timestep = 0.01
    steps = 100
    states = simulate([0.0, 0.0], params, timestep=timestep, steps=steps)

    moment_of_inertia = params["mass"] * params["length"] ** 2
    expected_velocity = params["torque"] / moment_of_inertia * timestep * steps

    assert np.isclose(states[-1, 1], expected_velocity)


def test_damping_removes_energy():
    params = pendulum.generate_params()
    params["gravity"] = 0.0
    params["damping_coeff"] = 0.5
    params["torque"] = 0.0
    states = simulate([0.0, 1.0], params)

    kinetic, potential = pendulum.calculate_energy(states.T, params)
    total_energy = kinetic + potential

    assert total_energy[-1] < total_energy[0]
