import numpy as np


def dynamics(t, state, params):
    gravity = params["gravity"]
    mass = params["mass"]
    damping_coeff = params["damping_coeff"]

    velocity = state[1]

    acceleration = (
        -mass * gravity
        # - damping_coeff * velocity  # <-- DAMPING TERM
    ) / mass

    state_derivative = np.array([velocity, acceleration])
    return state_derivative


def generate_params():
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "mass": 0.2,  # ball mass
        "damping_coeff": 0.1,  # damping coefficient (kg/s)
    }
    return params


def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    gravity = params["gravity"]
    mass = params["mass"]

    height = state[0]  # indexes entire row "vectorized" if state is (2, N)
    velocity = state[1]

    kinetic_energy = 0.5 * mass * velocity ** 2
    potential_energy = mass * gravity * height
    return kinetic_energy, potential_energy
