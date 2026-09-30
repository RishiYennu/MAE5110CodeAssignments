import numpy as np


def dynamics(t, state, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]
    damping_coeff = params["damping_coeff"]
    torque = params["torque"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (
        mass * gravity * length * np.sin(angle)
        - damping_coeff * angular_velocity
        + torque
    ) / (mass * length**2)

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative


def generate_params():
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "length": 1,  # rod length (m)
        "mass": 1,  # point mass at end of rod (kg)
        "damping_coeff": 0.1,  # damping coefficient (kg*m^2/s)
        "torque": 0.0,  # applied torque (N m)
    }
    return params


def generate_initial_condition():
    return np.array([0.0, 0.0])


def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]  # indexes entire row "vectorized" if state is (2, N)
    angular_velocity = state[1]

    kinetic_energy = 0.5 * mass * (length * angular_velocity) ** 2
    potential_energy = mass * gravity * length * np.cos(angle)
    return kinetic_energy, potential_energy
