import numpy as np

from assignments.assignment_1.rimless_wheel_model import RimlessWheel


def generate_params():
    return {
        "spoke_count": 8,
        "slope_angle": np.deg2rad(5.0),
        "spoke_length": 1.0,
    }


def generate_initial_condition():
    return np.array([0.0, 1.0])


def dynamics(t, state, params):
    wheel = RimlessWheel(
        spoke_count=params["spoke_count"],
        slope_angle=params["slope_angle"],
        spoke_length=params["spoke_length"],
    )

    return np.asarray(wheel.compute_derivative(t, state))
