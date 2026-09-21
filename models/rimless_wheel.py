import numpy as np


def dynamics(t, state, params):
    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = gravity_over_length(params) * np.sin(angle)

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative


def gravity_over_length(params):
    # the only combination of gravity and length the stance dynamics depend on
    gravity_over_length = params["gravity"] / params["length"]

    return gravity_over_length


def stance_geometry(params):
    # the stance interval endpoints, and the half-angle they are built from
    half_spoke_angle = np.pi / params["n_spokes"]
    trailing_angle = params["slope"] - half_spoke_angle
    leading_angle = params["slope"] + half_spoke_angle

    return half_spoke_angle, trailing_angle, leading_angle


def momentum_scale(params):
    # fraction of angular velocity surviving an impact, from conserving angular
    # momentum about the new contact point as the stance spoke switches
    half_spoke_angle, _, _ = stance_geometry(params)

    momentum_scale = np.cos(2 * half_spoke_angle)

    return momentum_scale


def detect_impact(t, state, params):
    slope = params["slope"]
    half_spoke_angle, _, _ = stance_geometry(params)

    angle = state[0]

    # signed distance out of the stance interval; zero exactly at an impact
    distance_past_spoke = abs(angle - slope) - half_spoke_angle

    return distance_past_spoke


detect_impact.terminal = True
detect_impact.direction = 1  # only on crossings that leave the stance interval


def apply_reset(state, params):
    half_spoke_angle, trailing_angle, leading_angle = stance_geometry(params)
    spoke_spacing = 2 * half_spoke_angle
    impact_scale = momentum_scale(params)

    angle = state[0]
    angular_velocity = state[1]

    if angle >= leading_angle:  # rolling downhill, leading spoke lands
        angle = angle - spoke_spacing
        angular_velocity = angular_velocity * impact_scale
    elif angle <= trailing_angle:  # rocking back, trailing spoke lands
        angle = angle + spoke_spacing
        angular_velocity = angular_velocity * impact_scale

    state = np.array([angle, angular_velocity])
    return state


def speed_to_clear_apex(params):
    # speed needed at the trailing spoke to reach the apex

    _, trailing_angle, _ = stance_geometry(params)

    speed_to_clear_apex = np.sqrt(
        2 * gravity_over_length(params) * (1 - np.cos(trailing_angle))
    )

    return speed_to_clear_apex


def generate_params():
    params = {
        "gravity": 9.81,  # gravity (m/s^2)
        "length": 1.0,  # spoke length, hub to tip (m)
        "mass": 1.0,  # point mass at the hub (kg)
        "n_spokes": 6,  # total spokes on the wheel (unitless)
        "slope": 0.2,  # angle of slop to ground (rad)
    }
    return params
