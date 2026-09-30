"""Compass gait with a hip mass, two leg-midpoint masses, and a hip torque.

State: [stance_angle, swing_angle, stance_velocity, swing_velocity]. Both angles
describe the foot-to-hip direction, clockwise from upward vertical. Positive
hip torque increases swing_angle - stance_angle. The legs have equal length.

The stance foot is pinned. Forward heel-strike is plastic and exchanges the
legs. As in the standard compass-gait model, the swing foot is assumed to retract
while passing the stance foot; scuffing is ignored, without changing the masses'
motion. Flight, slip, and double support are not modeled.

Reference: https://underactuated.mit.edu/simple_legs.html
"""

import matplotlib.pyplot as plt
import numpy as np


def generate_params():
    return {
        "gravity": 9.81,  # m/s^2
        "length": 1.0,  # m, each leg
        "hip_mass": 10.0,  # kg
        "leg_mass": 5.0,  # kg, at each leg's midpoint
        "incline": 0.0525,  # rad, downhill to the right
        "hip_torque": 0.0,  # N m, the only control input
    }


def generate_initial_condition():
    """Start near a passive downhill gait for the default parameters."""
    return np.array([0.0, 0.0, 0.4, -2.0])


def calculate_mass_matrix(state, params):
    stance_angle, swing_angle = state[:2]
    length = params["length"]
    hip_mass = params["hip_mass"]
    leg_mass = params["leg_mass"]
    half_length = length / 2
    coupling = -leg_mass * length * half_length * np.cos(stance_angle - swing_angle)
    return np.array(
        [
            [(hip_mass + leg_mass) * length**2 + leg_mass * half_length**2, coupling],
            [coupling, leg_mass * half_length**2],
        ]
    )


def dynamics(t, state, params):
    """Evaluate M(q) qddot + C(q, qdot) qdot = gravity + [-torque, torque]."""
    stance_angle, swing_angle, stance_velocity, swing_velocity = state
    gravity = params["gravity"]
    length = params["length"]
    hip_mass = params["hip_mass"]
    leg_mass = params["leg_mass"]
    half_length = length / 2

    coupling = leg_mass * length * half_length * np.sin(stance_angle - swing_angle)
    velocity_forces = coupling * np.array([-(swing_velocity**2), stance_velocity**2])
    gravity_forces = gravity * np.array(
        [
            ((hip_mass + leg_mass) * length + leg_mass * half_length)
            * np.sin(stance_angle),
            -leg_mass * half_length * np.sin(swing_angle),
        ]
    )
    hip_torque = params["hip_torque"] * np.array([-1.0, 1.0])
    accelerations = np.linalg.solve(
        calculate_mass_matrix(state, params),
        gravity_forces + hip_torque - velocity_forces,
    )
    return np.array([stance_velocity, swing_velocity, *accelerations])


def event_guard_value(state, params):
    """Signed touchdown guard: the leg angles sum to twice the incline."""
    stance_angle, swing_angle = state[:2]
    return 2 * params["incline"] - stance_angle - swing_angle


def event_guard(previous_state, next_state, params):
    """Detect forward touchdown, excluding coincident feet and backward steps."""
    return (
        event_guard_value(previous_state, params) > 0
        and event_guard_value(next_state, params) <= 0
        and next_state[0] > next_state[1]
    )


def event_dynamics(state, params):
    """Apply a plastic, no-slip impact at touchdown, then exchange the legs.

    The old stance foot lifts without an impulse. Momentum conservation gives
    M_post @ new_velocities = momentum_transfer @ old_velocities. With identical
    legs, exchanging the angles leaves the mass matrix unchanged.
    """
    stance_angle, swing_angle = state[:2]
    length = params["length"]
    hip_mass = params["hip_mass"]
    leg_mass = params["leg_mass"]
    half_length = length / 2
    momentum_transfer = np.array(
        [
            [
                (hip_mass + leg_mass) * length**2 * np.cos(stance_angle - swing_angle),
                -leg_mass * half_length**2,
            ],
            [-leg_mass * half_length**2, 0.0],
        ]
    )
    velocities = np.linalg.solve(
        calculate_mass_matrix(state, params), momentum_transfer @ state[2:]
    )
    return np.array([swing_angle, stance_angle, *velocities])


def calculate_energy(state, params):
    """Return kinetic and potential energy relative to the current stance foot."""
    stance_angle, swing_angle = state[:2]
    velocities = np.asarray(state[2:])
    length = params["length"]
    hip_mass = params["hip_mass"]
    leg_mass = params["leg_mass"]
    kinetic = 0.5 * velocities @ calculate_mass_matrix(state, params) @ velocities
    potential = (
        params["gravity"]
        * length
        * (
            (hip_mass + 1.5 * leg_mass) * np.cos(stance_angle)
            - 0.5 * leg_mass * np.cos(swing_angle)
        )
    )
    return kinetic, potential


def calculate_positions(state, params):
    """Return hip and swing-foot positions relative to the stance foot."""
    stance_angle, swing_angle = state[:2]
    hip = params["length"] * np.array([np.sin(stance_angle), np.cos(stance_angle)])
    swing_foot = hip - params["length"] * np.array(
        [np.sin(swing_angle), np.cos(swing_angle)]
    )
    return hip, swing_foot


def ground_guard_value(state, params):
    """Nonnegative at first hip-ground contact; angles are not wrapped."""
    return abs(state[0] - params["incline"]) - np.pi / 2


def visualize(state, params, ax=None, *, stance_position=(0.0, 0.0), view_limits=None):
    """Draw the nominal straight legs and point masses; return the axes.

    The view follows the stance foot unless fixed view_limits are supplied.
    The idealized swing-foot retraction is not drawn.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.clear()
    stance_foot = np.asarray(stance_position)
    hip, swing_foot = calculate_positions(state, params)
    hip = hip + stance_foot
    swing_foot = swing_foot + stance_foot
    if view_limits is None:
        length = params["length"]
        view_limits = (
            stance_foot[0] - 1.5 * length,
            stance_foot[0] + 1.5 * length,
            stance_foot[1] - 0.5 * length,
            stance_foot[1] + 1.5 * length,
        )
    ground_x = np.array(view_limits[:2])
    ground_y = stance_foot[1] - np.tan(params["incline"]) * (ground_x - stance_foot[0])
    ax.fill_between(ground_x, ground_y, view_limits[2], color="#eee7dc")
    ax.plot(ground_x, ground_y, color="#7b6651", linewidth=2)
    for foot, color, label in [
        (stance_foot, "#23699b", "Stance leg"),
        (swing_foot, "#df8a25", "Swing leg"),
    ]:
        midpoint = (foot + hip) / 2
        ax.plot(
            [foot[0], hip[0]], [foot[1], hip[1]], color=color, linewidth=3, label=label
        )
        ax.plot(*midpoint, "o", color=color, markersize=10)
        ax.plot(*foot, "s", color=color, markersize=6)
    ax.plot(*hip, "o", color="#333333", markersize=15)
    ax.set(
        xlim=view_limits[:2],
        ylim=view_limits[2:],
        xlabel="x (m)",
        ylabel="y (m)",
        title=f"Compass gait: hip torque = {params['hip_torque']:.2f} N m",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    return ax
