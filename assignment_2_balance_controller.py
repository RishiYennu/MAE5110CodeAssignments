"""Standing controller for the inverted pendulum walker, and where it works.

Feedback linearization cancels gravity and leaves a PD law in its place, so the upright
equilibrium becomes stable. The ankle bounds are what limit it: outside a narrow band of
states no torque can catch the walker, and that band is the region of attraction the
step-to-step policy has to aim for.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from integrators.rk4_events import rk4_step
from models import inverted_pendulum_walker as model

# Ankle torque bounds, as fractions of the weight torque m*g*L. The assignment sets
# these; they are asymmetric, so the walker can brake a forward lean twice as hard as
# it can catch a backward one.
MIN_TORQUE_FRACTION = -0.10
MAX_TORQUE_FRACTION = 0.05

# Angle of attack bounds, also set by the assignment. Only the widest one is used here,
# to frame the range of angles the stance leg can ever occupy.
MIN_ANGLE_OF_ATTACK = np.pi / 8
MAX_ANGLE_OF_ATTACK = np.pi / 7

# Closed-loop gains, for the linear system left behind once gravity is cancelled:
# theta_ddot = -BALANCE_STIFFNESS * theta - BALANCE_DAMPING * theta_dot.
#
# These are critically damped, and the choice matters less than it looks. Measured
# against the largest capture band any bounded-torque controller could reach, gains
# from stiffness 6 through 16 all attain it to within one grid cell. Weaker gains fall
# 0.41 rad/s short because they never saturate hard enough to brake; much stiffer ones
# fall 0.24 rad/s short because they saturate on position error and spend the authority
# they needed for braking. The region of attraction belongs to the ankle bounds, not to
# the gains.
BALANCE_STIFFNESS = 9.0
BALANCE_DAMPING = 6.0

# Grid used to measure the region of attraction. 161x161 puts the grid's own resolution
# (0.025 rad/s) at about a twentieth of the band's width and takes ~1.5 s to build; the
# error is exactly one cell at every resolution tried, so this is a straight cost trade.
N_ANGLES = 161
N_VELOCITIES = 161
MAX_BALANCE_VELOCITY = 2.0

# Settling simulation behind each grid cell. 6 s is converged: 4 s still misclassifies
# 13 cells, while timesteps from 2e-2 down to 1e-3 all agree cell for cell.
BALANCE_TIMESTEP = 5e-3
BALANCE_SETTLE_TIME = 6.0
UPRIGHT_TOLERANCE = 1e-3
FALLEN_ANGLE = np.pi / 2

CAPTURE_COLORS = ListedColormap(["#b2182b", "#2166ac"])


def torque_limits(params):
    # the ankle bounds in N m, scaled off the weight torque

    weight_torque = params["mass"] * params["gravity"] * params["length"]

    min_torque = MIN_TORQUE_FRACTION * weight_torque
    max_torque = MAX_TORQUE_FRACTION * weight_torque

    return min_torque, max_torque


def compute_ankle_torque(state, params):
    # Feedback linearization. The first term cancels gravity exactly, which would leave
    # theta_ddot = -stiffness*theta - damping*velocity; the clip is what stops it being
    # that simple, since the ankle cannot always supply the cancelling torque.

    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]
    min_torque, max_torque = torque_limits(params)

    angle = state[0]
    angular_velocity = state[1]

    cancelling_torque = -mass * gravity * length * np.sin(angle)
    restoring_torque = -mass * length**2 * (
        BALANCE_STIFFNESS * angle + BALANCE_DAMPING * angular_velocity
    )

    clipped_torque = np.clip(
        cancelling_torque + restoring_torque, min_torque, max_torque
    )

    return clipped_torque


def sustainable_angles(params):
    # where the walker can be held still at all: the ankle has to balance m*g*L*sin(theta),
    # so the asymmetric torque bounds give an asymmetric range of holdable angles

    weight_torque = params["mass"] * params["gravity"] * params["length"]
    min_torque, max_torque = torque_limits(params)

    min_angle = np.arcsin(-max_torque / weight_torque)
    max_angle = np.arcsin(-min_torque / weight_torque)

    return min_angle, max_angle


def constant_torque_curve(angles, hold_angle, torque, params):
    # Angular velocity along the trajectory that arrives at rest at `hold_angle` under a
    # constant ankle torque, from the invariant
    #     0.5*velocity^2 + (g/L)*cos(angle) - (torque/mL^2)*angle = constant.

    gravity_over_length = params["gravity"] / params["length"]
    angular_acceleration = torque / (params["mass"] * params["length"] ** 2)

    squared_velocity = 2 * (
        gravity_over_length * (np.cos(hold_angle) - np.cos(angles))
        + angular_acceleration * (angles - hold_angle)
    )

    # the branch behind the hold angle is approached moving forward, the one ahead of it
    # moving backward, so the sign flips as the curve passes through it
    angular_velocity = np.sign(hold_angle - angles) * np.sqrt(
        np.maximum(squared_velocity, 0.0)
    )

    return angular_velocity


def capture_band_edges(angles, params):
    # The largest set of states any controller respecting the ankle bounds could bring
    # to a standstill, bounded by the hardest-braking curve into the most forward angle
    # the ankle can hold and the hardest-pushing curve into the most backward one.
    #
    # This is a yardstick, not the guard: it is an upper bound on what the PD above can
    # do, so comparing it against the measured grid says whether the grid is merely
    # resolution-limited or actually wrong.

    min_angle, max_angle = sustainable_angles(params)
    min_torque, max_torque = torque_limits(params)

    lower_edge = constant_torque_curve(angles, min_angle, max_torque, params)
    upper_edge = constant_torque_curve(angles, max_angle, min_torque, params)

    return lower_edge, upper_edge


def classify_balance_states(
    angle_grid,
    velocity_grid,
    params,
    timestep=BALANCE_TIMESTEP,
    settle_time=BALANCE_SETTLE_TIME,
):
    # Run the closed loop from every state at once and see which ones end up upright.
    #
    # The torque is recomputed once per timestep and held across the four RK4 stages,
    # which is not a convenience: it is how the controller actually runs from the
    # experiment script, so the grid measures the controller that will be flown.

    balance_params = dict(params)

    state = np.array([angle_grid, velocity_grid], dtype=float)
    still_standing = np.ones(state.shape[1:], dtype=bool)

    for step in range(round(settle_time / timestep)):
        balance_params["ankle_torque"] = compute_ankle_torque(state, balance_params)
        state = rk4_step(step * timestep, state, timestep, model, balance_params)

        # park fallen states at the origin so they cannot overflow the remaining steps;
        # still_standing is what excludes them from the verdict, not their parked value
        still_standing &= np.abs(state[0]) < FALLEN_ANGLE
        state = np.where(still_standing, state, 0.0)

    reaches_upright = (
        still_standing
        & (np.abs(state[0]) < UPRIGHT_TOLERANCE)
        & (np.abs(state[1]) < UPRIGHT_TOLERANCE)
    )

    return reaches_upright


def build_region_of_attraction(
    params,
    n_angles=N_ANGLES,
    n_velocities=N_VELOCITIES,
    max_velocity=MAX_BALANCE_VELOCITY,
):
    # Measures the region of attraction and hands back the axes alongside it, so the
    # guard can index into the grid and the plot can draw it without classifying twice.
    #
    # The angle range is every angle the stance leg can occupy, for the widest angle of
    # attack allowed. The velocity range is chosen to frame the result: the captured set
    # never reaches 1.75 rad/s anywhere in that angle range.

    angles = np.linspace(
        params["incline"] - MAX_ANGLE_OF_ATTACK,
        params["incline"] + MAX_ANGLE_OF_ATTACK,
        n_angles,
    )
    velocities = np.linspace(-max_velocity, max_velocity, n_velocities)
    angle_grid, velocity_grid = np.meshgrid(angles, velocities)

    reaches_upright = classify_balance_states(angle_grid, velocity_grid, params)

    region_of_attraction = (angles, velocities, reaches_upright)

    return region_of_attraction


def is_in_region_of_attraction(state, region_of_attraction):
    # The event guard: has the walker reached a state the ankle controller can catch?
    #
    # Nearest-cell lookup into the measured grid. Anything off the grid is outside by
    # definition, and rounding to the nearest cell means a state up to half a cell
    # (0.0125 rad/s at the default resolution) beyond the true edge can still read as
    # inside -- the price of reading the answer off a grid rather than a formula.

    angles, velocities, reaches_upright = region_of_attraction

    angle = state[0]
    angular_velocity = state[1]

    angle_index = np.rint((angle - angles[0]) / (angles[1] - angles[0]))
    velocity_index = np.rint(
        (angular_velocity - velocities[0]) / (velocities[1] - velocities[0])
    )

    on_grid = (
        (angle_index >= 0)
        & (angle_index < angles.size)
        & (velocity_index >= 0)
        & (velocity_index < velocities.size)
    )

    in_region = on_grid & reaches_upright[
        np.clip(velocity_index, 0, velocities.size - 1).astype(int),
        np.clip(angle_index, 0, angles.size - 1).astype(int),
    ]

    return in_region


def draw_region_of_attraction(axes, params, region_of_attraction):
    # Paints the measured region into an existing axes, with the analytic capture band
    # over the top so the two can be read against each other.

    angles, velocities, reaches_upright = region_of_attraction

    axes.pcolormesh(
        angles,
        velocities,
        reaches_upright.astype(int),
        cmap=CAPTURE_COLORS,
        norm=BoundaryNorm([-0.5, 0.5, 1.5], CAPTURE_COLORS.N),
        shading="auto",
    )

    lower_edge, upper_edge = capture_band_edges(angles, params)
    axes.plot(angles, lower_edge, "k--", linewidth=1.2)
    axes.plot(angles, upper_edge, "k--", linewidth=1.2)

    return reaches_upright


def plot_region_of_attraction(params, region_of_attraction):
    figure, axes = plt.subplots(figsize=(8, 6))

    draw_region_of_attraction(axes, params, region_of_attraction)

    angles, velocities, _ = region_of_attraction
    axes.set_xlim(angles[0], angles[-1])
    axes.set_ylim(velocities[0], velocities[-1])
    axes.set_xlabel("Angle from vertical (rad)")
    axes.set_ylabel("Angular velocity (rad/s)")
    axes.set_title(
        f"Ankle controller region of attraction "
        f"(stiffness={BALANCE_STIFFNESS:g}, damping={BALANCE_DAMPING:g})"
    )

    axes.legend(
        handles=[
            Patch(facecolor=CAPTURE_COLORS(1), label="Caught, comes to a standstill"),
            Patch(facecolor=CAPTURE_COLORS(0), label="Falls"),
            plt.Line2D([], [], color="k", linestyle="--", label="Largest possible band"),
        ],
        loc="upper right",
        framealpha=0.95,
    )

    figure.tight_layout()

    return figure


if __name__ == "__main__":
    params = model.generate_params()

    min_torque, max_torque = torque_limits(params)
    min_angle, max_angle = sustainable_angles(params)

    print(f"ankle torque bounds: [{min_torque:+.4f}, {max_torque:+.4f}] N m")
    print(
        f"angles the ankle can hold still: [{min_angle:+.4f}, {max_angle:+.4f}] rad "
        f"({np.degrees(min_angle):+.2f} to {np.degrees(max_angle):+.2f} deg)"
    )

    region_of_attraction = build_region_of_attraction(params)
    angles, velocities, reaches_upright = region_of_attraction
    print(
        f"measured region: {reaches_upright.sum()} of {reaches_upright.size} cells, "
        f"resolution {velocities[1] - velocities[0]:.4f} rad/s"
    )

    for angle in (0.0, params["incline"] - MIN_ANGLE_OF_ATTACK):
        lower_edge, upper_edge = capture_band_edges(np.array([angle]), params)
        print(
            f"capture band at angle {angle:+.4f} rad: "
            f"[{lower_edge[0]:+.4f}, {upper_edge[0]:+.4f}] rad/s"
        )

    plot_region_of_attraction(params, region_of_attraction)
    plt.show()
