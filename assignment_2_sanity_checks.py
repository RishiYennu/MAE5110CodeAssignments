import numpy as np

from assignment_2_balance_controller import (
    BALANCE_DAMPING,
    BALANCE_STIFFNESS,
    build_region_of_attraction,
    capture_band_edges,
    classify_balance_states,
    compute_ankle_torque,
    is_in_region_of_attraction,
    sustainable_angles,
    torque_limits,
)
from models import inverted_pendulum_walker as model


def report(name, expected, measured, passed):
    # One line per check, expectation first so the comparison reads in order
    print(f"{'PASS' if passed else 'FAIL'}  {name}")
    print(f"        expected: {expected}")
    print(f"        measured: {measured}")

    return passed


def check_feedback_linearization_is_exact(params, n_samples=2000):
    # The point of cancelling gravity is that what is left behind is linear. Away from
    # the torque bounds the closed loop must be exactly theta_ddot = -k_p*theta -
    # k_d*velocity, with no trace of the sin(theta) it was built from.

    rng = np.random.default_rng(0)
    min_torque, max_torque = torque_limits(params)

    # small enough states that the commanded torque never reaches a bound
    angles = rng.uniform(-0.01, 0.01, n_samples)
    velocities = rng.uniform(-0.01, 0.01, n_samples)
    state = np.array([angles, velocities])

    unsaturated_params = dict(params)
    unsaturated_params["ankle_torque"] = compute_ankle_torque(state, params)
    unclipped = np.all(
        (unsaturated_params["ankle_torque"] > min_torque)
        & (unsaturated_params["ankle_torque"] < max_torque)
    )

    acceleration = model.dynamics(0.0, state, unsaturated_params)[1]
    linear = -BALANCE_STIFFNESS * angles - BALANCE_DAMPING * velocities
    worst_error = float(np.abs(acceleration - linear).max())

    return report(
        "feedback linearization leaves an exactly linear closed loop",
        f"theta_ddot = -{BALANCE_STIFFNESS:g}*theta - {BALANCE_DAMPING:g}*velocity "
        "to machine precision",
        f"worst error {worst_error:.2e} over {n_samples} states, none clipped: {unclipped}",
        unclipped and worst_error < 1e-12,
    )


def check_torque_respects_bounds(params, n_samples=200):
    # The assignment fixes the ankle bounds, so no state may command past them -- and
    # standing perfectly upright must ask for nothing at all.

    min_torque, max_torque = torque_limits(params)

    angles = np.linspace(-np.pi / 2, np.pi / 2, n_samples)
    velocities = np.linspace(-10.0, 10.0, n_samples)
    state = np.array(np.meshgrid(angles, velocities))

    torque = compute_ankle_torque(state, params)
    upright_torque = float(compute_ankle_torque(np.array([0.0, 0.0]), params))

    within_bounds = bool(np.all((torque >= min_torque) & (torque <= max_torque)))

    return report(
        "ankle torque stays within the permitted bounds",
        f"every command inside [{min_torque:+.4f}, {max_torque:+.4f}] N m, "
        "and exactly zero when upright and still",
        f"range [{torque.min():+.4f}, {torque.max():+.4f}] N m over "
        f"{torque.size} states, upright command {upright_torque:+.2e} N m",
        within_bounds and abs(upright_torque) < 1e-15,
    )


def check_sustainable_angles(params):
    # A standstill needs the ankle to hold m*g*L*sin(theta) on its own, so the reachable
    # standstill angles follow straight from the bounds. This is the asymmetry that makes
    # the whole region of attraction lean.

    weight_torque = params["mass"] * params["gravity"] * params["length"]
    min_torque, max_torque = torque_limits(params)
    min_angle, max_angle = sustainable_angles(params)

    # the torque needed to hold each end, which should sit exactly on a bound
    held_at_min = -weight_torque * np.sin(min_angle)
    held_at_max = -weight_torque * np.sin(max_angle)

    # just outside, the commanded torque can no longer cancel gravity
    outside = max_angle + 1e-3
    residual = weight_torque * np.sin(outside) + float(
        compute_ankle_torque(np.array([outside, 0.0]), params)
    )

    return report(
        "standstill angles match the torque bounds",
        f"[{np.arcsin(-max_torque / weight_torque):+.6f}, "
        f"{np.arcsin(-min_torque / weight_torque):+.6f}] rad, "
        "with gravity uncancelled just outside",
        f"[{min_angle:+.6f}, {max_angle:+.6f}] rad, holding torques "
        f"{held_at_min:+.4f}/{held_at_max:+.4f} N m, residual {residual:+.2e} N m",
        abs(held_at_min - max_torque) < 1e-12
        and abs(held_at_max - min_torque) < 1e-12
        and residual > 0,
    )


def check_grid_matches_largest_possible_band(params, resolutions=(41, 81, 161)):
    # The measured region is compared against the largest band any bounded-torque
    # controller could capture. Agreeing to within one cell at every resolution is what
    # says the grid is merely resolution-limited rather than biased -- and it is the
    # argument for picking the coarsest grid that is still good enough.

    errors = []

    for n in resolutions:
        angles, velocities, reaches_upright = build_region_of_attraction(
            params, n_angles=n, n_velocities=n
        )
        lower_edge, upper_edge = capture_band_edges(angles, params)
        cell = velocities[1] - velocities[0]

        worst = 0.0
        for column, captured in enumerate(reaches_upright.T):
            rows = np.flatnonzero(captured)
            if rows.size == 0:
                continue
            worst = max(
                worst,
                abs(velocities[rows.max()] - upper_edge[column]),
                abs(velocities[rows.min()] - lower_edge[column]),
            )
        errors.append((n, cell, worst))

    within_one_cell = all(worst <= cell * 1.001 for _, cell, worst in errors)

    return report(
        "measured region matches the largest possible capture band",
        "boundary error no worse than one grid cell at every resolution",
        ", ".join(
            f"{n}x{n}: cell {cell:.4f}, error {worst:.4f}" for n, cell, worst in errors
        ),
        within_one_cell,
    )


def check_guard_agrees_with_simulation(params, n_samples=400):
    # The guard reads a grid, so a state just outside the true region can round into a
    # captured cell. This measures that exposure rather than assuming it away: every
    # state the guard accepts is simulated, and any that falls is a false "you're done".
    #
    # What matters is not how many disagree -- the boundary is sampled deliberately, so
    # some will -- but how far outside the band the guard can be talked into reaching.
    # Rounding caps that, on both axes: half a cell of velocity directly, plus half a
    # cell of angle multiplied by the slope of the band's edge, since a state rounded
    # sideways is compared against the edge somewhere else. Anything past that bound is
    # an indexing error rather than resolution.

    angles, velocities, _ = region = build_region_of_attraction(params)
    cell = velocities[1] - velocities[0]
    rng = np.random.default_rng(1)

    # sample right along the boundary, where disagreement can actually happen
    sample_angles = rng.uniform(angles[0], angles[-1], n_samples)
    lower_edge, upper_edge = capture_band_edges(sample_angles, params)
    offsets = rng.uniform(-2 * cell, 2 * cell, n_samples)
    sample_velocities = np.where(
        rng.random(n_samples) < 0.5, upper_edge + offsets, lower_edge - offsets
    )

    state = np.array([sample_angles, sample_velocities])
    accepted = is_in_region_of_attraction(state, region)
    converges = classify_balance_states(sample_angles, sample_velocities, params)

    # how far beyond the nearer edge each state sits; negative means inside the band
    outside_by = np.maximum(
        lower_edge - sample_velocities, sample_velocities - upper_edge
    )

    overreaching = accepted & ~converges
    worst_overreach = float(outside_by[overreaching].max()) if overreaching.any() else 0.0

    angle_cell = angles[1] - angles[0]
    grid_lower, grid_upper = capture_band_edges(angles, params)
    band_slope = max(
        np.abs(np.gradient(grid_upper, angle_cell)).max(),
        np.abs(np.gradient(grid_lower, angle_cell)).max(),
    )
    rounding_bound = cell / 2 + band_slope * angle_cell / 2

    return report(
        "guard reaches no further past the band than grid rounding allows",
        f"accepted states no more than {rounding_bound:.4f} rad/s outside "
        f"(half a cell of velocity {cell / 2:.4f}, plus the sloped edge)",
        f"{int(overreaching.sum())} of {n_samples} boundary states accepted but fell, "
        f"worst {worst_overreach:.4f} rad/s outside the band",
        worst_overreach <= rounding_bound,
    )


if __name__ == "__main__":
    params = model.generate_params()

    print(
        f"inverted pendulum walker sanity checks "
        f"(incline={params['incline']} rad, stiffness={BALANCE_STIFFNESS:g}, "
        f"damping={BALANCE_DAMPING:g})\n"
    )

    results = [
        check_feedback_linearization_is_exact(params),
        check_torque_respects_bounds(params),
        check_sustainable_angles(params),
        check_grid_matches_largest_possible_band(params),
        check_guard_agrees_with_simulation(params),
    ]

    print(f"\n{sum(results)}/{len(results)} checks passed")
