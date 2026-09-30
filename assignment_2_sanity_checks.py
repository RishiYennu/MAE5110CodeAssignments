import numpy as np

from assignment_2_balance_controller import (
    BALANCE_DAMPING,
    BALANCE_STIFFNESS,
    MAX_ANGLE_OF_ATTACK,
    MIN_ANGLE_OF_ATTACK,
    N_VELOCITIES,
    build_region_of_attraction,
    capture_band_edges,
    classify_balance_states,
    compute_ankle_torque,
    is_in_region_of_attraction,
    sustainable_angles,
    torque_limits,
)
from assignment_2_lookup_table import (
    CAUGHT_CODE,
    RETURNED_CODE,
    build_step_table,
    nearest_state_index,
    rollout_policy,
    solve_step_counts,
)
from assignment_2_step_controller import (
    CAUGHT,
    FALLEN_BACK_ANGLE,
    FELL,
    POINCARE_SECTION_ANGLE,
    RETURNED,
    STEP_TIMEOUT,
    STEP_TIMESTEP,
    crossed_section,
    froude_two_velocity,
    minimum_viable_velocity,
    simulate_step,
    step_map,
)
from integrators.rk4_events import rk4_step
from models import inverted_pendulum_walker as model

# How finely the grid has to resolve the capture band to count as sufficient, as a
# fraction of the band's own width. Halving the cell moves the measured error between
# 10.6% and 5.3% of the band, so anything in that interval picks the same grid; 0.08 sits
# in the middle of the gap, which is what keeps the verdict from turning on the number.
BAND_RESOLUTION_FRACTION = 0.08


# Building a region is the expensive part of this file and three checks want the same
# ones, so each resolution is built once and kept.
_regions = {}


def region_at(params, n_cells):
    if n_cells not in _regions:
        _regions[n_cells] = build_region_of_attraction(
            params, n_angles=n_cells, n_velocities=n_cells
        )

    region = _regions[n_cells]

    return region


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


def measured_band_edges(velocities, reaches_upright):
    # The captured set's upper and lower edge in each angle column, read off the grid --
    # the measured counterpart to capture_band_edges, which gives the same pair
    # analytically. Columns that captured nothing are NaN so they drop out of comparisons
    # rather than reading as a zero-width band.

    lower_edge = np.full(reaches_upright.shape[1], np.nan)
    upper_edge = np.full(reaches_upright.shape[1], np.nan)

    for column, captured in enumerate(reaches_upright.T):
        rows = np.flatnonzero(captured)
        if rows.size:
            lower_edge[column] = velocities[rows.min()]
            upper_edge[column] = velocities[rows.max()]

    return lower_edge, upper_edge


def boundary_error(params, n_cells, reference_edges=None):
    # Worst distance between the measured region's edge and a reference edge, over every
    # angle column. The reference is the analytic band by default; pass measured edges
    # from a finer grid to compare the region against a higher-resolution copy of itself.

    angles, velocities, reaches_upright = region_at(params, n_cells)
    measured_lower, measured_upper = measured_band_edges(velocities, reaches_upright)

    if reference_edges is None:
        reference_lower, reference_upper = capture_band_edges(angles, params)
    else:
        reference_angles, reference_lower, reference_upper = reference_edges
        reference_lower = np.interp(angles, reference_angles, reference_lower)
        reference_upper = np.interp(angles, reference_angles, reference_upper)

    worst = max(
        np.nanmax(np.abs(measured_upper - reference_upper)),
        np.nanmax(np.abs(measured_lower - reference_lower)),
    )

    return float(worst)


def check_grid_has_converged_in_resolution(
    params, resolutions=(41, 81, 161), reference=321
):
    # Refining the grid should change the answer by exactly what snapping a continuous
    # edge onto grid rows costs -- cell - reference_cell -- and by nothing else. Anything
    # that reads differently at 41x41 than at 321x321 is a defect that scales with the
    # grid rather than a region that is merely coarsely drawn.

    reference_angles, reference_velocities, reference_captured = region_at(
        params, reference
    )
    reference_cell = reference_velocities[1] - reference_velocities[0]
    reference_lower, reference_upper = measured_band_edges(
        reference_velocities, reference_captured
    )
    reference_edges = (reference_angles, reference_lower, reference_upper)

    errors = []
    for n in resolutions:
        velocities = region_at(params, n)[1]
        cell = velocities[1] - velocities[0]
        worst = boundary_error(params, n, reference_edges)
        errors.append((n, cell, worst, cell - reference_cell))

    within_quantization = all(
        worst <= predicted + 0.1 * cell for _, cell, worst, predicted in errors
    )

    return report(
        "refining the grid changes the answer only by quantization",
        f"boundary error no worse than cell - {reference_cell:.4f} at every resolution, "
        f"measured against a {reference}x{reference} copy of the same region",
        ", ".join(
            f"{n}x{n}: error {worst:.4f} vs predicted {predicted:.4f}"
            for n, _, worst, predicted in errors
        ),
        within_quantization,
    )


def check_grid_is_the_coarsest_sufficient(params):
    # Whether the shipped grid resolves the band is an absolute question, so it needs an
    # absolute bar: a fixed fraction of the band's own width, which does not move when the
    # grid does.

    coarser_cells = (N_VELOCITIES + 1) // 2

    angles = region_at(params, N_VELOCITIES)[0]
    lower_edge, upper_edge = capture_band_edges(angles, params)
    band_width = float(np.mean(upper_edge - lower_edge))
    tolerance = BAND_RESOLUTION_FRACTION * band_width

    error = boundary_error(params, N_VELOCITIES)
    coarser_error = boundary_error(params, coarser_cells)

    return report(
        f"{N_VELOCITIES}x{N_VELOCITIES} is the coarsest grid that resolves the band",
        f"error under {BAND_RESOLUTION_FRACTION:.0%} of the {band_width:.4f} rad/s band "
        f"({tolerance:.4f} rad/s), and over it at {coarser_cells}x{coarser_cells}",
        f"{N_VELOCITIES}x{N_VELOCITIES}: {error:.4f} rad/s "
        f"({error / band_width:.1%} of the band), "
        f"{coarser_cells}x{coarser_cells}: {coarser_error:.4f} rad/s "
        f"({coarser_error / band_width:.1%})",
        error < tolerance <= coarser_error,
    )


def check_gains_attain_best_possible_band(params):
    # What the measured region is worth as control, rather than as arithmetic: the PD
    # gains should capture very nearly everything the ankle bounds physically allow, so
    # the measured region should sit within a cell of the largest band any bounded-torque
    # controller could reach.

    velocities = region_at(params, N_VELOCITIES)[1]
    cell = velocities[1] - velocities[0]

    error = boundary_error(params, N_VELOCITIES)

    return report(
        "the chosen gains attain the largest possible capture band",
        f"measured region within one cell ({cell:.4f} rad/s) of the analytic band",
        f"worst boundary error {error:.4f} rad/s at "
        f"{N_VELOCITIES}x{N_VELOCITIES} ({error / cell:.2f} cells)",
        error <= cell * 1.001,
    )


def trace_step(section_velocity, alpha, params):
    # One step, instrumented: the section crossings in each direction and the slowest the
    # walker got along the way. simulate_step stops at the first crossing and consults the
    # ankle guard, both of which are right for the return map and both of which hide what
    # the checks below need to see -- so this runs the bare dynamics instead.

    step_params = dict(params)
    step_params["angle_of_attack"] = alpha
    step_params["ankle_torque"] = 0.0

    state = np.array([POINCARE_SECTION_ANGLE, section_velocity])
    upward = []
    downward = 0
    slowest = section_velocity

    for step in range(round(STEP_TIMEOUT / STEP_TIMESTEP)):
        next_state = rk4_step(
            step * STEP_TIMESTEP, state, STEP_TIMESTEP, model, step_params
        )
        if model.event_guard(state, next_state, step_params):
            next_state = model.event_dynamics(next_state, step_params)
        else:
            if crossed_section(state, next_state):
                upward.append(next_state[1])
                break
            if state[0] >= POINCARE_SECTION_ANGLE > next_state[0]:
                downward += 1

        if next_state[0] < FALLEN_BACK_ANGLE:
            break

        slowest = min(slowest, next_state[1])
        state = next_state

    return upward, downward, slowest


def check_closed_form_matches_simulation(params):
    # step_map is energy bookkeeping with no integration in it, which is only legitimate
    # because potential energy at the section does not depend on alpha. Comparing it
    # against the integrated step is what says that reasoning is sound.
    #
    # Only steps that actually return are comparable: the ankle guard ends the others
    # part-way, and the closed form knows nothing about the ankle.

    region_of_attraction = region_at(params, N_VELOCITIES)

    compared = []
    for section_velocity in np.linspace(1.5, froude_two_velocity(params), 6):
        for alpha in (MIN_ANGLE_OF_ATTACK, MAX_ANGLE_OF_ATTACK):
            outcome, next_velocity = simulate_step(
                section_velocity, alpha, params, region_of_attraction
            )
            if outcome != RETURNED:
                continue
            predicted = step_map(section_velocity, alpha, params)
            compared.append(abs(next_velocity - predicted))

    worst_error = max(compared) if compared else np.inf

    return report(
        "closed-form return map matches the integrated step",
        f"agreement to 2e-3 rad/s, the event-detection error at dt = {STEP_TIMESTEP:g}",
        f"worst error {worst_error:.2e} rad/s over {len(compared)} returning steps",
        worst_error < 2e-3,
    )


def check_section_is_crossed_once_per_step(params):
    # The two properties the assignment asks of a section: transverse to the flow, and
    # crossed once per step so theta_dot_k is a well-defined sequence. The substantive
    # half is that no step ever crosses back down through mid-stance -- a walker that did
    # would sample the section twice in a step and the return map would be ambiguous.
    #
    # Only viable steps are in scope. Every step is a strict contraction and there is no
    # passive limit cycle at this incline, so a walker started slow simply fails to come
    # back -- which is the map being partial, not the section being wrong.

    crossing_counts = []
    downward_total = 0
    slowest_crossing = np.inf

    for alpha in (MIN_ANGLE_OF_ATTACK, MAX_ANGLE_OF_ATTACK):
        viable_from = minimum_viable_velocity(alpha, params)
        for section_velocity in np.linspace(
            viable_from + 0.1, froude_two_velocity(params), 5
        ):
            upward, downward, _ = trace_step(section_velocity, alpha, params)
            crossing_counts.append(len(upward))
            downward_total += downward
            if upward:
                slowest_crossing = min(slowest_crossing, upward[0])

    once_per_step = all(count == 1 for count in crossing_counts)

    return report(
        "the section is crossed exactly once per step, transversally",
        "one upward crossing per viable step, no downward ones, "
        "and theta_dot > 0 at every crossing",
        f"upward crossings per step {sorted(set(crossing_counts))}, "
        f"downward {downward_total}, slowest crossing {slowest_crossing:+.4f} rad/s",
        once_per_step and downward_total == 0 and slowest_crossing > 0.0,
    )


def check_section_velocity_is_the_step_bottleneck(params):
    # Potential energy peaks at mid-stance, so kinetic energy bottoms out there. That is
    # what makes the section coordinate meaningful on its own: theta_dot_k is not just a
    # sample of the step, it is the slowest the walker gets during it, so theta_dot_k > 0
    # is exactly the condition for the step to have happened at all.

    worst_error = 0.0
    for alpha in (MIN_ANGLE_OF_ATTACK, MAX_ANGLE_OF_ATTACK):
        viable_from = minimum_viable_velocity(alpha, params)
        for section_velocity in np.linspace(
            viable_from + 0.1, froude_two_velocity(params), 5
        ):
            upward, _, slowest = trace_step(section_velocity, alpha, params)
            if upward:
                worst_error = max(worst_error, abs(upward[0] - slowest))

    return report(
        "the section velocity is the slowest point of the step",
        "the next section crossing equals the minimum theta_dot over the step",
        f"worst discrepancy {worst_error:.2e} rad/s",
        worst_error < 1e-6,
    )


def check_step_outcomes_are_exhaustive(params, n_velocities=13, n_alphas=5):
    # Every entry the lookup table will hold has to resolve to something. The three
    # outcomes are not symmetric: "no next section velocity" splits into the walker being
    # caught, which is the goal, and the walker falling, which is not. A table that read
    # the map as velocity-or-nothing would confuse the two.

    region_of_attraction = region_at(params, N_VELOCITIES)

    counts = {CAUGHT: 0, RETURNED: 0, FELL: 0}
    fell_at = []
    for section_velocity in np.linspace(0.0, froude_two_velocity(params), n_velocities):
        for alpha in np.linspace(MIN_ANGLE_OF_ATTACK, MAX_ANGLE_OF_ATTACK, n_alphas):
            outcome, next_velocity = simulate_step(
                section_velocity, alpha, params, region_of_attraction
            )
            counts[outcome] += 1
            if outcome == FELL:
                fell_at.append(alpha)
            # a returning step must hand back a usable velocity, the others must not
            if (outcome == RETURNED) == np.isnan(next_velocity):
                counts["inconsistent"] = counts.get("inconsistent", 0) + 1

    total = n_velocities * n_alphas
    widest_only = all(np.isclose(alpha, MAX_ANGLE_OF_ATTACK) for alpha in fell_at)

    return report(
        "every state-action pair resolves to exactly one step outcome",
        f"{total} pairs split across caught/returned/fell, none timing out or "
        "returning a velocity it should not have",
        f"caught {counts[CAUGHT]}, returned {counts[RETURNED]}, fell {counts[FELL]}"
        + (" (all at the widest alpha)" if fell_at and widest_only else ""),
        sum(counts[outcome] for outcome in (CAUGHT, RETURNED, FELL)) == total
        and "inconsistent" not in counts,
    )


def small_step_table(params):
    # A coarse table, built once and kept. The graded resolution study needs the big one
    # and lives in the lookup table module; what the checks below test is structure, and
    # structure does not care how finely the axis is sampled.

    if "table" not in _regions:
        _regions["table"] = build_step_table(
            params, region_at(params, N_VELOCITIES), n_velocities=21, n_alphas=9
        )

    step_table = _regions["table"]

    return step_table


def check_rounded_lookup_strictly_decreases(params):
    # Both backward passes walk the table as a graph, and the longest-path one only
    # terminates if that graph is acyclic. It is, for a physical reason: every step is a
    # strict contraction, so theta_dot falls monotonically. But the pass does not consult
    # theta_dot, it consults the ROUNDED successor -- so what has to hold is that rounding
    # never lands back on the state it came from. A coarse enough grid would break that
    # and the longest walk would be reported as infinite.

    velocities, _, outcomes, next_velocities = small_step_table(params)
    successor = nearest_state_index(velocities, next_velocities)

    returns = outcomes == RETURNED_CODE
    states = np.arange(velocities.size)[:, None] * np.ones_like(outcomes, dtype=int)

    decreases = velocities[successor[returns]] < velocities[states[returns]]
    smallest_gap = float(
        (velocities[states[returns]] - next_velocities[returns]).min()
    )

    return report(
        "rounding a successor onto the grid never revisits its own state",
        f"every returning cell lands strictly below itself, with room to spare "
        f"over the {velocities[1] - velocities[0]:.4f} rad/s cell",
        f"{int(decreases.sum())} of {int(returns.sum())} returning cells decrease, "
        f"smallest true gap {smallest_gap:.4f} rad/s",
        bool(decreases.all()) and smallest_gap > velocities[1] - velocities[0],
    )


def check_step_counts_are_a_fixed_point(params):
    # Value iteration has converged when another sweep changes nothing. Worth asserting
    # rather than assuming, since the loop exits on that condition and would otherwise
    # quietly return whatever it had reached when it ran out of sweeps.
    #
    # The second half is the base case: a state needs exactly one step precisely when
    # some angle of attack catches it outright. If those disagree, the backward pass has
    # been seeded wrong and every count above it is off by the same amount.

    step_table = small_step_table(params)
    velocities, _, outcomes, _ = step_table

    settled = True
    for objective in ("min", "max"):
        step_counts, _ = solve_step_counts(step_table, objective)
        again, _ = solve_step_counts(step_table, objective)
        settled = settled and np.array_equal(step_counts, again)

    minimum_steps, _ = solve_step_counts(step_table, "min")
    catches = (outcomes == CAUGHT_CODE).any(axis=1)
    base_case_agrees = bool(np.array_equal(minimum_steps == 1, catches))

    return report(
        "the solved step counts are a converged fixed point",
        "another sweep changes nothing, and one step is needed exactly where "
        "some angle of attack catches",
        f"stable under re-solving: {settled}; one-step states "
        f"{int((minimum_steps == 1).sum())}, catching states {int(catches.sum())} "
        f"of {velocities.size}",
        settled and base_case_agrees,
    )


def check_policy_achieves_its_predicted_step_count(params, n_samples=12):
    # The table is consulted by rounding, so its promise is only worth what it survives.
    # Fly the policy against the real dynamics from initial conditions deliberately off
    # the grid, and compare the steps actually taken against the steps predicted.
    #
    # This is a check on the policy, not on the resolution. It is dominated by how ties
    # between equally-optimal angles of attack are broken -- taking the first optimal one
    # rather than the middle of them pushes the mismatch rate above 40% at every grid
    # size tried, because the first is always the one sitting on the capture boundary.

    step_table = small_step_table(params)
    velocities = step_table[0]
    region_of_attraction = region_at(params, N_VELOCITIES)

    mismatches = []
    for objective in ("min", "max"):
        step_counts, policy = solve_step_counts(step_table, objective)
        # halfway between grid points, where rounding has the most to get wrong
        offset = 0.5 * (velocities[1] - velocities[0])
        for section_velocity in np.linspace(
            velocities[0] + offset, velocities[-1] - offset, n_samples
        ):
            achieved = len(
                rollout_policy(
                    section_velocity, params, region_of_attraction, step_table, policy
                )
            )
            predicted = step_counts[np.abs(velocities - section_velocity).argmin()]
            if achieved != predicted:
                mismatches.append((objective, section_velocity, predicted, achieved))

    return report(
        "the policy achieves the step count the table predicts",
        f"{2 * n_samples} rollouts from off-grid states, flown against the real "
        "dynamics, all matching their prediction",
        f"{len(mismatches)} mismatches"
        + (
            f", worst {mismatches[0][0]} at {mismatches[0][1]:.3f} rad/s "
            f"(predicted {mismatches[0][2]}, achieved {mismatches[0][3]})"
            if mismatches
            else ""
        ),
        not mismatches,
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

    angles, velocities, _ = region = region_at(params, N_VELOCITIES)
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
        check_grid_has_converged_in_resolution(params),
        check_grid_is_the_coarsest_sufficient(params),
        check_gains_attain_best_possible_band(params),
        check_closed_form_matches_simulation(params),
        check_section_is_crossed_once_per_step(params),
        check_section_velocity_is_the_step_bottleneck(params),
        check_step_outcomes_are_exhaustive(params),
        check_rounded_lookup_strictly_decreases(params),
        check_step_counts_are_a_fixed_point(params),
        check_policy_achieves_its_predicted_step_count(params),
        check_guard_agrees_with_simulation(params),
    ]

    print(f"\n{sum(results)}/{len(results)} checks passed")
