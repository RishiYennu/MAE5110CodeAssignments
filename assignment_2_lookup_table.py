"""Step-to-step control as a lookup table over the state-action space.

One axis is the state, theta_dot at the Poincare section; the other is the control, the
angle of attack chosen for the coming step. Each cell holds what that step does: the
walker is caught by the ankle controller, it returns to the section at some new velocity,
or it falls. Working backwards from the caught cells gives, for every starting velocity,
the fewest steps to a standstill and the angle of attack sequence that achieves it.

The gap between a velocity and its successor is far larger than a grid cell, so rounding
to the nearest grid state cannot round back onto the state you came from. That is
asserted in the sanity checks, because a violation of it would turn the longest
path search below into an infinite loop.
"""

from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from assignment_2_balance_controller import (
    CAPTURE_COLORS,
    MAX_ANGLE_OF_ATTACK,
    MIN_ANGLE_OF_ATTACK,
    build_region_of_attraction,
    compute_ankle_torque,
    draw_region_of_attraction,
    is_in_region_of_attraction,
)
from assignment_2_step_controller import (
    CAUGHT,
    FELL,
    POINCARE_SECTION_ANGLE,
    RETURNED,
    crossed_section,
    draw_poincare_section,
    froude_two_velocity,
    minimum_viable_velocity,
    simulate_step,
)
from integrators.rk4_events import rk4_step
from models import inverted_pendulum_walker as model

# The state grid. 161 is the coarsest that locates the step-count boundaries of both
# backward passes to within STEP_BAND_FRACTION of their narrowest band.
N_SECTION_VELOCITIES = 161
N_ANGLES_OF_ATTACK = 33

# Only for the resolution study
REFERENCE_VELOCITIES = 321

# How precisely the step-count boundaries have to be located, as a fraction of the
# narrowest step-count band.
STEP_BAND_FRACTION = 0.02

# Outcome codes. simulate_step returns strings, which are annoying for an array
OUTCOME_CODES = {CAUGHT: 0, RETURNED: 1, FELL: 2}
CAUGHT_CODE, RETURNED_CODE, FELL_CODE = 0, 1, 2

# Stands in for "cannot reach a standstill".
UNREACHABLE = 10**6

CACHE_DIRECTORY = Path("output/assignment_2/tables")

OUTCOME_COLORS = ("#2166ac", "#d9b268", "#b2182b")
OUTCOME_LABELS = (
    "Caught by the ankle controller",
    "Returns to the section",
    "Falls backward",
)


def build_step_table(
    params,
    region_of_attraction,
    n_velocities=N_SECTION_VELOCITIES,
    n_alphas=N_ANGLES_OF_ATTACK,
):
    # The state-action space itself: one simulated step per cell.

    velocities = np.linspace(0.0, froude_two_velocity(params), n_velocities)
    alphas = np.linspace(MIN_ANGLE_OF_ATTACK, MAX_ANGLE_OF_ATTACK, n_alphas)

    outcomes = np.empty((n_velocities, n_alphas), dtype=np.int8)
    next_velocities = np.full((n_velocities, n_alphas), np.nan)

    for row, section_velocity in enumerate(velocities):
        for column, alpha in enumerate(alphas):
            outcome, next_velocity = simulate_step(
                section_velocity, alpha, params, region_of_attraction
            )
            outcomes[row, column] = OUTCOME_CODES[outcome]
            next_velocities[row, column] = next_velocity

    step_table = (velocities, alphas, outcomes, next_velocities)

    return step_table


def load_step_table(
    params,
    region_of_attraction,
    n_velocities=REFERENCE_VELOCITIES,
    n_alphas=N_ANGLES_OF_ATTACK,
):
    # Build once, reuse thereafter. The cache lives under output/, which is gitignored

    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIRECTORY / f"step_table_{n_velocities}x{n_alphas}.npz"

    if cache_path.exists():
        cached = np.load(cache_path)
        step_table = (
            cached["velocities"],
            cached["alphas"],
            cached["outcomes"],
            cached["next_velocities"],
        )
        return step_table

    step_table = build_step_table(
        params, region_of_attraction, n_velocities, n_alphas
    )
    velocities, alphas, outcomes, next_velocities = step_table
    np.savez(
        cache_path,
        velocities=velocities,
        alphas=alphas,
        outcomes=outcomes,
        next_velocities=next_velocities,
    )

    return step_table


def subsample_step_table(step_table, stride):
    # A coarser state grid taken from a finer one.

    velocities, alphas, outcomes, next_velocities = step_table

    coarse_table = (
        velocities[::stride],
        alphas,
        outcomes[::stride],
        next_velocities[::stride],
    )

    return coarse_table


def nearest_state_index(velocities, next_velocities):
    # Where each successor velocity lands on the state grid.

    rounded = np.abs(next_velocities[..., None] - velocities).argmin(axis=-1)

    return rounded


def solve_step_counts(step_table, objective="min"):
    # Backward pass over the state-action space. A cell that catches ends the walk in one
    # step; a cell that returns costs one step plus whatever the successor state costs;
    # a cell that falls is never usable, so even the longest walk has to end caught.

    velocities, alphas, outcomes, next_velocities = step_table
    successor = nearest_state_index(velocities, next_velocities)

    better = min if objective == "min" else max
    step_counts = np.full(velocities.size, UNREACHABLE)
    policy = np.full(velocities.size, -1)

    for _ in range(velocities.size):
        updated = step_counts.copy()
        updated_policy = policy.copy()

        for state in range(velocities.size):
            costs = []
            for action in range(alphas.size):
                if outcomes[state, action] == CAUGHT_CODE:
                    costs.append(1)
                elif outcomes[state, action] == RETURNED_CODE:
                    costs.append(1 + step_counts[successor[state, action]])
                else:
                    costs.append(UNREACHABLE)

            best = better([cost for cost in costs if cost < UNREACHABLE], default=UNREACHABLE)
            tied = [action for action, cost in enumerate(costs) if cost == best]

            updated[state] = best
            updated_policy[state] = tied[len(tied) // 2] if tied else -1

        if np.array_equal(updated, step_counts):
            break
        step_counts, policy = updated, updated_policy

    solution = (step_counts, policy)

    return solution


def rollout_policy(section_velocity, params, region_of_attraction, step_table, policy):
    # The policy flown against the real dynamics: the table is consulted by rounding to
    # the nearest tabulated state, but every step is simulated from the true velocity.

    velocities, alphas, _, _ = step_table

    walk = []
    velocity = section_velocity

    for _ in range(velocities.size):
        action = policy[np.abs(velocities - velocity).argmin()]
        if action < 0:
            break

        alpha = alphas[action]
        outcome, next_velocity = simulate_step(
            velocity, alpha, params, region_of_attraction
        )
        walk.append((velocity, alpha, outcome))

        if outcome != RETURNED:
            break
        velocity = next_velocity

    return walk


def step_count_boundaries(velocities, step_counts):
    # Where the step count changes, placed midway between the two grid states that
    # straddle it. Comparing these against a finer table is how the state grid's
    # resolution is judged.

    boundaries = [
        0.5 * (velocities[state] + velocities[state + 1])
        for state in range(step_counts.size - 1)
        if step_counts[state] != step_counts[state + 1]
    ]

    return boundaries


def simulate_walk(
    initial_state, params, region_of_attraction, choose_alpha, timestep=1e-3,
    sim_time=20.0, balance_time=3.0,
):
    # The continuous trajectory behind a walk: the ankle controller off and the chosen
    # angle of attack held for each step, until the walker enters the region of attraction
    # and the balance controller takes over.

    walk_params = dict(params)
    walk_params["angle_of_attack"] = choose_alpha(initial_state[1])
    walk_params["ankle_torque"] = 0.0

    n_timesteps = round(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state
    torque_traj = np.zeros(n_timesteps)
    angle_of_attack_traj = np.zeros(n_timesteps)

    # The two-state model does not track translation, so carry the stance foot alongside
    # it: each impact moves the foot one step length down the slope.
    foot_traj = np.zeros((2, n_timesteps))

    balance_start_step = None
    completed_steps = 0

    for step, t in enumerate(time_traj[:-1]):
        state = state_traj[:, step]
        foot = foot_traj[:, step]

        if balance_start_step is None and is_in_region_of_attraction(
            state, region_of_attraction
        ):
            balance_start_step = step

        balancing = balance_start_step is not None
        walk_params["ankle_torque"] = (
            compute_ankle_torque(state, walk_params) if balancing else 0.0
        )
        torque_traj[step] = walk_params["ankle_torque"]
        angle_of_attack_traj[step] = walk_params["angle_of_attack"]

        next_state = rk4_step(t, state, timestep, model, walk_params)

        if not balancing:
            if model.event_guard(state, next_state, walk_params):
                # the angle of attack that governed this step is still in walk_params,
                # which is what the displacement has to be measured with
                next_state = model.event_dynamics(next_state, walk_params)
                foot = foot + model.find_step_displacement(walk_params)
                completed_steps += 1
            elif crossed_section(state, next_state):
                # a step is chosen once per stance, at the section that defines the map
                walk_params["angle_of_attack"] = choose_alpha(next_state[1])

        state_traj[:, step + 1] = next_state
        foot_traj[:, step + 1] = foot

        if balancing and t - time_traj[balance_start_step] >= balance_time:
            break

    walk = (
        time_traj[: step + 2],
        state_traj[:, : step + 2],
        foot_traj[:, : step + 2],
        torque_traj[: step + 2],
        angle_of_attack_traj[: step + 2],
        balance_start_step,
        completed_steps,
    )

    return walk


def plot_step_table(step_table, params):
    # The curve is the analytic boundary below which no step can return to 
    # the section, which the simulated outcomes should hug.

    velocities, alphas, outcomes, _ = step_table

    colors = ListedColormap(OUTCOME_COLORS)
    figure, axes = plt.subplots(figsize=(9, 6), layout="constrained")
    axes.pcolormesh(
        alphas,
        velocities,
        outcomes,
        cmap=colors,
        norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], colors.N),
        shading="auto",
    )
    axes.plot(
        alphas,
        [minimum_viable_velocity(alpha, params) for alpha in alphas],
        "k--",
        linewidth=1.5,
        label="Slowest step that returns",
    )
    axes.set(
        xlabel=r"Angle of attack $\alpha$ (rad)",
        ylabel=r"$\dot\theta_k$ at the section (rad/s)",
        title="Step-to-step state-action space",
        xlim=(alphas[0], alphas[-1]),
        ylim=(velocities[0], velocities[-1]),
    )

    axes.set_axisbelow(False)
    axes.xaxis.set_major_locator(MultipleLocator(0.01))
    axes.xaxis.set_minor_locator(MultipleLocator(0.002))
    axes.yaxis.set_major_locator(MultipleLocator(0.5))
    axes.yaxis.set_minor_locator(MultipleLocator(0.25))
    axes.grid(which="major", color="white", alpha=0.7, linewidth=1.0)
    axes.grid(which="minor", color="white", alpha=0.3, linewidth=0.6)
    axes.tick_params(which="minor", length=3)

    axes.legend(
        handles=[
            *(
                Patch(facecolor=color, label=label)
                for color, label in zip(OUTCOME_COLORS, OUTCOME_LABELS)
            ),
            plt.Line2D([], [], color="k", linestyle="--", label="Slowest returning step"),
        ],
        loc="lower left",
        framealpha=0.95,
    )

    return figure


def plot_steps_to_standstill(step_table, minimum, maximum):
    # How many steps a given starting velocity needs, and how many it can be made to last.
    # The gap between the two is the room the policy has to play with.

    velocities, alphas, _, _ = step_table
    min_steps, min_policy = minimum
    max_steps, _ = maximum

    figure, (axes, policy_axes) = plt.subplots(
        2, 1, figsize=(9, 7), height_ratios=(2, 1), sharex=True, layout="constrained"
    )

    axes.step(velocities, min_steps, where="mid", color="#23699b", linewidth=2,
              label="Fewest steps to standstill")
    axes.step(velocities, max_steps, where="mid", color="#df8a25", linewidth=2,
              label="Most steps before the walker is caught")
    axes.set(
        ylabel="Steps to standstill",
        title="Steps to a standstill from a given section velocity",
        ylim=(0, max_steps.max() + 1),
    )
    axes.grid(alpha=0.3)
    axes.legend(loc="upper left", framealpha=0.95)

    policy_axes.step(velocities, alphas[min_policy], where="mid", color="#23699b",
                     linewidth=2)
    policy_axes.axhline(alphas[0], color="0.6", linestyle=":", linewidth=1)
    policy_axes.axhline(alphas[-1], color="0.6", linestyle=":", linewidth=1)
    policy_axes.set(
        xlabel=r"$\dot\theta_0$ at the section (rad/s)",
        ylabel=r"chosen $\alpha$ (rad)",
        xlim=(velocities[0], velocities[-1]),
    )
    policy_axes.grid(alpha=0.3)

    return figure


def plot_policy_trajectories(params, region_of_attraction, walks):

    figure, (axes, time_axes) = plt.subplots(
        1, 2, figsize=(13, 5.5), width_ratios=(1.15, 1), layout="constrained"
    )

    axes.set_facecolor(CAPTURE_COLORS(0))
    draw_region_of_attraction(axes, params, region_of_attraction)
    draw_poincare_section(axes, params, region_of_attraction)

    # the mesh supplies no legend handles of its own, so name its two colours here
    region_handles = [
        Patch(facecolor=CAPTURE_COLORS(1), label="Caught, comes to a standstill"),
        Patch(facecolor=CAPTURE_COLORS(0), label="Falls"),
    ]

    angles, velocities, _ = region_of_attraction
    reach = 0.0
    for (label, color), walk in walks.items():
        time_traj, state_traj = walk[0], walk[1]
        reach = max(reach, np.abs(state_traj[1]).max())
        axes.plot(state_traj[0], state_traj[1], color=color, linewidth=1.6,
                  label=label, zorder=7, alpha=0.9)
        axes.plot(*state_traj[:, 0], "o", color=color, markersize=9,
                  markeredgecolor="white", zorder=9)

        time_axes.plot(time_traj, state_traj[1], color=color, linewidth=1.8, label=label)
        # One marker per section crossing, so the steps can be counted off the figure.
        # Only while walking: once the ankle controller has the walker, it settles through
        # mid-stance on its way to upright, and those passes are not steps.
        walking = walk[5] if walk[5] is not None else state_traj.shape[1]
        crossings = np.flatnonzero(
            (state_traj[0, :-1] < POINCARE_SECTION_ANGLE)
            & (state_traj[0, 1:] >= POINCARE_SECTION_ANGLE)
            & (state_traj[1, 1:] > 0)
        )
        crossings = crossings[crossings < walking]
        time_axes.plot(time_traj[crossings + 1], state_traj[1, crossings + 1], "o",
                       color=color, markersize=7, markeredgecolor="white", zorder=5)

    axes.set(
        xlabel="Angle from vertical (rad)",
        ylabel="Angular velocity (rad/s)",
        title="State space",
        xlim=(angles[0], angles[-1]),
        # down to the bottom of the measured region, up to the fastest the walker goes
        ylim=(1.05 * velocities.min(), 1.05 * reach),
    )
    trajectory_handles = [
        handle for handle in axes.get_legend_handles_labels()[0]
    ]
    axes.legend(handles=region_handles + trajectory_handles, loc="lower left",
                framealpha=0.95, fontsize=8)

    time_axes.set(
        xlabel="Time (s)",
        ylabel=r"$\dot\theta$ (rad/s)",
        title="Against time, with each section crossing marked",
    )
    time_axes.grid(alpha=0.3)
    time_axes.legend(loc="upper right", framealpha=0.95)

    return figure


def resolution_study(params, region_of_attraction, objective="min", strides=(8, 4, 2)):
    # How coarse can the state grid be before the lookup stops working? Run once

    reference = load_step_table(
        params, region_of_attraction, REFERENCE_VELOCITIES, N_ANGLES_OF_ATTACK
    )
    reference_velocities = reference[0]

    # A boundary is placed midway between the two grid states straddling it, on the coarse
    # grid and the reference alike, so the gap between two placements is quantified by the
    # reference cell and saturates at one and a half of them. Grids closer together than
    # that report the same number and cannot be told apart.
    measurement_floor = 1.5 * (reference_velocities[1] - reference_velocities[0])

    reference_steps, _ = solve_step_counts(reference, objective)
    reference_boundaries = step_count_boundaries(reference_velocities, reference_steps)

    # The narrowest run of equal step count is the finest feature the table has to
    # resolve, so it sets the scale the bar is a fraction of.
    bands = [reference_velocities[0], *reference_boundaries, reference_velocities[-1]]
    tolerance = STEP_BAND_FRACTION * min(
        upper - lower for lower, upper in pairwise(bands)
    )

    study = []
    # strides index into REFERENCE_VELOCITIES: 8, 4, 2 give 41, 81 and 161 states
    for stride in strides:
        # exact, since a step's outcome does not depend on the grid it was tabulated on
        coarse = subsample_step_table(reference, stride)
        velocities = coarse[0]
        step_counts, _ = solve_step_counts(coarse, objective)
        boundaries = step_count_boundaries(velocities, step_counts)

        # Boundary displacement is the measure that scales cleanly with the cell. The
        # fraction of the axis misclassified does not, because it turns on where the
        # boundaries happen to fall between grid points.
        if len(boundaries) == len(reference_boundaries):
            shift = max(
                abs(coarse_edge - fine_edge)
                for coarse_edge, fine_edge in zip(boundaries, reference_boundaries)
            )
        else:
            shift = np.inf

        study.append((velocities.size, velocities[1] - velocities[0], shift))

    resolution = (study, reference_boundaries, tolerance, measurement_floor)

    return resolution


if __name__ == "__main__":
    params = model.generate_params()
    region_of_attraction = build_region_of_attraction(params)

    # Both maps get judged. The shipped grid has to satisfy the stricter of them, which
    # is not the one the assignment's wording points at: the fewest-steps map is coarse
    # and forgiving, and it is the longest-walk map that sets the resolution.
    sufficient = {}
    for objective in ("min", "max"):
        study, reference_boundaries, tolerance, floor = resolution_study(
            params, region_of_attraction, objective
        )
        decidable = tolerance > floor
        print(
            f"{objective}-steps map: boundaries at "
            + ", ".join(f"{edge:.4f}" for edge in reference_boundaries)
            + f" rad/s, measured against {REFERENCE_VELOCITIES} states"
        )
        print(
            f"  bar: {STEP_BAND_FRACTION:.0%} of the narrowest band "
            f"= {tolerance:.4f} rad/s"
            + (
                ""
                if decidable
                else f"  -- BELOW the {floor:.4f} rad/s the reference can resolve, "
                "so this map cannot settle the question"
            )
        )
        for n_velocities, cell, shift in study:
            passes = shift < tolerance
            if decidable:
                sufficient.setdefault(n_velocities, []).append(passes)
            print(
                f"    {n_velocities:3d} states (cell {cell:.4f}): "
                f"shift {shift:.4f} rad/s -- "
                + ("passes" if passes else "FAILS")
                + ("" if decidable else " (not decisive)")
            )
        print()

    reference_table = load_step_table(
        params, region_of_attraction, REFERENCE_VELOCITIES, N_ANGLES_OF_ATTACK
    )

    start_velocity = 4.0
    print(f"longest walk from {start_velocity:.2f} rad/s, promised against delivered:")
    for stride in (8, 4, 2):
        coarse = subsample_step_table(reference_table, stride)
        coarse_velocities = coarse[0]
        longest, longest_policy = solve_step_counts(coarse, "max")
        promised = longest[np.abs(coarse_velocities - start_velocity).argmin()]
        delivered = len(
            rollout_policy(
                start_velocity, params, region_of_attraction, coarse, longest_policy
            )
        )
        print(
            f"  {coarse_velocities.size:3d} states: promised {promised}, "
            f"delivered {delivered}"
            + ("" if promised == delivered else "  <- the table over-promises")
        )
    print()

    step_table = subsample_step_table(
        reference_table,
        (REFERENCE_VELOCITIES - 1) // (N_SECTION_VELOCITIES - 1),
    )
    velocities, alphas, outcomes, _ = step_table
    minimum = solve_step_counts(step_table, "min")
    maximum = solve_step_counts(step_table, "max")

    print(
        f"\nshipped table {velocities.size}x{alphas.size}: "
        f"caught {(outcomes == CAUGHT_CODE).sum()}, "
        f"returns {(outcomes == RETURNED_CODE).sum()}, "
        f"falls {(outcomes == FELL_CODE).sum()}"
    )
    print(
        f"fewest steps: {minimum[0].min()} to {minimum[0].max()};  "
        f"most steps: {maximum[0].max()};  "
        f"states that cannot reach a standstill: {(minimum[0] >= UNREACHABLE).sum()}"
    )

    # An initial condition in the three-step band, flown both ways.
    start_velocity = 4.0
    walks = {}
    for label, (steps, policy), color in (
        ("Fewest steps", minimum, "#23699b"),
        ("Most steps", maximum, "#df8a25"),
    ):
        rollout = rollout_policy(
            start_velocity, params, region_of_attraction, step_table, policy
        )
        predicted = steps[np.abs(velocities - start_velocity).argmin()]
        print(
            f"\n{label} from {start_velocity:.2f} rad/s: predicted {predicted}, "
            f"achieved {len(rollout)}"
        )
        for velocity, alpha, outcome in rollout:
            print(f"    theta_dot {velocity:6.3f}  alpha {alpha:.4f}  -> {outcome}")

        def choose_alpha(velocity, policy=policy):
            return alphas[policy[np.abs(velocities - velocity).argmin()]]

        walks[(f"{label} ({len(rollout)})", color)] = simulate_walk(
            np.array([0.0, start_velocity]), params, region_of_attraction, choose_alpha
        )

    output = Path("output/assignment_2")
    output.mkdir(parents=True, exist_ok=True)
    plot_step_table(step_table, params).savefig(output / "step_table.png", dpi=150)
    plot_steps_to_standstill(step_table, minimum, maximum).savefig(
        output / "steps_to_standstill.png", dpi=150
    )
    plot_policy_trajectories(params, region_of_attraction, walks).savefig(
        output / "policy_trajectories.png", dpi=150
    )
    print(f"\nSaved three figures to {output}")
    plt.show()
