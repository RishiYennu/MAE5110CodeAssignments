"""Step-to-step dynamics of the walker, sampled on a Poincaré section at mid-stance.

The rimless wheel could use touchdown as its section, because geometry fixed the touchdown
angle. Here the angle of attack is a control input, so that section moves every time the
controller chooses: states from steps taken with different alpha would not be comparable.

Mid-stance is the section that stays put. theta = 0 is fixed regardless of alpha and of the
incline, so every step is sampled on the same surface, and the remaining state is the
single number theta_dot. Two properties of that particular angle do the real work:

- Potential energy there is m*g*L, independent of alpha. Since the passive dynamics
  conserve energy between impacts and the impact only scales theta_dot, the return map
  closes in a form that needs no simulation at all -- see step_map.
- theta_dot is at its minimum there, because the mass is at the top of its arc. So the
  section velocity is the bottleneck speed of the whole step, and theta_dot > 0 at the
  section is precisely the statement that the walker made it over. Transversality and
  "the step succeeded" are the same condition, which is why the section needs no separate
  viability test.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from assignment_2_balance_controller import (
    build_region_of_attraction,
    is_in_region_of_attraction,
    plot_region_of_attraction,
)
from integrators.rk4_events import rk4_step
from models import inverted_pendulum_walker as model

# Mid-stance: the stance leg vertical, the hub directly above the foot.
POINCARE_SECTION_ANGLE = 0.0

# Step outcomes. A step does not always produce a next section velocity, and the two ways
# it can fail to are opposites -- the ankle controller caught the walker (done, and the
# reason for walking at all), or the walker ran out of speed and fell back (infeasible).
CAUGHT = "caught"
RETURNED = "returned"
FELL = "fell"

# Integration of a single step. 1e-4 puts the event-detection error on the section well
# below the grid resolution the lookup table will use.
STEP_TIMESTEP = 1e-4
STEP_TIMEOUT = 20.0

# A walker that has swung this far back is not coming back over the top; the ankle is off
# outside the region of attraction, so nothing is left to arrest it.
FALLEN_BACK_ANGLE = -np.pi / 2


def crossed_section(previous_state, next_state):
    # True on the timestep that carries the stance leg up through mid-stance. Only upward
    # crossings count: theta increases monotonically through the section whenever the
    # walker is walking, so this fires exactly once per step.

    crossed_upward = (
        previous_state[0] < POINCARE_SECTION_ANGLE <= next_state[0]
        and next_state[1] > 0.0
    )

    return crossed_upward


def step_map(section_velocity, alpha, params):
    # The return map in closed form, straight from energy bookkeeping.
    #
    # Potential energy at the section does not depend on alpha, so the two flow arcs can
    # be written against the same datum: climb to touchdown at gamma + alpha, lose a
    # factor cos(2*alpha) in the collision, then climb back from gamma - alpha to the
    # section.
    #
    # NaN where the walker cannot get back over the top, which is the same condition as
    # the squared velocity going negative.

    gravity_over_length = params["gravity"] / params["length"]
    incline = params["incline"]

    climb_to_touchdown = 2 * gravity_over_length * (1 - np.cos(incline + alpha))
    climb_from_impact = 2 * gravity_over_length * (1 - np.cos(incline - alpha))

    squared_velocity = (
        np.cos(2 * alpha) ** 2 * (section_velocity**2 + climb_to_touchdown)
        - climb_from_impact
    )

    next_velocity = np.sqrt(np.where(squared_velocity > 0.0, squared_velocity, np.nan))

    return next_velocity


def minimum_viable_velocity(alpha, params):
    # The slowest the walker can cross the section and still return to it with this alpha,
    # by inverting step_map at zero. Below this the step commits to an impact it cannot
    # climb back from.

    gravity_over_length = params["gravity"] / params["length"]
    incline = params["incline"]

    climb_to_touchdown = 2 * gravity_over_length * (1 - np.cos(incline + alpha))
    climb_from_impact = 2 * gravity_over_length * (1 - np.cos(incline - alpha))

    squared_velocity = climb_from_impact / np.cos(2 * alpha) ** 2 - climb_to_touchdown
    minimum_velocity = np.sqrt(np.maximum(squared_velocity, 0.0))

    return minimum_velocity


def froude_two_velocity(params):
    # Top of the state sweep the assignment prescribes: Froude number 2, where
    # Fr = L*theta_dot^2/g.

    section_velocity = np.sqrt(2 * params["gravity"] / params["length"])

    return section_velocity


def simulate_step(section_velocity, alpha, params, region_of_attraction):
    # One step of the dynamics, from the section to whichever of the three outcomes
    # arrives first. This is the measurement the closed form above is checked against, and
    # the one the lookup table is built from.
    #
    # The ankle stays off for the whole step: the balance controller only takes over once
    # the walker is inside the region of attraction, and at that point this step is the
    # last one. 

    step_params = dict(params)
    step_params["angle_of_attack"] = alpha
    step_params["ankle_torque"] = 0.0

    state = np.array([POINCARE_SECTION_ANGLE, section_velocity])
    next_velocity = np.nan

    for step in range(round(STEP_TIMEOUT / STEP_TIMESTEP)):
        if is_in_region_of_attraction(state, region_of_attraction):
            return CAUGHT, next_velocity

        next_state = rk4_step(
            step * STEP_TIMESTEP, state, STEP_TIMESTEP, model, step_params
        )
        if model.event_guard(state, next_state, step_params):
            next_state = model.event_dynamics(next_state, step_params)
        elif crossed_section(state, next_state):
            return RETURNED, next_state[1]

        if next_state[0] < FALLEN_BACK_ANGLE:
            return FELL, next_velocity

        state = next_state

    return FELL, next_velocity


def section_capture_interval(params, region_of_attraction):
    # Where the section meets the region of attraction: the velocities at mid-stance the
    # ankle controller can bring to a standstill. The step controller's target, and a
    # plain interval because the section pins theta.

    angles, velocities, reaches_upright = region_of_attraction

    column = int(np.argmin(np.abs(angles - POINCARE_SECTION_ANGLE)))
    captured = np.flatnonzero(reaches_upright[:, column])

    if captured.size == 0:
        capture_interval = (np.nan, np.nan)
    else:
        capture_interval = (velocities[captured.min()], velocities[captured.max()])

    return capture_interval


def draw_poincare_section(axes, params, region_of_attraction):
    # The section drawn onto the state-space plot: a vertical line at mid-stance, with the
    # stretch of it the ankle controller can catch picked out.

    lower_capture, upper_capture = section_capture_interval(params, region_of_attraction)

    axes.axvline(
        POINCARE_SECTION_ANGLE,
        color="#111111",
        linewidth=2.0,
        zorder=5,
        label=f"Poincaré section ($\\theta$ = {POINCARE_SECTION_ANGLE:g})",
    )
    axes.plot(
        [POINCARE_SECTION_ANGLE, POINCARE_SECTION_ANGLE],
        [lower_capture, upper_capture],
        color="#ffd400",
        linewidth=5,
        solid_capstyle="butt",
        zorder=6,
        label=(
            f"Caught on the section: "
            f"[{lower_capture:+.3f}, {upper_capture:+.3f}] rad/s"
        ),
    )

    capture_interval = (lower_capture, upper_capture)

    return capture_interval


if __name__ == "__main__":
    params = model.generate_params()
    region_of_attraction = build_region_of_attraction(params)

    lower_capture, upper_capture = section_capture_interval(
        params, region_of_attraction
    )
    print(
        f"section theta = {POINCARE_SECTION_ANGLE:g} meets the region of attraction in "
        f"[{lower_capture:+.4f}, {upper_capture:+.4f}] rad/s"
    )

    froude_velocity = np.sqrt(2 * params["gravity"] / params["length"])
    print(f"Froude number 2 sweep reaches {froude_velocity:.4f} rad/s\n")

    for alpha in (np.pi / 8, np.pi / 7):
        print(
            f"alpha = {alpha:.4f}: contraction {np.cos(2 * alpha) ** 2:.4f} per step, "
            f"needs theta_dot > {minimum_viable_velocity(alpha, params):.4f} rad/s"
        )

    print()
    for section_velocity in (0.3, 0.8, 1.5, 2.5, 4.0):
        for alpha in (np.pi / 8, np.pi / 7):
            outcome, next_velocity = simulate_step(
                section_velocity, alpha, params, region_of_attraction
            )
            predicted = step_map(section_velocity, alpha, params)
            print(
                f"theta_dot = {section_velocity:.2f}, alpha = {alpha:.4f}: {outcome:<8} "
                f"next {next_velocity:+.4f} (closed form {predicted:+.4f})"
            )

    # The section drawn onto the state-space plot it shares with the region of attraction.
    figure = plot_region_of_attraction(params, region_of_attraction)
    axes = figure.axes[0]

    # plot_region_of_attraction builds its legend from explicit handles, since a
    # pcolormesh supplies none of its own. Keep those and add the section's to them,
    # rather than replacing the legend and losing the region's key.
    region_handles = axes.get_legend().legend_handles
    draw_poincare_section(axes, params, region_of_attraction)
    section_handles = axes.get_legend_handles_labels()[0]
    axes.legend(handles=region_handles + section_handles, loc="upper right",
                framealpha=0.95, fontsize=9)

    output = Path("output/assignment_2")
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "poincare_section.png", dpi=150)
    print(f"\nSaved {output / 'poincare_section.png'}")
    plt.show()
