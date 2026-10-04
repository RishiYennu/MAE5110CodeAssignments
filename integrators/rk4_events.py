import numpy as np
from types import SimpleNamespace

# Step cap for event-to-event integration. Measured on the walker, post-impact
# velocity error against the closed-form result, with the dynamics calls it costs:
#
#     cap      calls    error
#     5e-2      18      9.7e-07
#     2e-02     38      3.9e-08      <- default
#     1e-02     66      2.6e-09
#     5e-03    126      1.8e-10
#     1e-03    590      2.9e-13
#
# For reference the old fixed-1e-3 path with bisection cost ~748 calls for ~1e-13.
#
# Two things degrade as the cap grows. Accuracy is the one that bites here, since
# both the RK4 step and the cubic Hermite interpolation lose ground. Separately, a
# step long enough to carry the state across a guard and back would miss the event
# outright; that cannot happen while theta rises monotonically through touchdown, as
# it does for forward walking, but it is a real hazard for a guard the trajectory can
# cross twice within one step.
DEFAULT_MAX_TIMESTEP = 0.02


def rk4_step(t, state, timestep, model, params):

    k_1 = model.dynamics(t, state, params)
    k_2 = model.dynamics(t + timestep / 2, state + timestep / 2 * k_1, params)
    k_3 = model.dynamics(t + timestep / 2, state + timestep / 2 * k_2, params)
    k_4 = model.dynamics(t + timestep, state + timestep * k_3, params)

    return state + (timestep / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)


def interpolate_step(state, next_state, start_derivative, end_derivative, timestep, fraction):
    # Cubic Hermite interpolation of the step already taken
    # tracks the trajectory inside the step without integrating anything

    s = fraction

    interpolated_state = (
        (2 * s**3 - 3 * s**2 + 1) * state
        + (s**3 - 2 * s**2 + s) * timestep * start_derivative
        + (-2 * s**3 + 3 * s**2) * next_state
        + (s**3 - s**2) * timestep * end_derivative
    )

    return interpolated_state


def find_impact_time(t, state, next_state, timestep, model, params, tolerance=1e-14):
    # How far into a step that crossed the guard the impact actually happened.
    #
    # The search runs on an interpolant of the step rather than re-integrating from
    # the start each time, so an iteration costs a polynomial evaluation instead of
    # a whole RK4 step: two dynamics calls in total rather than ~164. Interpolating
    # the state (not a guard value) is what lets the model keep a boolean
    # event_guard, since the guard only ever has to answer "has it crossed yet".

    start_derivative = model.dynamics(t, state, params)
    end_derivative = model.dynamics(t + timestep, next_state, params)

    fraction_before_impact = 0.0  # largest fraction known to fall short of the guard
    fraction_to_impact = 1.0  # smallest fraction known to reach it

    while fraction_to_impact - fraction_before_impact > tolerance:
        trial_fraction = 0.5 * (fraction_before_impact + fraction_to_impact)
        trial_state = interpolate_step(
            state, next_state, start_derivative, end_derivative, timestep, trial_fraction
        )

        if model.event_guard(state, trial_state, params):
            fraction_to_impact = trial_fraction
        else:
            fraction_before_impact = trial_fraction

    step_to_impact = fraction_to_impact * timestep

    return step_to_impact


def require_event_model(model):
    # The integrator needs continuous dynamics, something to say when a step crossed
    # an event, and the jump to apply when it did.

    for required in ("dynamics", "event_guard", "event_dynamics"):
        if not hasattr(model, required):
            raise AttributeError(
                f"model '{model.__name__}' has no {required}(); rk4_with_events "
                "needs dynamics(), event_guard() and event_dynamics()"
            )


def step_through_impacts(t, state, timestep, model, params, max_impacts):
    # Advance exactly `timestep`, resolving any impacts met along the way. Each
    # impact is landed on with a real RK4 step, so only the impact *time* carries
    # interpolation error; the state itself stays RK4-accurate.

    time_into_step = 0.0
    impacts = 0

    while time_into_step < timestep:
        remaining = timestep - time_into_step
        trial_state = rk4_step(t + time_into_step, state, remaining, model, params)

        if not model.event_guard(state, trial_state, params):
            state = trial_state  # no impact in what is left of the step
            break

        sub_step = find_impact_time(
            t + time_into_step, state, trial_state, remaining, model, params
        )
        state = rk4_step(t + time_into_step, state, sub_step, model, params)
        state = model.event_dynamics(state, params)
        time_into_step += sub_step

        impacts += 1
        if impacts >= max_impacts:
            raise RuntimeError(
                f"{max_impacts} impacts within one timestep at t={t:.6g}; "
                "the trajectory is likely Zeno or the guard never clears after reset"
            )

    return state


def rk4_with_events(
    dynamics,
    time,
    state,
    timestep,
    params,
    event_guard=None,
    event_dynamics=None,
    max_impacts_per_step=100,
):
    # Integrate with impacts, sampling the trajectory every `timestep`.
    #
    # `timestep` sets the *sampling* of the returned trajectory, not the accuracy:
    # one RK4 step spans each sample interval, and impacts inside an interval are
    # landed on exactly. Results match to six decimals from 1e-4 through 1e-2, so
    # prefer a coarse grid -- 1e-4 costs 500x more than 1e-2 and buys nothing.
    # Keep the interval at or below DEFAULT_MAX_TIMESTEP for accurate impact timing.

    if event_guard is None and event_dynamics is None:
        return rk4_step(time, state, timestep, SimpleNamespace(dynamics=dynamics), params)

    if event_guard is None or event_dynamics is None:
        raise ValueError("pass both event_guard and event_dynamics, or neither")

    model = SimpleNamespace(
        dynamics=dynamics, event_guard=event_guard, event_dynamics=event_dynamics
    )
    return step_through_impacts(time, state, timestep, model, params, max_impacts_per_step)



def integrate_to_next_impact(
    t,
    state,
    model,
    params,
    max_timestep=DEFAULT_MAX_TIMESTEP,
    max_time=10.0,
):
    # Advance until the next impact and return the state just after it, plus how
    # long that took. No output grid, so the step is limited only by accuracy --
    # this is the step-to-step map, and it is called once per entry when sweeping
    # a state-action table. See DEFAULT_MAX_TIMESTEP for the accuracy/cost trade.
    #
    # Returns (post_impact_state, elapsed_time), or (None, elapsed_time) if no
    # impact arrives within max_time -- which is the answer for a walker that
    # stalls short of touchdown rather than an error.

    require_event_model(model)

    elapsed_time = 0.0

    while elapsed_time < max_time:
        timestep = min(max_timestep, max_time - elapsed_time)
        next_state = rk4_step(t + elapsed_time, state, timestep, model, params)

        if model.event_guard(state, next_state, params):
            sub_step = find_impact_time(
                t + elapsed_time, state, next_state, timestep, model, params
            )
            impact_state = rk4_step(t + elapsed_time, state, sub_step, model, params)
            post_impact_state = model.event_dynamics(impact_state, params)

            return post_impact_state, elapsed_time + sub_step

        state = next_state
        elapsed_time += timestep

    return None, elapsed_time

def simulate_with_events(
    initial_state, timestep, sim_time, model, params, max_impacts_per_step=100
):
    require_event_model(model)

    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((len(initial_state), n_timesteps))
    state_traj[:, 0] = initial_state

    for step, t in enumerate(time_traj[:-1]):
        state_traj[:, step + 1] = rk4_with_events(
            model.dynamics, t, state_traj[:, step], timestep, params,
            model.event_guard, model.event_dynamics, max_impacts_per_step,
        )

    return time_traj, state_traj