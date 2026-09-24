from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from models import inverted_pendulum_walker as model

OUTPUT = Path("output/assignment_2")


def compute_ankle_torque(state, params, kp, kd) -> float:
    length = params["length"]
    gravity = params["gravity"]
    mass = params["mass"]

    theta = state[0]
    angular_velocity = state[1]

    min_ankle_torque = params["min_ankle_torque"]
    max_ankle_torque = params["max_ankle_torque"]

    new_torque = (-mass * gravity * length * np.sin(theta)
                  - mass * length**2 * (theta * kp + angular_velocity * kd))

    return np.clip(new_torque, min_ankle_torque, max_ankle_torque)


def simulate_balance(initial_state, params, kp, kd, dt=1e-3, sim_time=6.0) -> bool:
    local_params = params.copy()
    state = np.asarray(initial_state, float)

    for i in range(int(sim_time / dt)):
        local_params["ankle_torque"] = compute_ankle_torque(state, local_params, kp, kd)
        state = state + dt * model.dynamics(i * dt, state, local_params)
        if abs(state[0]) > 1.5:
            return False

    return abs(state[0]) < 0.01 and abs(state[1]) < 0.01


def map_roa(params, kp, kd, n_theta=96, n_theta_dot=112):
    theta_grid = np.linspace(-0.40, 0.55, n_theta)
    theta_dot_grid = np.linspace(-0.80, 1.40, n_theta_dot)
    stable = np.zeros((n_theta_dot, n_theta), dtype=bool)

    for row in range(n_theta_dot):
        if row % 20 == 0:
            print(f"  row {row}/{n_theta_dot}")
        for col in range(n_theta):
            stable[row, col] = simulate_balance(
                [theta_grid[col], theta_dot_grid[row]], params, kp, kd)

    return theta_grid, theta_dot_grid, stable


def plot_roa(theta_grid, theta_dot_grid, stable, kp, kd, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pcolormesh(theta_grid, theta_dot_grid, stable.astype(int),
                  cmap=ListedColormap(["lightcoral", "lightgreen"]),
                  vmin=0, vmax=1, shading="nearest")
    ax.axvline(0, color="white", ls=":", lw=1)
    ax.set_xlabel(r"$\theta$ [rad]")
    ax.set_ylabel(r"$\dot{\theta}$ [rad/s]")
    ax.set_title(f"Ankle-controller region of attraction, kp={kp} kd={kd}")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def is_in_roa(state, theta_grid, theta_dot_grid, stable) -> bool:
    theta = state[0]
    angular_velocity = state[1]

    if not (theta_grid[0] <= theta <= theta_grid[-1]):
        return False
    if not (theta_dot_grid[0] <= angular_velocity <= theta_dot_grid[-1]):
        return False

    col = np.argmin(np.abs(theta_grid - theta))
    row = np.argmin(np.abs(theta_dot_grid - angular_velocity))
    return bool(stable[row, col])


def simulate_one_step(theta_dot, alpha, params, roa, dt=1e-3, max_time=5.0):
    local_params = params.copy()
    local_params["angle_of_attack"] = alpha
    local_params["ankle_torque"] = 0.0

    theta_grid, theta_dot_grid, stable = roa
    state = np.array([0.0, theta_dot])

    for i in range(int(max_time / dt)):
        next_state = state + dt * model.dynamics(i * dt, state, local_params)

        if is_in_roa(next_state, theta_grid, theta_dot_grid, stable):
            return "caught", None

        if model.event_guard(state, next_state, local_params):
            next_state = model.event_dynamics(next_state, local_params)

        if state[0] < 0.0 <= next_state[0]:
            return "continue", next_state[1]

        state = next_state

    return "lost", None


def build_table(params, roa, n_state=180, n_alpha=20):
    state_grid = np.linspace(0.0, np.sqrt(2 * params["gravity"] / params["length"]),
                             n_state)
    alpha_grid = np.linspace(params["min_angle_of_attack"],
                             params["max_angle_of_attack"], n_alpha)
    next_speed = np.full((n_state, n_alpha), np.nan)
    caught = np.zeros((n_state, n_alpha), dtype=bool)

    for i in range(n_state):
        if i % 30 == 0:
            print(f"  state {i}/{n_state}")
        for j in range(n_alpha):
            status, result = simulate_one_step(state_grid[i], alpha_grid[j],
                                               params, roa)
            if status == "caught":
                caught[i, j] = True
            elif status == "continue":
                next_speed[i, j] = result

    return state_grid, alpha_grid, next_speed, caught


def find_steps(state_grid, alpha_grid, next_speed, caught, roa, mode="min"):
    n_state = len(state_grid)
    steps = np.full(n_state, -1, dtype=int)
    best_alpha = np.full(n_state, np.nan)

    for i in range(n_state):
        if is_in_roa([0.0, state_grid[i]], *roa):
            steps[i] = 0
            continue

        options = []
        for j in range(len(alpha_grid)):
            if caught[i, j]:
                options.append((1, alpha_grid[j]))
            elif np.isfinite(next_speed[i, j]):
                nxt = np.argmin(np.abs(state_grid - next_speed[i, j]))
                if nxt < i and steps[nxt] >= 0:
                    options.append((1 + steps[nxt], alpha_grid[j]))

        if options:
            best_count, best_a = options[0]
            for count, a in options[1:]:
                better = count < best_count if mode == "min" else count > best_count
                if better:
                    best_count, best_a = count, a
            steps[i] = best_count
            best_alpha[i] = best_a

    return steps, best_alpha


def plot_steps(state_grid, steps_min, steps_max, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ok = steps_min >= 0
    ax.plot(state_grid[ok], steps_min[ok], "o-", ms=3, label="fewest steps")
    ok = steps_max >= 0
    ax.plot(state_grid[ok], steps_max[ok], "s--", ms=3, label="most steps")
    ax.set_xlabel(r"initial $\dot{\theta}$ [rad/s]")
    ax.set_ylabel("steps to standstill")
    ax.set_title("Footsteps needed to reach the standing controller")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def roll_out_policy(params, roa, state_grid, best_alpha, start_speed,
                    dt=1e-3, max_steps=40):
    """Replay the table's plan in the continuous simulator.

    Counts one step per angle-of-attack decision, which is how the lookup table
    counts them: a step that ends with the ankle catching the walker mid-swing
    still cost one footstep.
    """
    local_params = params.copy()
    local_params["ankle_torque"] = 0.0

    index = np.argmin(np.abs(state_grid - start_speed))
    state = np.array([0.0, state_grid[index]])
    theta_history = [state[0]]
    theta_dot_history = [state[1]]
    steps_taken = 0

    for _ in range(max_steps):
        if not np.isfinite(best_alpha[index]):
            break

        local_params["angle_of_attack"] = best_alpha[index]
        steps_taken += 1

        for i in range(int(5.0 / dt)):
            next_state = state + dt * model.dynamics(i * dt, state, local_params)

            if is_in_roa(next_state, *roa):
                theta_history.append(next_state[0])
                theta_dot_history.append(next_state[1])
                return theta_history, theta_dot_history, steps_taken

            if model.event_guard(state, next_state, local_params):
                next_state = model.event_dynamics(next_state, local_params)

            crossed = state[0] < 0.0 <= next_state[0]
            state = next_state
            theta_history.append(state[0])
            theta_dot_history.append(state[1])

            if crossed:
                index = np.argmin(np.abs(state_grid - state[1]))
                break
        else:
            break

    return theta_history, theta_dot_history, steps_taken


def plot_trajectory(params, roa, state_grid, alpha_min, alpha_max, start_speed, path):
    fig, ax = plt.subplots(figsize=(8, 5))

    for best_alpha, label, style in [(alpha_min, "fewest steps", "-"),
                                     (alpha_max, "most steps", "--")]:
        theta, theta_dot, n = roll_out_policy(params, roa, state_grid,
                                              best_alpha, start_speed)
        ax.plot(theta, theta_dot, style, lw=1.2, label=f"{label} ({n} steps)")

    ax.set_xlabel(r"$\theta$ [rad]")
    ax.set_ylabel(r"$\dot{\theta}$ [rad/s]")
    ax.set_title(rf"Controlled walker from $\dot{{\theta}}_0$ = {start_speed:.2f} rad/s")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    params = model.generate_params()
    kp, kd = 10.0, 5.0

    print("mapping region of attraction")
    roa = map_roa(params, kp, kd)
    theta_grid, theta_dot_grid, stable = roa
    plot_roa(theta_grid, theta_dot_grid, stable, kp, kd, OUTPUT / "roa.png")

    col = np.argmin(np.abs(theta_grid))
    rows = np.where(stable[:, col])[0]
    print(f"RoA at theta = 0: theta_dot in "
          f"[{theta_dot_grid[rows[0]]:+.3f}, {theta_dot_grid[rows[-1]]:+.3f}]")
    print(f"stable cells: {stable.sum()}/{stable.size}")

    print("building state-action table")
    state_grid, alpha_grid, next_speed, caught = build_table(params, roa)
    print(f"caught cells: {caught.sum()},  completed steps: "
          f"{np.isfinite(next_speed).sum()},  lost: "
          f"{(~caught & ~np.isfinite(next_speed)).sum()}")

    steps_min, alpha_min = find_steps(state_grid, alpha_grid, next_speed,
                                      caught, roa, mode="min")
    steps_max, alpha_max = find_steps(state_grid, alpha_grid, next_speed,
                                      caught, roa, mode="max")
    print("min steps:", steps_min[::20])
    print("max steps:", steps_max[::20])

    plot_steps(state_grid, steps_min, steps_max, OUTPUT / "steps.png")

    candidates = np.where(steps_min >= 3)[0]
    start_speed = state_grid[candidates[0]]
    print(f"trajectory from theta_dot = {start_speed:.3f} "
          f"(min {steps_min[candidates[0]]}, max {steps_max[candidates[0]]})")
    plot_trajectory(params, roa, state_grid, alpha_min, alpha_max,
                    start_speed, OUTPUT / "trajectory.png")
    print(f"figures in {OUTPUT}/")
    


if __name__ == "__main__":
    main()