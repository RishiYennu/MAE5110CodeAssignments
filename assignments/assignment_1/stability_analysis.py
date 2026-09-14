import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from rimless_wheel_model import RimlessWheel, simulate, step_return_map

FIGURES = Path(__file__).parent / "figures"
ROLLING_COLOUR, STOPPED_COLOUR, LINE_COLOUR = "#2F6F4E", "#B4643C", "#3B4CA8"


def tidy(axes, title, xlabel, ylabel):
    axes.set_title(title, fontsize=10, loc="left")
    axes.set_xlabel(xlabel, fontsize=9)
    axes.set_ylabel(ylabel, fontsize=9)
    axes.tick_params(labelsize=8)
    axes.grid(alpha=0.15, linewidth=0.6)
    for edge in ("top", "right"):
        axes.spines[edge].set_visible(False)


# ------------------------------------------------------------- sanity checking


def check_model(wheel):
    """Three checks, each isolating one part of the model.

    Each has a prediction stated before it runs, so a wrong model fails loudly
    rather than producing a plausible-looking number.
    """
    print("Sanity checks")
    figure, axes_row = plt.subplots(1, 3, figsize=(13, 3.8))
    flat_wheel = RimlessWheel(spoke_count=wheel.spoke_count, slope_angle=0.0)

    # 1. Only the smooth flow. Energy is conserved between impacts, so any drift
    #    is the integrator, not the physics. Expect it at the solver tolerance.
    solution = solve_ivp(flat_wheel.compute_derivative, (0, 1.0), [0.05, 0.0],
                         rtol=1e-11, atol=1e-11, dense_output=True)
    times = np.linspace(0, 1.0, 400)
    energy = np.array([flat_wheel.compute_energy(s) for s in solution.sol(times).T])
    drift = np.ptp(energy)
    print(f"  energy drift between impacts : {drift:.2e}   (expect ~0)")
    axes_row[0].plot(times, energy - energy[0], color=LINE_COLOUR, linewidth=1.4)
    tidy(axes_row[0], f"1. Smooth flow only\nenergy drift {drift:.1e}, expected ~0",
         "time [s]", "E(t) - E(0)")

    # 2. Only the collision law. Every impact must scale the rate by cos(2 alpha)
    #    exactly. A constant but wrong ratio means the alpha convention is off; a
    #    drifting ratio means the event localisation is sloppy.
    guards = wheel.build_guards()
    state = np.array([wheel.launch_angle, 2.5])
    ratios = []
    for _ in range(12):
        segment = solve_ivp(wheel.compute_derivative, (0, 30), state,
                            events=guards, rtol=1e-11, atol=1e-11)
        struck = segment.y[:, -1]
        after = wheel.apply_impact(struck, segment.t_events[0].size > 0)
        ratios.append(abs(after[1] / struck[1]))
        state = after
    ratio_error = max(abs(r - wheel.impact_speed_ratio) for r in ratios)
    print(f"  impact rate ratio            : {ratios[0]:.12f}   "
          f"(expect {wheel.impact_speed_ratio:.12f})")
    axes_row[1].plot(range(1, 13), ratios, "o-", color=LINE_COLOUR, markersize=4)
    axes_row[1].axhline(wheel.impact_speed_ratio, color=STOPPED_COLOUR, linestyle="--")
    tidy(axes_row[1], f"2. Collision law only\nmax error {ratio_error:.1e} from cos(2a)",
         "impact number", "|rate after| / |rate before|")

    # 3. The whole system, against a prediction needing no computation: with no
    #    slope, gravity adds nothing and impacts subtract, so it must stop.
    outcomes = {simulate(flat_wheel, [flat_wheel.launch_angle, rate])[0]
                for rate in (0.5, 2.0, 6.0)}
    print(f"  flat ground outcomes         : {outcomes}   (expect only stopped)")
    _, decay = simulate(flat_wheel, [flat_wheel.launch_angle, 6.0], rolling_steps=99)
    axes_row[2].semilogy(range(1, len(decay) + 1), decay, "o-",
                         color=STOPPED_COLOUR, markersize=4)
    tidy(axes_row[2], f"3. Whole system on flat ground\noutcomes {outcomes}, expected stopped",
         "downhill step", "post-impact rate [rad/s]")

    figure.suptitle(f"Sanity checks   {wheel.describe()}", fontsize=11)
    figure.tight_layout()
    figure.savefig(FIGURES / "sanity_checks.png", dpi=150)
    plt.close(figure)


def compare_speed(wheel, sample_count=40, seed=0):
    """Show the early exit is faster AND gives the same classification.

    The agreement count is the important half: it says the shortcut is free
    rather than an approximation traded for speed.
    """
    generator = np.random.default_rng(seed)
    points = [[generator.uniform(wheel.launch_angle, wheel.strike_angle),
               generator.uniform(-6.0, 6.0)] for _ in range(sample_count)]

    start = time.perf_counter()
    quick = [simulate(wheel, point)[0] for point in points]
    quick_seconds = time.perf_counter() - start

    start = time.perf_counter()
    thorough = [simulate(wheel, point, rolling_steps=40)[0] for point in points]
    thorough_seconds = time.perf_counter() - start

    agree = sum(a == b for a, b in zip(quick, thorough))
    print(f"  early exit {1e3 * quick_seconds / sample_count:6.1f} ms/point, "
          f"run to steady state {1e3 * thorough_seconds / sample_count:6.1f} ms/point")
    print(f"  speedup {thorough_seconds / quick_seconds:.1f}x with "
          f"{agree}/{sample_count} identical classifications")


# --------------------------------------------------------- regions of attraction


def map_regions_of_attraction(wheel, angle_samples=51, rate_samples=71, rate_limit=6.0):
    """Grid the state space and simulate from every point.

    The stance angle can only lie between the two guards; outside that wedge a
    different spoke would already be touching the ground.
    """
    angles = np.linspace(wheel.launch_angle, wheel.strike_angle, angle_samples)
    rates = np.linspace(-rate_limit, rate_limit, rate_samples)
    classification = np.zeros((rate_samples, angle_samples))

    for row, rate in enumerate(rates):
        for column, angle in enumerate(angles):
            outcome, _ = simulate(wheel, [angle, rate])
            classification[row, column] = 1.0 if outcome == "rolling" else 0.0

    return angles, rates, classification


def plot_regions_of_attraction(wheel, angles, rates, classification, fixed_point):
    """Both attractors on one state-space map, each with its basin."""
    rolling_fraction = classification.mean()
    figure, axes = plt.subplots(figsize=(8.5, 6))
    axes.imshow(classification, origin="lower", aspect="auto",
                extent=[np.rad2deg(angles[0]), np.rad2deg(angles[-1]),
                        rates[0], rates[-1]],
                cmap=matplotlib.colors.ListedColormap([STOPPED_COLOUR, ROLLING_COLOUR]))

    # Attractor 1: the rolling limit cycle. It lives on the Poincare section, so
    # it appears as a single point at the launch angle.
    axes.plot([np.rad2deg(wheel.launch_angle)], [fixed_point], "*", color="white",
              markersize=20, markeredgecolor="black", markeredgewidth=0.8,
              label=f"limit cycle, v* = {fixed_point:.3f} rad/s")

    # Attractor 2: the resting state. The wheel ends up straddling two spokes
    # with no motion, which is the zero-rate line reached through a Zeno
    # accumulation of ever smaller rocks.
    axes.plot([np.rad2deg(wheel.slope_angle)], [0.0], "o", color="white",
              markersize=11, markeredgecolor="black", markeredgewidth=0.8,
              label="resting state, rocked to a halt")

    axes.axvline(0, color="white", linestyle=":", linewidth=1.2,
                 label="apex, theta = 0")
    axes.set_xlabel("stance angle theta [deg]")
    axes.set_ylabel("stance rate theta-dot [rad/s]")
    axes.set_title(f"Regions of attraction   {wheel.describe()}\n"
                   f"green reaches the limit cycle ({100 * rolling_fraction:.0f}% of the "
                   f"wedge), brown rocks to rest", fontsize=10)
    handles = [matplotlib.patches.Patch(color=ROLLING_COLOUR, label="basin of the limit cycle"),
               matplotlib.patches.Patch(color=STOPPED_COLOUR, label="basin of the resting state")]
    handles += axes.get_legend_handles_labels()[0]
    axes.legend(handles=handles, fontsize=8, loc="lower left", framealpha=0.9)
    figure.tight_layout()
    figure.savefig(FIGURES / "regions_of_attraction.png", dpi=150)
    plt.close(figure)
    return rolling_fraction


# ---------------------------------------------------- return map and multiplier


def find_fixed_point(wheel):
    """Solve P(v) = v on the simulated map. None when no rolling cycle exists.

    The map is undefined below the rate needed to clear the apex, so bisect for
    that boundary first. Stepping down to it on a fixed grid would overshoot the
    fixed point on shallow slopes, where the two sit very close together.
    """
    def residual(rate):
        image = step_return_map(wheel, rate)
        return np.nan if image is None else image - rate

    slowest, fastest = 1e-6, 20.0
    if np.isnan(residual(fastest)):
        return None
    if not np.isnan(residual(slowest)):
        lowest_rollable = slowest
    else:
        for _ in range(60):
            middle = 0.5 * (slowest + fastest)
            if np.isnan(residual(middle)):
                slowest = middle
            else:
                fastest = middle
        lowest_rollable = fastest

    if residual(lowest_rollable) * residual(20.0) > 0:
        return None
    return brentq(residual, lowest_rollable, 20.0, xtol=1e-12)


def estimate_floquet_multiplier(wheel, fixed_point, perturbation=1e-5):
    """Local slope of the return map at its fixed point, by central difference.

    Perturb the post-impact rate on both sides of v*, take one step from each,
    and divide the change in output by the change in input. A disturbance is
    multiplied by this number once per step, so the cycle is stable exactly when
    its magnitude is below one.
    """
    if fixed_point is None:
        return np.nan
    above = step_return_map(wheel, fixed_point + perturbation)
    below = step_return_map(wheel, fixed_point - perturbation)
    return (above - below) / (2 * perturbation)


def plot_return_map(wheel, fixed_point, multiplier):
    rates = np.linspace(0.05, 5.0, 140)
    images = np.array([np.nan if (image := step_return_map(wheel, rate)) is None
                       else image for rate in rates])

    figure, axes = plt.subplots(figsize=(7, 6.4))
    axes.plot([0, 5], [0, 5], "--", color="grey", linewidth=1.3,
              label="identity, v(n+1) = v(n)")
    axes.plot(rates, images, color=LINE_COLOUR, linewidth=2.2,
              label="return map from simulation")

    rate = 4.5                                   # cobweb, to show the iteration
    cobweb_x, cobweb_y = [rate], [0]
    for _ in range(14):
        image = step_return_map(wheel, rate)
        if image is None:
            break
        cobweb_x += [rate, image]
        cobweb_y += [image, image]
        rate = image
    axes.plot(cobweb_x, cobweb_y, color="grey", linewidth=0.8,
              label="cobweb: successive steps")

    tangent = np.linspace(fixed_point - 1.0, fixed_point + 1.0, 10)
    axes.plot(tangent, fixed_point + multiplier * (tangent - fixed_point),
              color=STOPPED_COLOUR, linestyle="-.", linewidth=1.4,
              label=f"local slope = Floquet multiplier = {multiplier:.4f}")
    axes.plot([fixed_point], [fixed_point], "o", color=ROLLING_COLOUR,
              markersize=11, markeredgecolor="white", markeredgewidth=1.2,
              zorder=5, label=f"fixed point v* = {fixed_point:.4f} rad/s")

    axes.set_xlim(0, 5)
    axes.set_ylim(0, 5)
    axes.set_aspect("equal")
    axes.set_xlabel("post-impact rate at step n [rad/s]")
    axes.set_ylabel("post-impact rate at step n+1 [rad/s]")
    axes.set_title(f"Poincare return map at the contact event   {wheel.describe()}",
                   fontsize=10)
    axes.legend(fontsize=8, frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(FIGURES / "return_map.png", dpi=150)
    plt.close(figure)


# ------------------------------------------------------------------- the sweeps


def sweep(wheels, angle_samples=35, rate_samples=45):
    """Measure the rolling basin and the multiplier across a list of wheels."""
    basins, multipliers, predictions = [], [], []
    for wheel in wheels:
        _, _, classification = map_regions_of_attraction(wheel, angle_samples, rate_samples)
        fixed_point = find_fixed_point(wheel)
        multiplier = estimate_floquet_multiplier(wheel, fixed_point)
        basins.append(classification.mean())
        multipliers.append(multiplier)
        predictions.append(wheel.predict_floquet_multiplier())
        note = "   (no rolling cycle exists)" if fixed_point is None else ""
        print(f"  {wheel.describe():<42} basin {100 * basins[-1]:5.1f}%   "
              f"multiplier {multiplier:.4f}   predicted {predictions[-1]:.4f}{note}")
    return np.array(basins), np.array(multipliers), np.array(predictions)


def plot_sweeps(slope_degrees, slope_results, spoke_counts, spoke_results):
    figure, axes_grid = plt.subplots(2, 2, figsize=(11, 7.5))
    rows = [(slope_degrees, slope_results, "slope gamma [deg]", "Slope"),
            (spoke_counts, spoke_results, "spoke count N", "Spokes")]

    for row, (x_values, results, x_label, name) in enumerate(rows):
        basins, multipliers, predictions = results

        axes = axes_grid[row, 0]
        axes.plot(x_values, 100 * basins, "o-", color=ROLLING_COLOUR, markersize=5)
        tidy(axes, f"{name} vs size of the rolling basin", x_label,
             "rolling basin [% of wedge]")

        axes = axes_grid[row, 1]
        axes.plot(x_values, multipliers, "o", color=LINE_COLOUR, markersize=6,
                  label="measured by finite difference")
        axes.plot(x_values, predictions, "--", color=STOPPED_COLOUR, linewidth=1.4,
                  label="cos^2(2 alpha)")
        axes.set_ylim(0, 1)
        axes.legend(fontsize=8, frameon=False)
        tidy(axes, f"{name} vs local convergence rate", x_label, "Floquet multiplier")

    figure.suptitle("Slope changes the basin but not the convergence rate; "
                    "spoke count changes both", fontsize=11)
    figure.tight_layout()
    figure.savefig(FIGURES / "sweeps.png", dpi=150)
    plt.close(figure)


# ------------------------------------------------------------------------ main


def main():
    FIGURES.mkdir(exist_ok=True)
    wheel = RimlessWheel(spoke_count=8, slope_angle=np.deg2rad(5.0))
    print(wheel.describe(), "\n")

    check_model(wheel)

    print("\nCost of the brute force")
    compare_speed(wheel)

    print("\nReturn map and Floquet multiplier")
    fixed_point = find_fixed_point(wheel)
    multiplier = estimate_floquet_multiplier(wheel, fixed_point)
    plot_return_map(wheel, fixed_point, multiplier)
    print(f"  fixed point simulated {fixed_point:.9f} rad/s")
    print(f"  fixed point predicted {wheel.predict_rolling_rate():.9f} rad/s")
    print(f"  multiplier measured   {multiplier:.9f}")
    print(f"  multiplier predicted  {wheel.predict_floquet_multiplier():.9f}")

    print("\nRegions of attraction")
    angles, rates, classification = map_regions_of_attraction(wheel)
    rolling_fraction = plot_regions_of_attraction(wheel, angles, rates,
                                                  classification, fixed_point)
    print(f"  rolling basin covers {100 * rolling_fraction:.1f}% of the wedge")

    print("\nSweeping the slope")
    slope_degrees = np.arange(4.0, 21.0, 2.0)
    slope_results = sweep([RimlessWheel(8, np.deg2rad(d)) for d in slope_degrees])

    print("\nSweeping the spoke count")
    spoke_counts = np.arange(6, 13)
    spoke_results = sweep([RimlessWheel(int(n), np.deg2rad(5.0)) for n in spoke_counts])

    plot_sweeps(slope_degrees, slope_results, spoke_counts, spoke_results)
    print(f"\nFigures written to {FIGURES}/")


if __name__ == "__main__":
    main()