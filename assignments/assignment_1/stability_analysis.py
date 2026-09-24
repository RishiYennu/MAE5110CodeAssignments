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

def check_energy(wheel):

    guards = wheel.build_guards()
    state = np.array([wheel.launch_angle, 3.0])
    clock = 0.0
    times, energies, drift, ratio_error = [], [], 0.0, 0.0

    for _ in range(6):
        swing = solve_ivp(wheel.compute_derivative, (clock, clock + 30.0), state,
                          events=guards, rtol=1e-11, atol=1e-11, dense_output=True)
        sample_times = np.linspace(swing.t[0], swing.t[-1], 200)
        sample_energy = [wheel.compute_energy(s) for s in swing.sol(sample_times).T]
        times.append(sample_times)
        energies.append(sample_energy)
        drift = max(drift, np.ptp(sample_energy))

        clock = swing.t[-1]
        before = swing.y[:, -1]
        after = wheel.apply_impact(before, swing.t_events[0].size > 0)
        ratio_error = max(ratio_error,
                          abs((after[1] / before[1]) ** 2 - wheel.impact_speed_ratio ** 2))
        state = after

    print("Sanity check: energy")
    print(f"  drift while swinging  : {drift:.2e}   (expect 0)")
    print(f"  kept at impact        : {wheel.impact_speed_ratio ** 2:.12f}   "
          f"(worst error {ratio_error:.1e})")

    figure, axes = plt.subplots(figsize=(8, 4.5))
    for sample_times, sample_energy in zip(times, energies):
        axes.plot(sample_times, sample_energy, color=LINE_COLOUR, linewidth=2.2)
    for sample_times in times[:-1]:
        axes.axvline(sample_times[-1], color=STOPPED_COLOUR, linestyle=":", linewidth=1.1)
    axes.plot([], [], color=LINE_COLOUR, linewidth=2.2, label="swinging: energy conserved")
    axes.plot([], [], color=STOPPED_COLOUR, linestyle=":", label="impact: energy lost")
    axes.legend(fontsize=8, frameon=False)
    axes.set_xlabel("time [s]")
    axes.set_ylabel("energy per unit mass per unit length")
    axes.set_title(f"Sanity check: energy   {wheel.describe()}\n"
                   f"flat to {drift:.0e} between impacts, each impact keeps "
                   f"cos^2(2a) = {wheel.impact_speed_ratio ** 2:.3f}", fontsize=10)
    figure.tight_layout()
    figure.savefig(FIGURES / "sanity_check_energy.png", dpi=150)
    plt.close(figure)


def compare_speed(wheel, sample_count=40, seed=0):
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


def map_regions_of_attraction(wheel, angle_samples=51, rate_samples=71, rate_limit=6.0):
    angles = np.linspace(wheel.launch_angle, wheel.strike_angle, angle_samples)
    rates = np.linspace(-rate_limit, rate_limit, rate_samples)
    classification = np.zeros((rate_samples, angle_samples))

    for row, rate in enumerate(rates):
        for column, angle in enumerate(angles):
            outcome, _ = simulate(wheel, [angle, rate])
            classification[row, column] = 1.0 if outcome == "rolling" else 0.0

    return angles, rates, classification


def plot_regions_of_attraction(wheel, angles, rates, classification, fixed_point):
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

def sweep(wheels, angle_samples=35, rate_samples=45):
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
        axes.set_title(f"{name} vs size of the region of attraction", fontsize=10)
        axes.set_xlabel(x_label)
        axes.set_ylabel("region of attraction [% of wedge]")

        axes = axes_grid[row, 1]
        axes.plot(x_values, multipliers, "o", color=LINE_COLOUR, markersize=6,
                  label="measured by finite difference")
        axes.plot(x_values, predictions, "--", color=STOPPED_COLOUR, linewidth=1.4,
                  label="cos^2(2 alpha)")
        axes.set_ylim(0, 1)
        axes.legend(fontsize=8, frameon=False)
        axes.set_title(f"{name} vs local convergence rate", fontsize=10)
        axes.set_xlabel(x_label)
        axes.set_ylabel("Floquet multiplier")

    figure.suptitle("Slope changes the basin but not the convergence rate; "
                    "spoke count changes both", fontsize=11)
    figure.tight_layout()
    figure.savefig(FIGURES / "sweeps.png", dpi=150)
    plt.close(figure)

def main():
    FIGURES.mkdir(exist_ok=True)
    wheel = RimlessWheel(spoke_count=8, slope_angle=np.deg2rad(5.0))
    print(wheel.describe(), "\n")

    check_energy(wheel)

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
    print(f"  region of attraction covers {100 * rolling_fraction:.1f}% of the wedge")

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