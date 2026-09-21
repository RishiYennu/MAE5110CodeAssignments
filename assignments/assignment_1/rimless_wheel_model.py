import numpy as np
from scipy.integrate import solve_ivp

EARTH_GRAVITY = 9.81


class RimlessWheel:
    def __init__(self, spoke_count=8, slope_angle=np.deg2rad(5.0), spoke_length=1.0):
        self.spoke_count = spoke_count
        self.slope_angle = slope_angle
        self.spoke_length = spoke_length
        self.gravity = EARTH_GRAVITY

        self.half_spoke_angle = np.pi / spoke_count          # alpha
        self.strike_angle = slope_angle + self.half_spoke_angle   # foot strikes here
        self.launch_angle = slope_angle - self.half_spoke_angle   # new stance starts here
        self.impact_speed_ratio = np.cos(2.0 * self.half_spoke_angle)

    def describe(self):
        return (f"N = {self.spoke_count}, "
                f"alpha = {np.rad2deg(self.half_spoke_angle):.1f} deg, "
                f"gamma = {np.rad2deg(self.slope_angle):.1f} deg")

    def compute_derivative(self, _time, state):
        theta, theta_rate = state
        return [theta_rate, (self.gravity / self.spoke_length) * np.sin(theta)]

    def compute_energy(self, state):
        theta, theta_rate = state
        return 0.5 * self.spoke_length * theta_rate ** 2 + self.gravity * np.cos(theta)

    def build_guards(self):
        def detect_downhill_strike(_time, state):
            return state[0] - self.strike_angle
        detect_downhill_strike.terminal = True
        detect_downhill_strike.direction = 1.0

        def detect_uphill_strike(_time, state):
            return state[0] - self.launch_angle
        detect_uphill_strike.terminal = True
        detect_uphill_strike.direction = -1.0

        return detect_downhill_strike, detect_uphill_strike


    def apply_impact(self, state, stepped_downhill):
        theta, theta_rate = state
        shift = -2.0 * self.half_spoke_angle if stepped_downhill else 2.0 * self.half_spoke_angle
        return np.array([theta + shift, theta_rate * self.impact_speed_ratio])

    def predict_rolling_rate(self):
        """Fixed point of the return map."""
        if np.sin(self.slope_angle) <= 0:
            return 0.0
        return float(np.sqrt(4 * self.gravity / self.spoke_length
                             * np.sin(self.slope_angle) * np.sin(self.half_spoke_angle)
                             * self.impact_speed_ratio ** 2
                             / np.sin(2 * self.half_spoke_angle) ** 2))

    def predict_floquet_multiplier(self):
        return float(self.impact_speed_ratio ** 2)



def simulate(wheel, initial_state, max_impacts=60, rolling_steps=6,
             rest_rate=1e-3, tolerance=1e-10):
    guards = wheel.build_guards()
    state = np.asarray(initial_state, dtype=float)
    clock = 0.0
    post_impact_rates = []
    downhill_run = 0

    for _ in range(max_impacts):
        solution = solve_ivp(wheel.compute_derivative, (clock, clock + 30.0), state,
                             events=guards, rtol=tolerance, atol=tolerance)
        clock = solution.t[-1]
        state = solution.y[:, -1]

        stepped_downhill = solution.t_events[0].size > 0
        if not (stepped_downhill or solution.t_events[1].size > 0):
            return "stopped", post_impact_rates          # never reached a guard

        state = wheel.apply_impact(state, stepped_downhill)

        if stepped_downhill:
            downhill_run += 1
            post_impact_rates.append(float(state[1]))
        else:
            downhill_run = 0

        if abs(state[1]) < rest_rate:
            return "stopped", post_impact_rates
        if downhill_run >= rolling_steps:
            return "rolling", post_impact_rates

    return "stopped", post_impact_rates


def step_return_map(wheel, post_impact_rate, tolerance=1e-12):
    guards = wheel.build_guards()
    solution = solve_ivp(wheel.compute_derivative, (0.0, 60.0),
                         np.array([wheel.launch_angle, float(post_impact_rate)]),
                         events=guards, rtol=tolerance, atol=tolerance)
    if solution.t_events[0].size == 0:
        return None
    return float(wheel.apply_impact(solution.y[:, -1], stepped_downhill=True)[1])