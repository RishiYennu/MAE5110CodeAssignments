import numpy as np
import matplotlib.pyplot as plt

from models import ball as model
from integrators import explicit_euler as integrator1
from integrators import rk4 as integrator2
import timeit

# Basic simulation of the pendulum

params = {
    "gravity": 9.81,  # gravity m/s^2)
    "length": 1,  # rod length (m)
    "mass": 0.2,  # point mass at end of rod (kg)
    "damping_coeff": 0.0,  # damping coefficient (kg*m^2/s)
}

timestep = 1e-5
# some set-up

initial_state = np.array([np.pi / 4, 0.0])


sim_time = 5.0
n = 10
# euler = timeit.timeit(   # pendulum
#     lambda: integrator1.run_integrator(sim_time, 0.001, initial_state, params, model),
#     number=n,
# )
# rk4 = timeit.timeit(
#     lambda: integrator2.run_integrator(sim_time, 0.05, initial_state, params, model),
#     number=n,
# )
# print(f"Euler: {euler/n*1e3:.3f} ms per run")
# print(f"RK4:   {rk4/n*1e3:.3f} ms per run")
time_traj, state_traj = integrator1.run_integrator(sim_time,timestep,initial_state,params,model)


# sanity check the energies: since there is no actuation, and no damping, total energy should stay
# constant. If we turn on the damping coefficient, it should slowly bleed out energy until it comes to
# a stand-still.

kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)

plt.figure()
plt.plot(time_traj, kinetic_energy, label="Kinetic Energy")
plt.plot(time_traj, potential_energy, label="Potential Energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Energy")
plt.legend()
plt.tight_layout()
plt.show()

# # TODO: make a phase portrait plot
# plt.figure()
# plt.plot(potential_energy, kinetic_energy, label="Potential energy vs Kinetic")
# plt.xlabel("Potential Energy (J)")
# plt.ylabel("Kinetic Energy (J)")
# plt.title("Pendulum energy")
# plt.legend()
# plt.tight_layout()
# plt.show()
