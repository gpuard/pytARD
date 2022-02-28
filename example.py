from ard.ard import ARDSimulator as ARDS
from ard.parameters import ARDParamters as ARDP
import matplotlib.pyplot as plt
import numpy as np


# param = ARDP([1], 1000, 1)
room_len = [1]
src_pos = [.75]
sim = ARDS(room_len, src_pos, 1000, 1, c=343, spatial_samples_per_wave_length=2, verbose=True, visualize=True)

sim.preprocessing()
sim.simulation()

room_dims = np.linspace(0., room_len[0], len(sim.pressure_field_results[0]))
ytop = np.max(sim.pressure_field_results)
ybtm = np.min(sim.pressure_field_results)

plt.figure()
for i in range(0, len(sim.pressure_field_results), 50):
    plt.clf()
    plt.title(f"ARD 1D (t = {(sim.T * (i / sim.number_of_samples)):.4f}s)")
    plt.plot(room_dims, sim.pressure_field_results[i])
    plt.xlabel("Position [m]")
    plt.ylabel("Displacement")
    plt.ylim(top=ytop)
    plt.ylim(bottom=ybtm)
    plt.vlines(np.min(room_dims), ybtm, ytop, color='gray')
    plt.vlines(np.max(room_dims), ybtm, ytop, color='gray')
    plt.grid()
    plt.pause(0.001)
