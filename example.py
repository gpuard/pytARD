from ard.ard import ARDSimulator as ARDS
from ard.parameters import ARDParameters as ARDP
import matplotlib.pyplot as plt
import numpy as np

# Room parameters
room_len = [1] # m
src_pos = [.75] # m
duration = 1 # seconds
upper_frequency_limit = 1000 # Hz
c = 343 # m/s
# Compilation of room parameters into parameter class
params = ARDP(room_len, src_pos, upper_frequency_limit, duration, c=c, spatial_samples_per_wave_length=12, verbose=True, visualize=True)

# Instantiating and executing simulation
sim = ARDS(params)
sim.preprocessing()
sim.simulation()

# Plotting waveform
room_dims = np.linspace(0., room_len[0], len(sim.pressure_field_results[0]))
ytop = np.max(sim.pressure_field_results)
ybtm = np.min(sim.pressure_field_results)

plt.figure()
for i in range(0, len(sim.pressure_field_results), 50):
    plt.clf()
    plt.title(f"ARD 1D (t = {(params.T * (i / params.number_of_samples)):.4f}s)")
    plt.plot(room_dims, sim.pressure_field_results[i])
    plt.xlabel("Position [m]")
    plt.ylabel("Displacement")
    plt.ylim(top=ytop)
    plt.ylim(bottom=ybtm)
    plt.vlines(np.min(room_dims), ybtm, ytop, color='gray')
    plt.vlines(np.max(room_dims), ybtm, ytop, color='gray')
    plt.grid()
    plt.pause(0.001)

plot_step = 100

