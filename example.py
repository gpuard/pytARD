from ard.ard import ARDSimulator as ARDS
from ard.parameters import ARDParameters as ARDP
import matplotlib.pyplot as plt
import numpy as np

# Room parameters
room_len = np.array([342]) # m
src_pos = [0] # m
duration = 2 # seconds
Fs = 8000 # sample rate
upper_frequency_limit = Fs # Hz
c = 342 # m/s

# Procedure parameters
enable_multicore = False
auralize = False
verbose = True
visualize = True

# Compilation of room parameters into parameter class
params = ARDP(
    room_len, 
    src_pos, 
    upper_frequency_limit, 
    duration, 
    c=c, 
    Fs=Fs,
    spatial_samples_per_wave_length=4, 
    enable_multicore=enable_multicore, 
    verbose=verbose,
    visualize=visualize
)

# Instantiating and executing simulation
sim = ARDS(params)
sim.preprocessing()
sim.simulation()

# Plotting waveform
if visualize:
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

