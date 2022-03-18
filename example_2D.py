from pytARD_2D.ard import ARDSimulator as ARDS
from pytARD_2D.parameters import SimulationParameters as SIMP
from pytARD_2D.partition_data import PartitionData as PARTD
from pytARD_2D.microphone import Microphone as Mic
import matplotlib.pyplot as plt
import numpy as np

# Room parameters
src_pos = [0] # m
duration = 2 # seconds
Fs = 8000 # sample rate
upper_frequency_limit = Fs # Hz
c = 342 # m/s
spatial_samples_per_wave_length = 12

# Procedure parameters
enable_multicore = False
auralize = False
verbose = True
visualize = True

# Compilation of room parameters into parameter class
sim_params = SIMP(
    upper_frequency_limit, 
    duration, 
    c=c, 
    Fs=Fs,
    spatial_samples_per_wave_length=spatial_samples_per_wave_length, 
    enable_multicore=enable_multicore, 
    verbose=verbose,
    visualize=visualize
)

partition_1 = PARTD(np.array([c * 2]), sim_params)
partition_2 = PARTD(np.array([c * 2]), sim_params,do_impulse=False)

part_data = [partition_1, partition_2]

# Instantiating and executing simulation
sim = ARDS(sim_params, part_data)
sim.preprocessing()
sim.simulation()

# Plotting waveform
if visualize:
    room_dims = np.linspace(0., partition_1.dimensions[0], len(partition_1.pressure_field_results[0]))
    ytop = np.max(partition_1.pressure_field_results)
    ybtm = np.min(partition_1.pressure_field_results)

    plt.figure()
    for i in range(0, len(partition_1.pressure_field_results), 50):
        plt.clf()
        plt.title(f"ARD 1D (t = {(sim_params.T * (i / sim_params.number_of_samples)):.4f}s)")
        plt.subplot(1, 2, 1)
        plt.plot(room_dims, partition_1.pressure_field_results[i], 'r', linewidth=1)
        plt.ylim(top=ytop)
        plt.ylim(bottom=ybtm)
        plt.vlines(np.min(room_dims), ybtm, ytop, color='gray')
        plt.vlines(np.max(room_dims), ybtm, ytop, color='gray')
        plt.subplot(1, 2, 2)
        plt.plot(room_dims, partition_2.pressure_field_results[i], 'b', linewidth=1)
        plt.xlabel("Position [m]")
        plt.ylabel("Displacement")
        plt.ylim(top=ytop)
        plt.ylim(bottom=ybtm)
        plt.vlines(np.min(room_dims), ybtm, ytop, color='gray')
        plt.vlines(np.max(room_dims), ybtm, ytop, color='gray')
        plt.grid()
        plt.pause(0.001)

    plot_step = 100

