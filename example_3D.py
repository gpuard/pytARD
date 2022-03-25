from common.parameters import SimulationParameters as SIMP

from pytARD_3D.ard import ARDSimulator as ARDS
from pytARD_3D.partition_data import PartitionData as PARTD
from common.impulse import Gaussian, WaveFile

import matplotlib.pyplot as plt
import numpy as np

# Room parameters
duration = 3 # seconds
Fs = 8000 # sample rate
upper_frequency_limit = Fs # Hz
c = 342 # m/s
spatial_samples_per_wave_length = 6

# Procedure parameters
enable_multicore = False
auralize = False
verbose = True
visualize = True

# Compilation of room parameters into parameter class (don't change this)
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

SCALE = 50 # Scale of room. Gets calculated by speed of sound divided by SCALE

# Define impulse location
impulse_location = np.array([
    [int((c / SCALE) / 2)], # X, width
    [int((c / SCALE) / 2)], # Y, depth
    [int((c / SCALE) / 2)]  # Z, height
])

# Define impulse location that gets emitted into the room
# impulse = Gaussian(sim_params, impulse_location, 10000)
impulse = WaveFile(sim_params, impulse_location, 'clap.wav', 100) # Uncomment for wave file injection

partition_1 = PARTD(np.array([
    [int(c / SCALE)], # X, width
    [int(c / SCALE)], # Y, depth
    [int(c / SCALE)]  # Z, height
]), sim_params, impulse)

# Compilation of all partitions into complete partition data (don't change this line)
part_data = [partition_1]

# Instantiating and executing simulation (don't change this)
sim = ARDS(sim_params, part_data)
sim.preprocessing()
sim.simulation()

# Plotting waveform
if visualize:
    room_dims = np.linspace(0., partition_1.dimensions[0], len(partition_1.pressure_field_results[0]))
    ytop = np.max(partition_1.pressure_field_results)
    ybtm = np.min(partition_1.pressure_field_results)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax_1 = fig.add_subplot(2, 2, 1)
    # ax_2 = fig.add_subplot(2, 2, 2)
    # ax_3 = fig.add_subplot(2, 2, 4)

    temp_X_1 = np.linspace(0, partition_1.space_divisions_x, partition_1.space_divisions_x)
    temp_Y_1 = np.linspace(0, partition_1.space_divisions_y, partition_1.space_divisions_y)
    X_1, Y_1 = np.meshgrid(temp_X_1, temp_Y_1)
    
    '''
    temp_X_2 = np.linspace(0, partition_2.space_divisions_x, partition_2.space_divisions_x)
    temp_Y_2 = np.linspace(0, partition_2.space_divisions_y, partition_2.space_divisions_y)
    X_2, Y_2 = np.meshgrid(temp_X_2, temp_Y_2)

    temp_X_3 = np.linspace(0, partition_3.space_divisions_x, partition_3.space_divisions_x)
    temp_Y_3 = np.linspace(0, partition_3.space_divisions_y, partition_3.space_divisions_y)
    X_3, Y_3 = np.meshgrid(temp_X_3, temp_Y_3)
    '''

    plot_limit_min = np.min(partition_1.pressure_field_results[:])
    plot_limit_max = np.max(partition_1.pressure_field_results[:])

    vertical_slice = int(partition_1.space_divisions_z / 2) # middle of Z axis

    for i in range(0, len(partition_1.pressure_field_results), 50):
        Z_1 = partition_1.pressure_field_results[i][vertical_slice]
        # Z_2 = partition_2.pressure_field_results[i]
        # Z_3 = partition_3.pressure_field_results[i]

        ax_1.cla()
        # ax_2.cla()
        # ax_3.cla()

        plt.title(f"t = {(sim_params.T * (i / sim_params.number_of_samples)):.4f}s")

        ax_1.imshow(Z_1)
        # ax_2.imshow(Z_2)
        # ax_3.imshow(Z_3)

        plt.pause(0.005)

    plot_step = 100

