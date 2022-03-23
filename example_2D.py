from common.parameters import SimulationParameters as SIMP

from pytARD_2D.ard import ARDSimulator as ARDS
from pytARD_2D.partition_data import PartitionData as PARTD

import matplotlib.pyplot as plt
from matplotlib import cm as coom
import numpy as np

# Room parameters
duration = 1 # seconds
Fs = 8000 # sample rate
upper_frequency_limit = Fs # Hz
c = 342 # m/s
spatial_samples_per_wave_length = 6

# Procedure parameters
enable_multicore = False
auralize = False
verbose = True
visualize = False

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

partition_1 = PARTD(np.array([[int(c / 25)],[int(c / 25)]]), sim_params)
partition_2 = PARTD(np.array([[int(c / 25)],[int(c / 25)]]), sim_params,do_impulse=False)

part_data = [partition_1, partition_2]

# Instantiating and executing simulation
sim = ARDS(sim_params, part_data)
sim.preprocessing()
sim.simulation()

# Plotting waveform
if True:
    room_dims = np.linspace(0., partition_1.dimensions[0], len(partition_1.pressure_field_results[0]))
    ytop = np.max(partition_1.pressure_field_results)
    ybtm = np.min(partition_1.pressure_field_results)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax_1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    ax_2 = fig.add_subplot(1, 2, 2, projection='3d')

    cool_bionicle_X_1 = np.linspace(0, partition_1.space_divisions_x, partition_1.space_divisions_x)
    cool_bionicle_Y_1 = np.linspace(0, partition_1.space_divisions_y, partition_1.space_divisions_y)
    X_1, Y_1 = np.meshgrid(cool_bionicle_X_1, cool_bionicle_Y_1)

    cool_bionicle_X_2 = np.linspace(0, partition_2.space_divisions_x, partition_2.space_divisions_x)
    cool_bionicle_Y_2 = np.linspace(0, partition_2.space_divisions_y, partition_2.space_divisions_y)
    X_2, Y_2 = np.meshgrid(cool_bionicle_X_2, cool_bionicle_Y_2)

    plot_limit = np.min(partition_2.pressure_field_results[:]), np.max(partition_2.pressure_field_results[:])

    for i in range(0, len(partition_1.pressure_field_results), 50):
        '''
        plt.clf()
        plt.title(f"ARD 2D (t = {(sim_params.T * (i / sim_params.number_of_samples)):.4f}s)")
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
        '''
        Z_1 = partition_1.pressure_field_results[i]
        Z_2 = partition_2.pressure_field_results[i]

        ax_1.cla()
        ax_2.cla()

        plt.title(f"Look at my cool Bionicle© product: (t = {(sim_params.T * (i / sim_params.number_of_samples)):.4f}s)")
        ax_1.plot_surface(X_1, Y_1, Z_1, cmap=coom.coolwarm, antialiased=False)
        ax_2.plot_surface(X_2, Y_2, Z_2, cmap=coom.coolwarm, antialiased=False)

        ax_1.set_zlim(plot_limit)
        ax_2.set_zlim(plot_limit)

        plt.pause(0.005)

    plot_step = 100

