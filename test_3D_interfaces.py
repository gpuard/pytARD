# -*- coding: utf-8 -*-
from pytARD_3D.ard import ARDSimulator
from pytARD_3D.partition import AirPartition3D
from pytARD_3D.interface import InterfaceData3D, Direction3D

from common.parameters import SimulationParameters
from common.impulse import Gaussian, Unit, WaveFile
from common.serializer import Serializer
from common.plotter import Plotter, AnimationPlotter
from common.microphone import Microphone as Mic

import numpy as np

if False:# SUPER FAST
    # Room parameters
    # duration = 1.5 # seconds
    duration = 0.5 # seconds
    Fs = 630 # sample rate
    upper_frequency_limit = 60 # Hz
    # c = 342 # m/s
    c = 4 # m/s
    spatial_samples_per_wave_length = 1
else:
    # Room parameters
    # duration = 1.5 # seconds
    duration = 0.5 # seconds
    Fs = 630 # sample rate
    upper_frequency_limit = 60 # Hz
    # c = 342 # m/s
    c = 4 # m/s
    spatial_samples_per_wave_length = 2

# Procedure parameters
verbose = True
auralize= True
visualize = True
write_to_file = False
compress_file = True

# For Debug
# np.seterr(all='raise')

# Compilation of room parameters into parameter class (don't change this)
sim_param = SimulationParameters(
    upper_frequency_limit, 
    duration, 
    c=c, 
    Fs=Fs,
    spatial_samples_per_wave_length=spatial_samples_per_wave_length, 
    verbose=verbose,
    visualize=visualize
)

SCALE = 150 # Scale of room. Gets calculated by speed of sound divided by SCALE

# Define impulse location
impulse_location = np.array([
    [int(1)], # X, width
    [int(1)], # Y, depth
    [int(1)]  # Z, height
])

# Define impulse location that gets emitted into the room
# impulse = Gaussian(sim_param, impulse_location, 1)
impulse = Unit(sim_param, impulse_location, 20, upper_frequency_limit - 1)
# impulse = WaveFile(sim_param, impulse_location, 'clap.wav', 100) # Uncomment for wave file injection

# room_width = int(c / SCALE)
room_width = int(2)

partitions = []

partitions.append(AirPartition3D(np.array([
    [room_width], # X, width
    [room_width], # Y, depth
    [room_width]  # Z, height
]), sim_param, impulse))

partitions.append(AirPartition3D(np.array([
    [room_width], # X, width
    [room_width], # Y, depth
    [room_width]  # Z, height
]), sim_param))
 
# Interfaces of the room. Interfaces connect the room together
interfaces = []

interfaces.append(InterfaceData3D(0, 1, Direction3D.X))
#interfaces.append(InterfaceData3D(0, 2, Direction3D.X))
#interfaces.append(InterfaceData3D(3, 0, Direction3D.Y))
#interfaces.append(InterfaceData3D(4, 0, Direction3D.Y))

# Initialize & position mics.
mics = []
mics.append(Mic(
    0, [
        int(partitions[0].dimensions[0] / 2), 
        int(partitions[0].dimensions[1] / 2), 
        int(partitions[0].dimensions[2] / 2)
    ], sim_param, "left"))
'''

mics.append(Mic(
    1, [
        int(partitions[1].dimensions[0] / 2), 
        int(partitions[1].dimensions[1] / 2), 
        int(partitions[1].dimensions[2] / 2)
    ], sim_param, "right"))
mics.append(Mic(
    2, [
        int(partitions[2].dimensions[0] / 2), 
        int(partitions[2].dimensions[1] / 2), 
        int(partitions[2].dimensions[2] / 2)
    ], sim_param, "bottom"))
'''
# Instantiation serializer for reading and writing simulation state data
serializer = Serializer(compress=compress_file)

# Instantiating and executing simulation (don't change this)
sim = ARDSimulator(sim_param, partitions, 1, interfaces, mics)
sim.preprocessing()
sim.simulation()

# Find best peak to normalize mic signal and write mic signal to file
if auralize:
    def find_best_peak(mics):
        peaks = []
        for i in range(len(mics)):
            peaks.append(np.max(mics[i].signal))
        return np.max(peaks)

    all_mic_peaks = []
    all_mic_peaks.append(find_best_peak(mics))
    best_peak = np.max(all_mic_peaks)

    def write_mic_files(mics, peak):
        for i in range(len(mics)):
            mics[i].write_to_file(peak, upper_frequency_limit)

    write_mic_files(mics, best_peak)

# Write partitions and state data to disk
if write_to_file:
    if verbose:
        print("Writing state data to disk. Please wait...")
    serializer.dump(sim_param, partitions)


# Plotting waveform
if visualize:
    a=0
    if a == 0:
        pfX0 = partitions[0].pressure_field_results
        pfX1 = partitions[1].pressure_field_results
        
        #AXIS
        # 0 Z
        # 1 y
        # 2 X
        
        fps=30
        pf_t = [np.concatenate((pfX0[t],pfX1[t]),axis=2) for t in range(sim_param.number_of_samples)]
        anim = AnimationPlotter().plot_3D(pf_t, 
                                          sim_param, 
                                          interval = 1000 / fps, # in ms
                                          zyx=partitions[0].src_grid_loc)
    else:
        plotter = Plotter()
        plotter.set_data_from_simulation(sim_param, partitions)
        plotter.plot_3D()