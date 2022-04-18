from pytARD_3D.ard import ARDSimulator as ARDS
from pytARD_3D.partition_data import PartitionData as PARTD
from pytARD_3D.interface import InterfaceData3D, Direction3D

from common.parameters import SimulationParameters as SIMP
from common.impulse import Gaussian, Unit, WaveFile
from common.serializer import Serializer
from common.plotter import Plotter
from common.microphone import Microphone as Mic

import matplotlib.pyplot as plt
import numpy as np

# Room parameters
duration = 1 # seconds
Fs = 8000 # sample rate
upper_frequency_limit = Fs / 10 # Hz
c = 342 # m/s
spatial_samples_per_wave_length = 6

# Procedure parameters
verbose = True
auralize= True
visualize = True
write_to_file = False
compress_file = True

# For Debug
# np.seterr(all='raise')

# Compilation of room parameters into parameter class (don't change this)
sim_param = SIMP(
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
    [int((c / SCALE) / 2)], # X, width
    [int((c / SCALE) / 2)], # Y, depth
    [int((c / SCALE) / 2)]  # Z, height
])

# Define impulse location that gets emitted into the room
# impulse = Gaussian(sim_params, impulse_location, 1)
impulse = Unit(sim_param, impulse_location, 1, upper_frequency_limit - 1)
# impulse = WaveFile(sim_params, impulse_location, 'clap.wav', 100) # Uncomment for wave file injection

partition_1 = PARTD(np.array([
    [int(c / SCALE)], # X, width
    [int(c / SCALE)], # Y, depth
    [int(c / SCALE)]  # Z, height
]), sim_param, impulse)

partition_2 = PARTD(np.array([
    [int(c / SCALE)], # X, width
    [int(c / SCALE)], # Y, depth
    [int(c / SCALE)]  # Z, height
]), sim_param)

partition_3 = PARTD(np.array([
    [int(c / SCALE)], # X, width
    [int(c / SCALE)], # Y, depth
    [int(c / SCALE)]  # Z, height
]), sim_param)

# Compilation of all partitions into complete partition data (don't change this line)
part_data = [partition_1, partition_2, partition_3]

# Interfaces of the room. Interfaces connect the room together

interfaces = []
interfaces.append(InterfaceData3D(0, 1, Direction3D.Y))
interfaces.append(InterfaceData3D(1, 2, Direction3D.X))

# Initialize & position mics.
mics = []
mics.append(Mic(
    0, [
        int(part_data[0].dimensions[0] / 2), 
        int(part_data[0].dimensions[1] / 2), 
        int(part_data[0].dimensions[2] / 2)
    ], sim_param, "left"))

mics.append(Mic(
    1, [
        int(part_data[1].dimensions[0] / 2), 
        int(part_data[1].dimensions[1] / 2), 
        int(part_data[1].dimensions[2] / 2)
    ], sim_param, "right"))

mics.append(Mic(
    2, [
        int(part_data[2].dimensions[0] / 2), 
        int(part_data[2].dimensions[1] / 2), 
        int(part_data[2].dimensions[2] / 2)
    ], sim_param, "bottom"))

# Instantiation serializer for reading and writing simulation state data
serializer = Serializer(compress=compress_file)

# Instantiating and executing simulation (don't change this)
sim = ARDS(sim_param, part_data, 1, interfaces, mics)
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
    serializer.dump(sim_param, part_data)

# Plotting waveform
if visualize:
    plotter = Plotter()
    plotter.set_data_from_simulation(sim_param, part_data)
    plotter.plot_3D()

