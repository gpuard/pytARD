from pytARD_3D.ard import ARDSimulator3D
from pytARD_3D.partition import AirPartition3D, PMLPartition3D, DampingProfile, PMLType
from pytARD_3D.interface import InterfaceData3D, Direction3D

from common.parameters import SimulationParameters
from common.impulse import Gaussian, Unit, WaveFile
from common.serializer import Serializer
from common.plotter import Plotter
from common.microphone import Microphone as Mic

import matplotlib.pyplot as plt
import numpy as np

# Room parameters
duration = 1 # seconds
Fs = 8000 # sample rate
upper_frequency_limit = 250 # Hz
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
sim_param = SimulationParameters(
    upper_frequency_limit, 
    duration, 
    c=c, 
    Fs=Fs,
    spatial_samples_per_wave_length=spatial_samples_per_wave_length, 
    verbose=verbose,
    visualize=visualize
)

SCALE = 80 # Scale of room. Gets calculated by speed of sound divided by SCALE

# Define impulse location
impulse_location = np.array([
    [int((c / SCALE) / 2)], # X, width
    [int((c / SCALE) / 2)], # Y, depth
    [int((c / SCALE) / 2)]  # Z, height
])

# Define impulse location that gets emitted into the room
# impulse = Gaussian(sim_param, impulse_location, 1)
impulse = Unit(sim_param, impulse_location, 1, upper_frequency_limit - 1)
# impulse = WaveFile(sim_param, impulse_location, 'clap.wav', 100) # Uncomment for wave file injection

room_width = int(c / SCALE)

# Damping profile with according Zetta value (how much is absorbed)
dp = DampingProfile(room_width, c, 1e-3)

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

'''
partitions.append(PMLPartition3D(np.array([
    [1.5], # X, width
    [room_width], # Y, depth
    [room_width]  # Z, height
]), sim_param, PMLType.LEFT, dp))

partitions.append(PMLPartition3D(np.array([
    [1.5], # X, width
    [room_width], # Y, depth
    [room_width]  # Z, height
]), sim_param, PMLType.LEFT, dp))

partitions.append(PMLPartition3D(np.array([
    [room_width], # X, width
    [1.5], # Y, depth
    [room_width]  # Z, height
]), sim_param, PMLType.LEFT, dp))

partitions.append(PMLPartition3D(np.array([
    [room_width], # X, width
    [1.5], # Y, depth
    [room_width]  # Z, height
]), sim_param, PMLType.LEFT, dp))
'''


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
sim = ARDSimulator3D(sim_param, partitions, 1, interfaces, mics)
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
    plotter = Plotter()
    plotter.set_data_from_simulation(sim_param, partitions)
    plotter.plot_3D()

