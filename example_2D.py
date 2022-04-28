from pytARD_2D.ard import ARDSimulator
from pytARD_2D.partition import AirPartition, PMLPartition, PMLType, DampingProfile
from pytARD_2D.interface import InterfaceData2D, Direction2D

from common.parameters import SimulationParameters
from common.impulse import Gaussian, Unit, WaveFile
from common.serializer import Serializer
from common.plotter import Plotter
from common.microphone import Microphone as Mic

import numpy as np
from datetime import date, datetime

# For Debug
#np.seterr(all='raise')

# Room parameters
duration = 1  #  seconds
Fs = 8000  # sample rate
upper_frequency_limit = Fs / 21  # Hz
c = 342  # m/s
spatial_samples_per_wave_length = 6 

# Procedure parameters
verbose = True
auralize = True
visualize = True
write_to_file = True
compress_file = True

# Compilation of room parameters into parameter class
sim_param = SimulationParameters(
    upper_frequency_limit,
    duration,
    c=c,
    Fs=Fs,
    spatial_samples_per_wave_length=spatial_samples_per_wave_length,
    verbose=verbose,
    visualize=visualize
)

# Scale of room. Gets calculated by speed of sound divided by SCALE
SCALE = 60  

# Location of impulse that gets emitted into the room.
impulse_location = np.array([[int((c / SCALE) / 2)], [int((c / SCALE) / 2)]])

# Define impulse that gets emitted into the room. Uncomment which kind of impulse you want
#impulse = Gaussian(sim_param, impulse_location, 10000)
impulse = Unit(sim_param, impulse_location, 1, 100)
#impulse = WaveFile(sim_param, impulse_location, 'clap_8000.wav', 1000)

# Compilation of all partitions into one part_data object. Add or remove rooms here.
partitions = []

# Width of room. Can be used for convenience, but can also be ignored if you want to define wall lengths yourself.
room_width = int(c / SCALE)

# Damping profile with according Zetta value (how much is absorbed)
dp = DampingProfile(room_width, c, 1e-3)

# Paritions of the room. Can be 1..n. Add or remove partitions here. 
# Also, provide impulse in the partition(s) of your choosing.
partitions.append(AirPartition(np.array([[room_width], [room_width]]), sim_param, impulse))
partitions.append(PMLPartition(np.array([[1.2], [room_width]]), sim_param, PMLType.LEFT, dp))
partitions.append(PMLPartition(np.array([[1.2], [room_width]]), sim_param, PMLType.RIGHT, dp))
partitions.append(PMLPartition(np.array([[room_width], [1.2]]), sim_param, PMLType.TOP, dp))
partitions.append(PMLPartition(np.array([[room_width], [1.2]]), sim_param, PMLType.BOTTOM, dp))

# Interfaces of the room. Interfaces connect the partitions together
interfaces = []
interfaces.append(InterfaceData2D(0, 1, Direction2D.X))
interfaces.append(InterfaceData2D(0, 2, Direction2D.X))
interfaces.append(InterfaceData2D(3, 0, Direction2D.Y))
interfaces.append(InterfaceData2D(4, 0, Direction2D.Y))

# Microphones. Add and remove microphones here by copying or deleting mic objects. 
# Only gets used if the auralization option is enabled.
if auralize:
    mics = []
    mics.append(Mic(
        0, # Parition number
        # Position
        [int(partitions[0].dimensions[0] / 2), 
        int(partitions[0].dimensions[1] / 2)], 
        sim_param, 
        f"pytARD_2D_{date.today()}_{datetime.now().time()}" # Name of resulting wave file
    ))

# Instantiation serializer for reading and writing simulation state data
serializer = Serializer(compress=compress_file)

# Instantiating and executing simulation
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
    serializer.dump(sim_param, partitions)

# Plotting waveform
if visualize:
    plotter = Plotter()
    plotter.set_data_from_simulation(sim_param, partitions)
    plotter.plot_2D()
