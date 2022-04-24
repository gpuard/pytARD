from pytARD_2D.ard import ARDSimulator as ARDS
from pytARD_2D.partition import AirPartition, PMLPartition, PMLType
from pytARD_2D.interface import InterfaceData2D, Direction2D

from common.parameters import SimulationParameters as SIMP
from common.impulse import Gaussian, Unit, WaveFile
from common.serializer import Serializer
from common.plotter import Plotter
from common.microphone import Microphone as Mic

import numpy as np

# For Debug
np.seterr(all='raise')

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
write_to_file = False
compress_file = True

# Compilation of room parameters into parameter class
sim_param = SIMP(
    upper_frequency_limit,
    duration,
    c=c,
    Fs=Fs,
    spatial_samples_per_wave_length=spatial_samples_per_wave_length,
    verbose=verbose,
    visualize=visualize
)

SCALE = 100  # Scale of room. Gets calculated by speed of sound divided by SCALE

# Define impulse that gets emitted into the room. Uncomment which kind of impulse you want
impulse_location = np.array([[int((c / SCALE) / 2)], [int((c / SCALE) / 2)]])
# impulse = Gaussian(sim_param, impulse_location, 10000)
#impulse = Unit(sim_param, impulse_location, 1, upper_frequency_limit-1)
impulse = WaveFile(sim_param, impulse_location, 'clap_8000.wav', 100)

# Paritions of the room. Can be 1..n. Add or remove rooms here.
air_partition = AirPartition(np.array([[int(c / SCALE)], [int(c / SCALE)]]), sim_param, impulse)
pml_partition = PMLPartition(np.array([[1.5], [int(c / SCALE)]]), sim_param, PMLType.LEFT)

# Compilation of all partitions into one part_data object. Add or remove rooms here. TODO change to obj.append()
part_data = [air_partition, pml_partition]

# Interfaces of the room. Interfaces connect the room together
interfaces = []
interfaces.append(InterfaceData2D(0, 1, Direction2D.X))

# Microphones (are optional)
mic1 = Mic(
    0, # Parition number
    # Position
    [int(part_data[0].dimensions[0] / 2), 
    int(part_data[0].dimensions[1] / 2)], 
    sim_param, 
    "left" # Name of resulting wave file
)

if auralize:
    # Compilation of all microphones into one mics object. Add or remove mics here. TODO change to obj.append()
    mics = [mic1]

    # Instantiation serializer for reading and writing simulation state data
    serializer = Serializer(compress=compress_file)

# Instantiating and executing simulation
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
        serializer.dump(sim_param, part_data)

# Plotting waveform
if visualize:
    plotter = Plotter()
    plotter.set_data_from_simulation(sim_param, part_data)
    plotter.plot_2D()
