from pytARD_2D.ard import ARDSimulator as ARDS
from pytARD_2D.partition_data import PartitionData as PARTD
from pytARD_2D.interface import InterfaceData2D, Direction2D

from common.parameters import SimulationParameters as SIMP
from common.impulse import Gaussian, Unit, WaveFile
from common.serializer import Serializer
from common.plotter import Plotter
from common.microphone import Microphone as Mic

import numpy as np

# Room parameters
duration = 1  #  seconds
Fs = 10000  # sample rate
upper_frequency_limit = Fs/20  # Hz
c = 342  # m/s
spatial_samples_per_wave_length = 4

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

SCALE = 60  # Scale of room. Gets calculated by speed of sound divided by SCALE

# Define impulse that gets emitted into the room. Uncomment which kind of impulse you want
impulse_location = np.array([[int((c / SCALE) / 2)], [int((c / SCALE) / 2)]])
# impulse = Gaussian(sim_param, impulse_location, 10000)
impulse = Unit(sim_param, impulse_location, 1, upper_frequency_limit-1)
#impulse = WaveFile(sim_param, impulse_location, 'clap_8000.wav', 100)

# Paritions of the room. Can be 1..n. Add or remove rooms here.
partition_1 = PARTD(np.array([[int(c / SCALE)], [int(c / SCALE)]]), sim_param, impulse)
partition_2 = PARTD(np.array([[int(c / SCALE)], [int(c / SCALE)]]), sim_param)
partition_3 = PARTD(np.array([[int(c / SCALE)], [int(c / SCALE)]]), sim_param)

# Compilation of all partitions into one part_data object. Add or remove rooms here. TODO change to obj.append()
part_data = [partition_1, partition_2, partition_3]

# Interfaces of the room. Interfaces connect the room together
interfaces = []
interfaces.append(InterfaceData2D(0, 1, Direction2D.X))
interfaces.append(InterfaceData2D(1, 2, Direction2D.Y))

# Microphones (are optional)
mic1 = Mic(
    0, # Parition number
    # Position
    [int(part_data[0].dimensions[0] / 2), 
    int(part_data[0].dimensions[1] / 2)], 
    sim_param, 
    "left" # Name of resulting wave file
)
mic2 = Mic(
    1, 
    [int(part_data[1].dimensions[0] / 2), 
    int(part_data[1].dimensions[1] / 2)], 
    sim_param, 
    "right"
)
mic3 = Mic(
    2, 
    [int(part_data[2].dimensions[0] / 2), 
    int(part_data[2].dimensions[1] / 2)], 
    sim_param, 
    "bottom"
)

if auralize:
    # Compilation of all microphones into one mics object. Add or remove mics here. TODO change to obj.append()
    mics = [mic1, mic2, mic3]

    # Instantiation serializer for reading and writing simulation state data
    serializer = Serializer(compress=compress_file)

# Instantiating and executing simulation
sim = ARDS(sim_param, part_data, 1, interfaces, mics)
sim.preprocessing()
sim.simulation()

# Write partitions and state data to disk
if write_to_file:
    if verbose:
        print("Writing state data to disk. Please wait...")
    serializer.dump(sim_param, part_data)

# Plotting waveform
if visualize:
    plotter = Plotter()
    plotter.set_data_from_simulation(sim_param, part_data)
    plotter.plot_2D()
