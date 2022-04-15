from pytARD_2D.pml_partition_2D import PMLPartition2D
from pytARD_2D.ard import ARDSimulator as ARDS
from pytARD_2D.partition_data import PartitionData as PARTD
from pytARD_2D.interface import InterfaceData2D, Direction

from common.parameters import SimulationParameters as SIMP
from common.impulse import Gaussian, Unit, WaveFile
from common.serializer import Serializer
from common.plotter import Plotter
from common.microphone import Microphone as Mic

import numpy as np

# Room parameters
duration = .5  #  seconds
Fs = 8000  # sample rate
upper_frequency_limit = Fs/6  # Hz
c = 342  # m/s
spatial_samples_per_wave_length = 6

# Procedure parameters
verbose = True
visualize = False
write_to_file = True
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

SCALE = 50  # Scale of room. Gets calculated by speed of sound divided by SCALE

# Define impulse that gets emitted into the room. Uncomment which kind of impulse you want
impulse_location = np.array([[int((c / SCALE) / 2)], [int((c / SCALE) / 2)]])
# impulse = Gaussian(sim_params, impulse_location, 10000)
# impulse = Unit(sim_params, impulse_location, 1)
impulse = WaveFile(sim_param, impulse_location, 'clap_8000.wav', 100)

# Paritions of the room. Can be 1..n. Add or remove rooms here.
air_dimensions = np.array([[int(c / SCALE)], [int(c / SCALE)]])
pml_dimensions = np.array([[.3], [int(c / SCALE)]])
air_partition = PARTD(air_dimensions, sim_param, impulse)
pml_partition = PMLPartition2D(pml_dimensions, sim_param)

# Compilation of all partitions into one part_data object. Add or remove rooms here. TODO change to obj.append()
air_partitions = [air_partition]
pml_partitions = [pml_partition]

# Interfaces of the room. Interfaces connect the room together
interfaces = []
#interfaces.append(InterfaceData2D(0, 1, Direction.Horizontal))

# Microphones (are optional)
mic1 = Mic(
    0, # Parition number
    # Position
    [int(air_partitions[0].dimensions[0] / 2), 
    int(air_partitions[0].dimensions[1] / 2)], 
    sim_param, 
    "left" # Name of resulting wave file
)

# Compilation of all microphones into one mics object. Add or remove mics here. TODO change to obj.append()
mics = [mic1]

# Instantiation serializer for reading and writing simulation state data
serializer = Serializer(compress=compress_file)

# Instantiating and executing simulation
sim = ARDS(sim_param, air_partitions, pml_partitions, normalization_factor=1, interface_data=[], mics=mics)
sim.preprocessing()
sim.simulation()
pml_partition.simulate()

# Write partitions and state data to disk
if write_to_file:
    if verbose:
        print("Writing state data to disk. Please wait...")
    serializer.dump(sim_param, air_partitions)

# Plotting waveform
plotter = Plotter()
plotter.set_data_from_simulation(sim_param, air_partitions)
plotter.plot_2D()


