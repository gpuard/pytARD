from pytARD_1D.ard import ARDSimulator1D
from pytARD_1D.partition import AirPartition1D
from pytARD_1D.interface import InterfaceData1D

from common.parameters import SimulationParameters
from common.impulse import Gaussian, Unit, WaveFile
from common.microphone import Microphone
from common.plotter import Plotter

import numpy as np
from datetime import date, datetime

# Simulation parameters
duration = 2 # seconds
Fs = 8000 # sample rate
upper_frequency_limit = 500 # Hz
c = 342 # m/s
spatial_samples_per_wave_length = 6

# Procedure parameters
auralize = False
verbose = True
visualize = True

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

# Location of impulse that gets emitted into the room.
impulse_location = np.array([[int((c) / 4)]])

# Define impulse that gets emitted into the room. Uncomment which kind of impulse you want
#impulse = Gaussian(sim_param, impulse_location, 10000)
#impulse = Unit(sim_param, impulse_location, 1, cutoff_frequency=upper_frequency_limit)
impulse = WaveFile(sim_param, impulse_location, 'common/impulse_files/clap_8000.wav', 1000)

partitions = []
partitions.append(AirPartition1D(np.array([c / 2]), sim_param, impulse))
partitions.append(AirPartition1D(np.array([c / 2]), sim_param))

interfaces = []
interfaces.append(InterfaceData1D(0, 1))


# Microphones. Add and remove microphones here by copying or deleting mic objects. 
# Only gets used if the auralization option is enabled.
if auralize:
    mics = []
    mics.append(Microphone(
        0, # Parition number
        # Position
        [int(partitions[0].dimensions[0] / 2)], 
        sim_param, 
        f"pytARD_1D_{date.today()}_{datetime.now().time()}" # Name of resulting wave file
    ))


# Instantiating and executing simulation
sim = ARDSimulator1D(sim_param, partitions, 1, interfaces)
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

# Plotting waveform
if visualize:
    plotter = Plotter()
    plotter.set_data_from_simulation(sim_param, partitions)
    plotter.plot_1D()

