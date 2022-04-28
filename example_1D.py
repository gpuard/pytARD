from pytARD_1D.ard import ARDSimulator
from pytARD_1D.partition import PartitionData
from pytARD_1D.interface import InterfaceData1D

from common.parameters import SimulationParameters
from common.impulse import Gaussian, Unit, WaveFile, ExperimentalUnit
from common.microphone import Microphone

import matplotlib.pyplot as plt
import numpy as np
from datetime import date, datetime

# Room parameters
src_pos = [0] # m
duration = 2 # seconds
Fs = 8000 # sample rate
upper_frequency_limit = 500 # Hz
c = 342 # m/s
spatial_samples_per_wave_length = 6

# Procedure parameters
auralize = True
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
impulse = ExperimentalUnit(sim_param, impulse_location, 1, cutoff_frequency=upper_frequency_limit)
#impulse = WaveFile(sim_param, impulse_location, 'clap_8000.wav', 1000)

partitions = []
partitions.append(PartitionData(np.array([c / 2]), sim_param, impulse))
partitions.append(PartitionData(np.array([c / 2]), sim_param))

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

# Plotting waveform
if visualize:
    room_dims = np.linspace(0., partitions[0].dimensions[0], len(partitions[0].pressure_field_results[0]))
    ytop = np.max(partitions[0].pressure_field_results)
    ybtm = np.min(partitions[0].pressure_field_results)

    plt.figure()
    for i in range(0, len(partitions[0].pressure_field_results), 50):
        plt.clf()
        plt.title(f"ARD 1D (t = {(sim_param.T * (i / sim_param.number_of_samples)):.4f}s)")
        plt.subplot(1, 2, 1)
        plt.plot(room_dims, partitions[0].pressure_field_results[i], 'r', linewidth=1)
        plt.ylim(top=ytop)
        plt.ylim(bottom=ybtm)
        plt.vlines(np.min(room_dims), ybtm, ytop, color='gray')
        plt.vlines(np.max(room_dims), ybtm, ytop, color='gray')
        plt.subplot(1, 2, 2)
        plt.plot(room_dims, partitions[1].pressure_field_results[i], 'b', linewidth=1)
        plt.xlabel("Position [m]")
        plt.ylabel("Displacement")
        plt.ylim(top=ytop)
        plt.ylim(bottom=ybtm)
        plt.vlines(np.min(room_dims), ybtm, ytop, color='gray')
        plt.vlines(np.max(room_dims), ybtm, ytop, color='gray')
        plt.grid()
        plt.pause(0.001)

    plot_step = 100

