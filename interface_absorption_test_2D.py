from pytARD_2D.ard import ARDSimulator as ARDS
from pytARD_2D.partition_data import PartitionData as PARTD
from pytARD_2D.interface import InterfaceData2D, Direction2D

from common.parameters import SimulationParameters as SIMP
from common.impulse import Gaussian, Unit, WaveFile
from common.serializer import Serializer
from common.plotter import Plotter
from common.microphone import Microphone as Mic

from wavdiff import wav_diff, visualize_diff 

import numpy as np

# Room parameters
Fs = 8000  # sample rate
upper_frequency_limit = Fs / 7  # Hz
c = 342  # m/s
duration = 0.2 #  seconds
spatial_samples_per_wave_length = 4

# Procedure parameters
verbose = True
visualize = False
write_to_file = True
compress_file = True

# Compilation of room parameters into parameter class
sim_params = SIMP(
    upper_frequency_limit,
    duration,
    c=c,
    Fs=Fs,
    spatial_samples_per_wave_length=spatial_samples_per_wave_length,
    normalization_constant=1,
    verbose=verbose,
    visualize=visualize
)

# Define impulse that gets emitted into the room. Uncomment which kind of impulse you want
impulse_location = np.array([[0.5], [0.5]])
impulse = Unit(sim_params, impulse_location, 1, upper_frequency_limit)
#impulse = WaveFile(sim_params, impulse_location, 'clap_8000.wav', 100)
#impulse = Gaussian(sim_params, impulse_location, 10000)

# Paritions of long room
control_room = PARTD(np.array([2, 1]), sim_params, impulse)

# Paritions of two concatenated small rooms
test_room_left = PARTD(np.array([1, 1]), sim_params, impulse)
test_room_right = PARTD(np.array([1, 1]), sim_params)

# Compilation of all partitions into one part_data object. Add or remove rooms here. TODO change to obj.append()
control_room = [control_room]
test_room = [test_room_left, test_room_right]

# Interfaces of the concatenated room.
interfaces = []
interfaces.append(InterfaceData2D(0, 1, Direction2D.Horizontal))

# Microphones
long_room_mic1 = Mic(
    0, # Parition number
    # Position
    [0.5, 0.5], 
    sim_params, 
    "before_control" # Name of resulting wave file
)
long_room_mic2 = Mic(
    0, # Parition number
    # Position
    [1.5, 0.5], 
    sim_params, 
    "after_control" # Name of resulting wave file
)

short_room_mic1 = Mic(
    0, 
    [0.5, 0.5], 
    sim_params, 
    "before_test"
)
short_room_mic2 = Mic(
    1,
    [0.5, 0.5], 
    sim_params, 
    "after_test"
)

# Compilation of all microphones into one mics object. Add or remove mics here. TODO change to obj.append()
control_mics = [long_room_mic1, long_room_mic2]
test_mics = [short_room_mic1, short_room_mic2]


# Instantiation serializer for reading and writing simulation state data
serializer = Serializer(compress=compress_file)
plotter = Plotter()

def write_and_plot(room):
    # Write partitions and state data to disk
    if write_to_file:
        if verbose:
            print("Writing state data to disk. Please wait...")
        serializer.dump(sim_params, room)

    # Plotting waveform
    plotter.set_data_from_simulation(sim_params, room)
    plotter.plot_2D()

# Instantiating and executing control simulation
control_sim = ARDS(sim_params, control_room, 1, mics=control_mics)
control_sim.preprocessing()
control_sim.simulation()

# write_and_plot(control_room)

# Instantiating and executing test simulation
test_sim = ARDS(sim_params, test_room, 1, interfaces, mics=test_mics)
test_sim.preprocessing()
test_sim.simulation()

# Find best peak to normalize mic signal and write mic signal to file

def find_best_peak(mics):
    peaks = []
    for i in range(len(mics)):
        peaks.append(np.max(mics[i].signal))
    return np.max(peaks)

both_mic_peaks = []
both_mic_peaks.append(find_best_peak(control_mics))
both_mic_peaks.append(find_best_peak(test_mics))
best_peak = np.max(both_mic_peaks)

def write_mic_files(mics, peak):
    for i in range(len(mics)):
        mics[i].write_to_file(peak, upper_frequency_limit)

write_mic_files(control_mics, best_peak)
write_mic_files(test_mics, best_peak)

# Call to plot concatenated room
# write_and_plot(test_room)

wav_diff("after_control.wav", "after_test.wav", "after_diff.wav")
wav_diff("before_control.wav", "before_test.wav", "before_diff.wav")
visualize_diff(["after_control.wav", "after_test.wav", "after_diff.wav","before_control.wav", "before_test.wav", "before_diff.wav"], dB=False)