from common.parameters import SimulationParameters as SIMP
from common.microphone import Microphone as Mic

from pytARD_1D.ard import ARDSimulator as ARDS
from pytARD_1D.partition import PartitionData as PARTD
from pytARD_1D.interface import InterfaceData1D

from wavdiff import wav_diff, visualize_diff 

import matplotlib.pyplot as plt
import numpy as np

# Room parameters
src_pos = [0] # m
duration = 2 # seconds
Fs = 8000 # sample rate
upper_frequency_limit = (Fs / 2) - 1 # Hz
c = 342 # m/s
spatial_samples_per_wave_length = 6

# Procedure parameters
auralize = False
verbose = True
visualize = True

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

# Define test and control rooms
control_partition = PARTD(np.array([c * 2]), sim_param)
control_room = [control_partition]

test_partition_1 = PARTD(np.array([c]), sim_param)
test_partition_2 = PARTD(np.array([c]), sim_param,do_impulse=False)
test_room = [test_partition_1, test_partition_2]

# Define Interfaces
interfaces = []
interfaces.append(InterfaceData1D(0, 1))

# Define and position mics

# Initialize & position mics. 
control_mic_before = Mic(0, int(c / 2), sim_param, "roomA_mic1")
control_mic_after = Mic(0, int(3 * (c / 2)), sim_param, "roomA_mic2")
control_mics = [control_mic_before, control_mic_after]

test_mic_before = Mic(0, int(c / 2), sim_param, "roomB_mic1")
test_mic_after = Mic(1, int(c / 2), sim_param, "roomB_mic2")
test_mics = [test_mic_before, test_mic_after]

# Instantiating and executing simulation
control_sim = ARDS(sim_param, control_room, .5, mics=control_mics)
control_sim.preprocessing()
control_sim.simulation()

test_sim = ARDS(sim_param, test_room, 1, interface_data=interfaces, mics=test_mics)
test_sim.preprocessing()
test_sim.simulation()

# Normalizing + writing recorded mic tracks
def find_best_peak(mics):
    peaks = []
    for i in range(len(mics)):
        peaks.append(np.max(mics[i].signal))
    return np.max(peaks)

both_mic_peaks = []
both_mic_peaks.append(find_best_peak(control_mics))
both_mic_peaks.append(find_best_peak(test_mics))
best_peak = np.max(both_mic_peaks)

def write_mic_files(mics, peak=1):
    for i in range(len(mics)):
        mics[i].write_to_file(peak, upper_frequency_limit)

write_mic_files(control_mics, best_peak)
write_mic_files(test_mics, best_peak)
wav_diff("roomA_mic1.wav", "roomB_mic1.wav", "mic1_diff.wav")
wav_diff("roomA_mic2.wav", "roomB_mic2.wav", "mic2_diff.wav")
visualize_diff(["roomA_mic1.wav", "roomB_mic1.wav", "mic1_diff.wav", "roomA_mic2.wav", "roomB_mic2.wav", "mic2_diff.wav"], dB=False)

'''
# Plotting waveform
if visualize:
    room_dims = np.linspace(0., test_partition_1.dimensions[0], len(test_partition_1.pressure_field_results[0]))
    ytop = np.max(test_partition_1.pressure_field_results)
    ybtm = np.min(test_partition_1.pressure_field_results)

    plt.figure()
    for i in range(0, len(test_partition_1.pressure_field_results), 50):
        plt.clf()
        plt.title(f"ARD 1D (t = {(sim_param.T * (i / sim_param.number_of_samples)):.4f}s)")
        plt.subplot(1, 2, 1)
        plt.plot(room_dims, test_partition_1.pressure_field_results[i], 'r', linewidth=1)
        plt.ylim(top=ytop)
        plt.ylim(bottom=ybtm)
        plt.vlines(np.min(room_dims), ybtm, ytop, color='gray')
        plt.vlines(np.max(room_dims), ybtm, ytop, color='gray')
        plt.subplot(1, 2, 2)
        plt.plot(room_dims, test_partition_2.pressure_field_results[i], 'b', linewidth=1)
        plt.xlabel("Position [m]")
        plt.ylabel("Displacement")
        plt.ylim(top=ytop)
        plt.ylim(bottom=ybtm)
        plt.vlines(np.min(room_dims), ybtm, ytop, color='gray')
        plt.vlines(np.max(room_dims), ybtm, ytop, color='gray')
        plt.grid()
        plt.pause(0.001)

    plot_step = 100
'''

