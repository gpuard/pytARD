from common.parameters import SimulationParameters as SIMP
from common.microphone import Microphone as Mic

from pytARD_1D.ard import ARDSimulator as ARDS
from pytARD_1D.partition_data import PartitionData as PARTD
from pytARD_1D.interface import InterfaceData1D

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
control_mic_before = Mic(0, int(c / 2), sim_param, "before_control")
control_mic_after = Mic(0, int(3 * (c / 2)), sim_param, "after_control")
control_mics = [control_mic_before, control_mic_after]

test_mic_before = Mic(0, int(c / 2), sim_param, "before_test")
test_mic_after = Mic(1, int(c / 2), sim_param, "after_test")
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

# Comparing waveforms
def diff(filename1, filename2, output_file_name):
    from scipy.io.wavfile import read, write
    fsl, left = read(filename1)
    _, right = read(filename2)

    left = np.array(left, dtype=np.float)
    right = np.array(right, dtype=np.float)

    diff = []

    for i in range(0, len(left)):
        diff.append(left[i] - right[i])

    diff = np.array(diff)

    write(output_file_name, fsl, diff.astype(np.float))

diff("after_control.wav", "after_test.wav", "after_diff.wav")
diff("before_control.wav", "before_test.wav", "before_diff.wav")

# shows the sound waves
def visualize_diff(paths, dB=False):
    from scipy.io.wavfile import read, write
    import matplotlib.pyplot as plt

    signals = []
    times = []

    for path in paths:
    # reading the audio file
        f_rate, x = read(path)
        
        # reads all the frames
        # -1 indicates all or max frames
        signal = np.array(x, dtype=np.float)
        signals.append(signal)

        time = np.linspace(
            0, # start
            len(signal) / f_rate,
            num = len(signal)
        )

        times.append(time)

    # using matplotlib to plot
    # creates a new figure
    fig = plt.figure()
    gs = fig.add_gridspec(len(paths), hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)

    for i in range(len(paths)):
        if dB:
            signal_to_plot = 20 * np.log10(np.abs(signals[i]))
        else:
            signal_to_plot = signals[i]
        axs[i].plot(times[i], signal_to_plot)
        axs[i].set_ylabel(paths[i] + "          ", rotation=0, labelpad=20)
        axs[i].grid()

    plt.xlabel("Time")
    plt.plot(time, signal)
    plt.show()

visualize_diff(["after_control.wav", "after_test.wav", "after_diff.wav","before_control.wav", "before_test.wav", "before_diff.wav"], dB=False)

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

