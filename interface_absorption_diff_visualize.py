import numpy as np

def diff(filename1, filename2, output_file_name):
    from scipy.io.wavfile import read, write
    fsl, left = read(filename1)
    fsr, right = read(filename2)

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
def visualize(paths, dB=False):
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

visualize(["after_control.wav", "after_test.wav", "after_diff.wav","before_control.wav", "before_test.wav", "before_diff.wav"], dB=False)