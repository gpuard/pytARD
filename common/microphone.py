import numpy as np
from scipy.io.wavfile import write

class Microphone():
    def __init__(self, partition_number, location, sim_param, name):   
        # TODO: Infer partition number and get location directly from partition number reference (maybe or so it goes lol)
        self.partition_number = partition_number
        self.location = location
        self.sim_param = sim_param
        self.signal = np.zeros(sim_param.number_of_samples)
        self.name = name + ".wav"
    
    def record(self, sample, index):
        self.signal[index] = sample

    def write_to_file(self, normalize_divisor=None, lowpass_frequency=None):
        # Normalize data
        if normalize_divisor:
            normalized_signal = self.signal / normalize_divisor
        else:
            normalized_signal = self.signal / np.max(self.signal)

        if lowpass_frequency:
            from scipy.signal import butter, sosfilt
            sos = butter(10, lowpass_frequency, 'lp', fs=self.sim_param.Fs, output='sos')
            normalized_signal = sosfilt(sos, normalized_signal)

        # Write to file
        write(self.name, self.sim_param.Fs, normalized_signal.astype(np.float))