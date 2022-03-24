import numpy as np
from scipy.io.wavfile import write

class Microphone():
    def __init__(self, location, sim_param, name):   
        self.location = location
        self.signal = np.zeros(sim_param.number_of_time_samples)
        self.name = name + ".wav"
    
    def record(self, sample, index):
        self.signal[index] = sample

    def write_to_file(self, Fs):
        # Normalize data
        normalized_signal = self.signal / np.max(self.signal)

        # Write to file
        write(self.name, Fs, normalized_signal.astype(np.float))