import numpy as np
from scipy.io.wavfile import write

from common.parameters import SimulationParameters

class Microphone():
    '''
    Virtual microphone. Can be placed anywhere in the domain, and can create IR (impulse response) files.
    '''

    def __init__(self, partition_number: int, location: np.ndarray, sim_param: SimulationParameters, name: str):   
        '''
        Creates and places a virtual microphone inside a partition, depending on the partition number.

        Parameters
        ----------
        partition_number : int
            Number of partition, in which the microphone will be placed.
        location : np.ndarray
            Location of microphone. Relative to chosen partition.
        sim_param : SimulationParameters
            Instance of simulation parameter class.
        name : str
            Name of resulting .WAV file.
        '''
        self.partition_number = partition_number
        self.location = location
        self.sim_param = sim_param
        self.signal = np.zeros(sim_param.number_of_samples)
        self.name = name + ".wav"
    
    def record(self, sample: float, index: int):
        '''
        Records the signal sample at given index.

        Parameters
        ----------
        sample : float
            Sampled sound.
        index : int
            Index (time slice) of the sample.
        '''
        self.signal[index] = sample

    def write_to_file(self, normalize_divisor: float=None, lowpass_frequency: float=None):
        '''
        Writes all the collected sound data to a file.

        Parameters
        ----------
        normalize_divisor : float
            Normalization divisor, which helps in getting a unified amplitude before writing the file.
            Makes sure the microphone signal isn't too loud or too quiet.
        lowpass_frequency : float
            Optional lowpass frequency. If desired, a Butterworth lowpass filter is applied before writing the file.
        '''
        # Normalize data
        if normalize_divisor:
            normalized_signal = self.signal / normalize_divisor
        else:
            normalized_signal = self.signal / np.max(self.signal)

        # Butterworth lowpass filtering. Optional.
        if lowpass_frequency:
            from scipy.signal import butter, sosfilt
            sos = butter(10, lowpass_frequency, 'lp', fs=self.sim_param.Fs, output='sos')
            normalized_signal = sosfilt(sos, normalized_signal)

        # Write to file
        write(self.name, self.sim_param.Fs, normalized_signal.astype(np.float))
        