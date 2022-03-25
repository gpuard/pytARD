import numpy as np
from scipy.io.wavfile import read


class Impulse():
    '''
    Abstract impulse class. Has no implementation
    '''
    def __init__(self):
        '''
        This method is deliberately not implemented
        '''
        pass

    def get(self):
        '''
        This method is deliberately not implemented
        '''
        raise NotImplementedError("This method is deliberately not implemented")


class Gaussian(Impulse):
    def __init__(self, sim_param, amplitude):
        self.sim_param = sim_param
        self.time_sample_indices = np.arange(
            0, self.sim_param.number_of_samples, 1)
        self.amplitude = amplitude

    @staticmethod
    def create_gaussian_impulse(x, mu, sigma):
        '''
        Generate gaussian impulse
        Parameters
        ----------
        x : int
            Current signal sample
        mu : float
            muTorrent i guess
        sigma : float
            Only sigma males understand this variable kek
        Returns
        -------
        float
            Gaussian impulse at signal sample (t(x))
        '''
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

    def get(self):
        return self.amplitude * Gaussian.create_gaussian_impulse(
            self.time_sample_indices, 2.5 * 4, 80) - self.amplitude * Gaussian.create_gaussian_impulse(self.time_sample_indices, 80 * 4 * 2, 80)


class WaveFile(Impulse):
    def __init__(self, sim_param, path_to_file, amplitude):
        self.sim_param = sim_param
        self.amplitude = amplitude

        (_, self.wav) = read(path_to_file)

    def get(self):
        return self.amplitude * self.wav[0:self.sim_param.number_of_samples]
