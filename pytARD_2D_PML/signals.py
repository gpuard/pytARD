# -*- coding: utf-8 -*-
from simulation_parameters import SimulationParameters
import numpy as np

class Signal():
    def __init__(self, simulation_parameters, signal_location):
        self.signal_loc = signal_location
        (self.y, self.x) = signal_location
        self.grid_loc_y = int(self.y/simulation_parameters.dy)
        self.grid_loc_x = int(self.x/simulation_parameters.dx)
        
        self.dt = simulation_parameters.dt
        self.time_steps = np.array(simulation_parameters.time_steps)
        self.time = self.time_steps * simulation_parameters.dt
        
        self.src = None # conatains signal amplitudes over time

    def plot(self):
        import matplotlib.pyplot as plt
        import scipy.fftpack as fft
        fig, ax = plt.subplots(1,2)
        ax[0].plot(self.time, self.src)
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Amplitude')
        
        spec = fft.fft(self.src) # source time function in frequency domain
        freq = fft.fftfreq(spec.size, d = self.dt ) # time domain to frequency domain
        ax[1].plot(np.abs(freq),np.abs(spec))
        # ax[1].plot(freq,spec)
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_ylabel('Amplitude')

class GaussianFirstDerivative(Signal):
    
    def __init__(self, 
                 simulation_parameters, 
                 signal_location, # in meter
                 dominant_frequency, # in Hz
                 time_offset): # in sec
    
        Signal.__init__(self, simulation_parameters, signal_location)
        
        self.f0 = dominant_frequency
        self.t0 = time_offset
        
    def generate(self):
        self.src = -8. * (self.time - self.t0) * self.f0 * (np.exp(-1.0 * (4*self.f0) ** 2 * (self.time - self.t0) ** 2))
        return self.src

if __name__ == '__main__':
    
    sim_params = SimulationParameters(  wave_speed = 20, # in meter per second
                                        max_simulation_frequency = 200, # in herz
                                        samples_per_wave_length = 7, # samples per meter
                                        simulation_time = 2, # in seconds
                                        time_sampling_rate = 200, # in samples per second
                                        verbose = True, 
                                        visualize = True)
    
    sig = GaussianFirstDerivative(sim_params, 
                            signal_location = (4,4), 
                            dominant_frequency = 25,
                            time_offset = 1)
    
    src = sig.generate()
    sig.plot()    