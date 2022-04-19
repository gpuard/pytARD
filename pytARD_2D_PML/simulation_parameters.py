# -*- coding: utf-8 -*-
import numpy as np

class SimulationParameters:
    def __init__(
        self,
        wave_speed = 343, # in meter per second
        max_simulation_frequency = 10, # in herz
        samples_per_wave_length = 4, # samples per meter
        simulation_time = 1, # in seconds
        time_sampling_rate = 8000, # in samples per second
        verbose = False,
        visualize = False,
        debug = False):
    
        self.Nt =[2,10][1] # 'Nyquist number'10 is the best option
        self.DCT_TYPE = 2
        
        dtypes = [np.int8, np.float16, np.float32, np.float64]
        self.dtype = dtypes[3]
        
        self.wave_speed = wave_speed
        
        self.max_simulation_frequency = max_simulation_frequency
        self.min_wave_length = wave_speed / max_simulation_frequency
        self.samples_per_wave_length = samples_per_wave_length
        
        self.simulation_time = simulation_time
        self.time_sampling_rate = time_sampling_rate
        
        # (Nyquist) Check, whether sampling rate high is enough 
        # to support maximal (wanted) wave frequency in room.
        if not self.time_sampling_rate_is_nyquist_conform ():
            self.time_sampling_rate = (self.Nt * self.max_simulation_frequency)
            print(f'FORCED (Nyquist): time_sampling_rate = {self.time_sampling_rate}')
        
        self.dt = 1 / self.time_sampling_rate
            
        self.num_time_samples = int(simulation_time * self.time_sampling_rate)
        
        # TODO is it good (lazy vs numpy)?
        self.time_steps = range(self.num_time_samples)

        # Checking if spacial sampling rate is conform with CFL condintion
        # (CFL) Check, if time stepping good enough
        dh = self.min_wave_length / self.samples_per_wave_length
        if not self.is_cfl_conform():
            # dh = 2 * self.wave_speed * self.dt * np.sqrt(3) # least value
            # self.samples_per_wave_length = self.min_wave_length / dh
            dh = 2 * self.wave_speed * self.dt * np.sqrt(3) 
            self.samples_per_wave_length = self.min_wave_length / dh
            print(f'FORCED (CFL): samples_per_wave_length = {self.samples_per_wave_length}')
        self.dx = dh
        self.dy = dh
        self.cfl = self.wave_speed*self.dt/self.dx
        self.verbose = verbose
        self.visualize = visualize
        self.debug = debug

        if verbose:
            self.show_info()

                    
    def show_info(self):
            print(f'Simulation time: {self.simulation_time}')
            print(f'Supported max-frequency: {self.max_simulation_frequency} Hz | min-wavelength: {self.min_wave_length}m')
            print(f'Time sampling frequency: {self.time_sampling_rate} samples per second ')
            print(f'Number time samples: {self.num_time_samples} samples | dt: {self.dt}s')
            print(f'Samples per wave length: {self.samples_per_wave_length}m')
            print(f'Grid spacing: {self.dx}m')
            print(f'CFL:{self.cfl}')
            # TODO take grid creation responsibility
            # print(f'Grid resolution: {self.grid_shape}')

        
    def time_sampling_rate_is_nyquist_conform (self):
        # Nyquist theorem  -> highest dt to avoid alliasing.
        return 2 * self.max_simulation_frequency < self.time_sampling_rate
    
    def is_cfl_conform(self):
        # we are in 2d -> 2 * on the right side
        return self.min_wave_length / self.samples_per_wave_length >= 2 * self.wave_speed * self.dt * np.sqrt(3)