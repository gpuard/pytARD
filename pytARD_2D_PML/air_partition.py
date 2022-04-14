# -*- coding: utf-8 -*-
from partition import Partition
from signals import GaussianFirstDerivative
import numpy as np
from scipy.fft import idctn, dctn

class AirPartition:
    
    def __init__(
        self,
        partition_dimensions,# in meter 
        simulation_parameters,
        signal = None):
        
        Partition.__init__(self, simulation_parameters, partition_dimensions)
        self.DCT_TYPE = simulation_parameters.DCT_TYPE
        
        self.signal = signal
        self.src = np.zeros((simulation_parameters.num_time_samples, self.grid_shape_y, self.grid_shape_x))

        # PRESSURE FIELDS
        self.p_spec_o = np.zeros(self.grid_shape)      # in previous time step
        self.p_spec = np.zeros(self.grid_shape)        # in current time step
        self.p_spec_n = np.zeros(self.grid_shape)      # in next time step
        
        self.pressure_fields = list()
        self.visualize = simulation_parameters.visualize

    def preprocessing(self):
        # Inital condition aka. @ time step n = 0
        p_o = np.zeros(self.grid_shape)
        p = np.zeros(self.grid_shape)
        
        p_spec = np.zeros(self.grid_shape)
        p_spec_n = np.zeros(self.grid_shape)
        
        self.pressure_fields.append(p_o)
        self.pressure_fields.append(p)
        
        # Precompute signal sources
        if self.src is not None:
            self.precompute_signal()
           
        # self.pressure_field.append(np.zeros(self.grid_shape))
        # #precompute omega - field
        self.omegas = np.zeros(self.grid_shape)
        
        for iy in self.gr_indx_y:
            for ix in self.gr_indx_x:
                self.omegas[iy, ix] = self.wave_speed * np.pi * np.sqrt( (ix/ self.x) ** 2 + (iy / self.y) ** 2)
        # To avoid devision by zero (omegas[0, 0] is 0)
        self.omegas[0, 0] = np.finfo(np.float).eps
    
    def precompute_signal(self):
        self.src[:, self.signal.grid_loc_y, self.signal.grid_loc_x] = self.signal.generate()
        if self.visualize:
            self.signal.plot()
    
    def simulate(self, n):
        # @ time step n-1:              p_spec_o,
        # @ time step n:    f,  f_spec, p_spec,
        # @ time step n+1               p_spec_n
        
        # f_spec is local
        
        # INJECT SINGNAL SOURCE
        self.f += self.src[n].copy()
        
        f_spec = dctn(self.f, type=self.DCT_TYPE)

        tmp_term = 2 * f_spec / (self.omegas ** 2) * (1 - np.cos(self.omegas * self.dt))
        self.p_spec_n = 2 * self.p_spec * np.cos(self.omegas * self.dt) - self.p_spec_o + tmp_term
        
        self.p_n = idctn(self.p_spec_n, type=self.DCT_TYPE) 
        
        # RESET FORCING FIELD
        self.f = np.zeros(self.grid_shape)
        
        self.p_spec_o = self.p_spec
        self.p_spec = self.p_spec_n
        self.p = self.p_n
        
        self.pressure_fields.append(self.p.copy())
        
    def show_info(self):
        Partition.show_info(self)