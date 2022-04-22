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
        if len(self.grid_shape) == 2:
            self.src = np.zeros((simulation_parameters.num_time_samples, self.grid_shape_y, self.grid_shape_x))
        else:
            self.src = np.zeros((simulation_parameters.num_time_samples, self.grid_shape_x))
            
        # PRESSURE FIELDS
        self.p_spec_o = np.zeros(self.grid_shape)      # in previous time step
        self.p_spec = np.zeros(self.grid_shape)        # in current time step
        self.p_spec_n = np.zeros(self.grid_shape)      # in next time step
        
        self.pressure_fields = list()
        self.visualize = simulation_parameters.visualize

    def preprocess(self):
        # Inital condition aka. @ time step n = 0
        p_o = np.zeros(self.grid_shape)
        p = np.zeros(self.grid_shape)
        
        # p_spec = np.zeros(self.grid_shape)
        # p_spec_n = np.zeros(self.grid_shape)
        
        self.pressure_fields.append(p_o)
        self.pressure_fields.append(p)
        
        # Precompute signal sources
        if self.signal is not None:
            self.precompute_signal()
           
        # self.pressure_field.append(np.zeros(self.grid_shape))
        # #precompute omega - field
        self.omegas = np.zeros(self.grid_shape)
        
        if len(self.grid_shape) == 2:
            for iy in self.gr_indx_y:
                for ix in self.gr_indx_x:
                    self.omegas[iy, ix] = self.wave_speed * np.pi * np.sqrt((ix/ self.x) ** 2 + (iy / self.y) ** 2)
            self.omegas[0, 0] = np.finfo(np.float64).eps # to get rid of erroe msg
        else:
            for ix in self.gr_indx_x:
                self.omegas[ix] = self.wave_speed * np.pi * (ix/ self.x)
            self.omegas[0] = np.finfo(np.float64).eps # to get rid of erroe msg
                
        # To avoid devision by zero in update rule (omegas[0, 0] is 0) the correction calculation should be done.
    
    def precompute_signal(self):
        if len(self.grid_shape) == 2:
            self.src[:, self.signal.grid_loc_y, self.signal.grid_loc_x] = self.signal.generate()
        else:
            self.src[:, self.signal.grid_loc_x] = self.signal.generate()
            
        if self.visualize:
            self.signal.plot()
    
    def simulate(self, n):
        # @ time step n-1:              p_spec_o,
        # @ time step n:    f,  f_spec, p_spec,
        # @ time step n+1               p_spec_n
        
        # f_spec is local
        # TODO check this part src is 1d array ->
        # INJECT SINGNAL SOURCE
        self.f += self.src[n]
        
        f_spec = dctn(self.f, type=self.DCT_TYPE)
        # f_spec = dctn(self.f, type=self.DCT_TYPE, norm='ortho')

        tmp_term = 2 * f_spec / (self.omegas ** 2) * (1 - np.cos(self.omegas * self.dt))
        self.p_spec_n = 2 * self.p_spec * np.cos(self.omegas * self.dt) - self.p_spec_o + tmp_term
        
        # correction for lim (update rule) for omega -> 0:
        if len(self.grid_shape) == 2:
            self.p_spec_n[0,0] = 2 * self.p_spec[0, 0] - self.p_spec_o[0, 0] + f_spec[0,0] * self.dt ** 2
        else:
            self.p_spec_n[0] = 2 * self.p_spec[0] - self.p_spec_o[0] + f_spec[0] * self.dt ** 2
        
        self.p_n = idctn(self.p_spec_n, type=self.DCT_TYPE) 
        # self.p_n = idctn(self.p_spec_n, type=self.DCT_TYPE, norm='ortho') 
        
        # RESET FORCING FIELD
        # TODO: sure? here?
        self.f = np.zeros(self.grid_shape)
        
        self.p_spec_o = self.p_spec
        self.p_spec = self.p_spec_n
        self.p = self.p_n
        # print(np.max(self.p))
        self.pressure_fields.append(self.p.copy())
        
    def show_info(self):
        Partition.show_info(self)