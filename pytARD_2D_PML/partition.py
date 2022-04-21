# -*- coding: utf-8 -*-
import numpy as np

class Partition():
    # convention  x - horizontal(second index), y - vertical(first index)
    # [f,p] o-previous timestep, n-nexttimestep
    # caps letters - spectral domain
    def __init__(self, simulation_parameters, partition_dimensions):
        
        
        self.wave_speed = simulation_parameters.wave_speed

        
        # Time spacing
        self.dt = simulation_parameters.dt
        # self.dimensions = np.array(partition_dimensions)
        self.dimensions = partition_dimensions
        if len(partition_dimensions) == 2:
            (self.y, self.x) = self.dimensions
        else:
            self.x, = self.dimensions
        
        # Grid Spacing
        self.dy = simulation_parameters.dy
        self.dx = simulation_parameters.dx

        # Grid
        if len(partition_dimensions) == 2:
            self.grid_y = np.arange(0, self.y, self.dy)
            self.grid_shape_y = len(self.grid_y)
            self.gr_indx_y = np.arange(self.grid_shape_y)

        self.grid_x = np.arange(0, self.x, self.dx)
        self.grid_shape_x = len(self.grid_x)
        self.gr_indx_x = np.arange(self.grid_shape_x)
        
        if len(partition_dimensions) == 2:
            self.grid_shape = (self.grid_shape_y, self.grid_shape_x)
        else:
            self.grid_shape = (self.grid_shape_x,)
            
        self.debug = simulation_parameters.debug
        
        if simulation_parameters.verbose:
            self.show_info()
            
        # TODO add type selection section
        # self.dtype = simulation_parameters.dtype
        
        # FORCING FIELD
        self.f = np.zeros(self.grid_shape) # in current time step
        self.p = np.zeros(self.grid_shape)     
        
    def preprocessing(self):
        pass
    
    def simulate(self, t):
         pass

    def show_info(self):
        if len(self.grid_shape) == 2:
            print(f"Partition dimensions y = {self.y} m, x = {self.x} m | Grid Shape: {self.grid_shape}")
        else:
            print(f"Partition dimensions x = {self.x} m | Grid Shape: {self.grid_shape}")
