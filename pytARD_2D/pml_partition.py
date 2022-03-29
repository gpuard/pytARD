2# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 19:11:54 2022

@author: smailnik@students.zhaw.ch
"""

import numpy as np

class PMLPartition():
    '''
    Ported implematation from Matlab: https://github.com/thecodeboss/AcousticSimulator.git
    '''
    def __init__(   self,
                    # absolute_coordinates, # absolute coordinates in the room
                    partition_dimensions, 
                    simulation_parameters):

        self.pml_dimesions = np.array(partition_dimensions)
        self.grid_coordinates = None
        self.coefs_central_d2_6 = np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0])/180.0
        self.coefs_central_d1_4 = np.array([1.0, -8.0, 0.0, 8.0, -1.0])/12.0
             
        self.dx = simulation_parameters.dx
        self.dy = simulation_parameters.dy
        
        self.grid_x = np.arange(0,self.pml_dimesions[0],self.dx)
        self.grid_y = np.arange(0,self.pml_dimesions[1],self.dy)      
        self.c = simulation_parameters.c
        
        self.grid_shape = (len(self.grid_x),len(self.grid_y))
        self.p0 = np.zeros(self.grid_shape) # pressure field in previous time step
        self.p = np.zeros(self.grid_shape) # pressure field in current time step
        self.pn = np.zeros(self.grid_shape) # pressure field in next time step
        
        self.phi1 = np.zeros(self.grid_shape)
        self.phi1n = np.zeros(self.grid_shape) # next time step
        
        self.phi2 = np.zeros(self.grid_shape)
        self.phi2n = np.zeros(self.grid_shape) # next time step
        
        self.f = np.zeros(self.grid_shape) # forcing field in current time step
        self.simulation_parameters = simulation_parameters
        self.dt = simulation_parameters.delta_t

        self.pressure_field_results = list()

    def preprocessing(self):
        pass
        
    def damping_profile(self,grid_coordinates):
        #  assumtion wave propogation occures from left to right (x)
        (i,j) = grid_coordinates
        (width_x,width_y) = self.pml_dimesions  # thickness
        # kxMin = 0.2
        kMax = 20.0
        
        # kx =  kMax * (width_x-i)**2 / width_x**2
        kx = 0.05
        # kx = 20
        ky = kMax * (width_y-j)**2 / width_y**2 # this means lateral to wave propagation profile is changing
        # ky = 0.05
        return kx,ky
        
    def simulate(self, t):
        # for the given time step t do the calculation for all voxels in pml
        for x in  range(1,len(self.grid_x)-1):
            for y in  range(1,len(self.grid_y)-1):
                (kx,ky) = self.damping_profile((x,y))
                term1 = (kx+ky) * self.dt/2
                term2 = 2*self.p[x,y] - (1 - term1)*self.p0[x,y]
                # second spacial derivative in x- and y-directions
                dp2dx = (self.p[x+1,y] - 2 *self.p[x,y] + self.p[x-1,y] ) / (self.dx*self.dx)
                dp2dy = (self.p[x,y+1] - 2 *self.p[x,y] + self.p[x,y-1] ) / (self.dy*self.dy)
                
                D2p = dp2dx + dp2dy
                
                # first spacial derivative in x- and y-directions
                dphi1dx = (self.phi1[x,y]+self.phi1[x,y-1]-self.phi1[x-1,y]-self.phi1[x-1,y-1])/(2*self.dx)
                dphi2dy = (self.phi2[x,y]+self.phi2[x-1,y]-self.phi2[x,y-1]-self.phi2[x-1,y-1])/(2*self.dy)
                
                Dphi = dphi1dx + dphi2dy
                
                term3 = self.c**2 * D2p + Dphi
                
                term4 = kx*ky*self.p[x,y]
                self.pn[x,y] = (term2 + self.dt*self.dt*(term3 - term4))/(1+term1) + self.f[x,y]  # do need forcing field???
                
                term5 = 1/self.dt + kx/2
                term6 = 1/self.dt - kx/2
                dudx1 = (self.pn[x+1,y]+self.pn[x+1,y+1]-self.pn[x,y]-self.pn[x,y+1])/(2*self.dx)
                dudx2 = (self.p[x+1,y]+self.p[x+1,y+1]-self.p[x,y]-self.p[x,y+1])/(2*self.dx)
                dudx = (dudx1 + dudx2)/2
                term7 = term6*self.phi1[x,y] + self.c*self.c*(ky-kx)*dudx
                self.phi1n[x,y] = term7/term5
                
                fac8 = 1/self.dt + ky/2
                term8 = 1/self.dt - ky/2
                dudy1 = (self.pn[x,y+1]+self.pn[x+1,y+1]-self.pn[x,y]-self.pn[x+1,y]) / (2*self.dy)
                dudy2 = (self.p[x,y+1]+self.p[x+1,y+1]-self.p[x,y]-self.p[x+1,y]) / (2*self.dy)
                dudx = (dudy1 + dudy2)/2
                term10 = term8*self.phi2[x,y] + self.c*self.c*(kx-ky)*dudx
                self.phi2n[x,y] = term10/fac8  
        self.phi1 = self.phi1n;
        self.phi2 = self.phi2n;
        self.p0 = self.p;
        self.p = self.pn;
           
        self.pressure_field_results.append(self.p)
        
    def getAbsoluteCoordinates(self):
        pass