# -*- coding: utf-8 -*-
from partition import Partition
import numpy as np
from enum import Enum

class PMLType(Enum):
    LEFT = 1
    TOP = 2
    RIGHT = 3
    BOTTOM = 4
        
class PMLPartition(Partition):
    
    def __init__(   self,
                    partition_dimensions,# in meter 
                    simulation_parameters,
                    pml_type = PMLType.RIGHT):
        # 'LEFT' stand for the boundry on the left of AIR Partion
        
        Partition.__init__(self, simulation_parameters, partition_dimensions)
        
        self.PMLType = pml_type
        
        # TODO i'm forcing the pml to have the width
        self.grid_x = 5
        self.dimensions[1] = self.grid_x
        
        # self.coefs_central_d2_6 = np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0])/180.0
        # self.coefs_central_d1_4 = np.array([1.0, -8.0, 0.0, 8.0, -1.0])/12.0
        
        # PRESSURE FIELD
        self.po = np.zeros(self.grid_shape) # pressure field in previous time step
        self.p = np.zeros(self.grid_shape) # pressure field in current time step
        self.pn = np.zeros(self.grid_shape) # pressure field in next time step
        
        # AUXILARY FUNCTIONS
        self.phi_x = np.zeros(self.grid_shape) # current time step
        self.phi_xn = np.zeros(self.grid_shape) # next time step
        
        self.phi_y = np.zeros(self.grid_shape)
        self.phi_yn = np.zeros(self.grid_shape)
        
        # FORCING FIELD
        self.f = np.zeros(self.grid_shape) # current time step
        
        # for compability reasons here
        self.pressure_field = self.p
        self.new_forces = self.f
        self.pressure_field_results = [np.zeros(self.grid_shape)] # first time step is skipped
        
    def simulate(self, t):
        print (t)
        for i in range(self.grid_shape[0]):
            kx = 0.0 # Damping profile
            ky = 0.0 # Damping profile
            
            # TODO fix
            # if self.PMLType == PMLType.LEFT:
            #     kx = (20 - i)*self.kxMin/10.0 if (i < 20) else 0.0
            #     ky = 0.05 if (i < 20) else 0.0            
            if self.PMLType == PMLType.LEFT: # RIGHT
                # kx = (i - 20)*self.kxMax/10.0 if (i > 20) else 0.0
                # kx = (i - 20)*self.kxMax/10.0
                # ky = 0.05 if (i > 20) else 0.0
                width = 4
                kx = 40 * ( i / width - np.sin(2*np.pi * i / width)/(2*np.pi))
            
            # kx = (self.kxMax*i + self.kxMin*(width - 1 - i)) / width
            for j in range(self.grid_shape[1]):
                # TODO fix
                # if True: #TOP
                #     ky = (20 - j)*self.kyMin / 10.0 if (j < 20) else 0.0
                #     kx = 0.05 if (j < 20) else 0.0
                # else: #BOT
                #     ky = (j - 20)*self.kyMax / 10.0 if (j > 20) else 0.0
                #     kx = 0.05 if (j > 20) else 0.0

                #ky = (self.kyMax*j + self.kyMin*(height - 1 - j)) / height
                coefs = [ 2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0 ] #/ 180.0
                KPx = 0.0
                KPy = 0.0
                for k in range(7): # number of points
                    if i + k - 3 < self.grid_shape[0]:
                        KPx += coefs[k] * self.p[i + k - 3, j]
    
                    if j + k - 3 < self.grid_shape[1]:
                        KPy += coefs[k] * self.p[i, j + k - 3]
                    
                KPx /= 180.0
                KPy /= 180.0

                term1 = 2 * self.p[i, j]
                term2 = -self.po[i, j]
                term3 = self.speedOfSound**2 * (KPx + KPy + self.f[i, j])
                term4 = -(kx + ky)*(self.p[i, j] - self.po[i, j])/self.dt
                term5 = -kx*ky*self.p[i, j]
                dphidx = 0.0
                dphidy = 0.0
                fourthCoefs = [ 1.0, -8.0, 0.0, 8.0, -1.0 ] # first derivative; acc = 4, we need it for dphi/dt
                for k in range(5): # len(fourthCoefs)
                    if i + k - 2 < self.grid_shape[0]:
                        dphidx += fourthCoefs[k] * self.phi_x[i + k - 2, j]
                    if j + k - 2 < self.grid_shape[1]:
                        dphidy += fourthCoefs[k] * self.phi_y[i, j + k - 2]
                
                dphidx /= 12.0
                dphidy /= 12.0

                term6 = dphidx + dphidy
                self.pn[i, j] = term1 + term2 + self.dt**2 * (term3 + term4 + term5 + term6)

                dudx = 0.0
                dudy = 0.0
                for k in range(5):
                    if i + k - 2 < self.grid_shape[1]:
                        dudx += fourthCoefs[k] * self.pn[i + k - 2, j]
                    if j + k - 2 < self.grid_shape[0]:                        
                        dudy += fourthCoefs[k] * self.pn[i, j + k - 2]
  
                dudx /= 12.0
                dudy /= 12.0

                self.phi_xn[i, j] = self.phi_x[i, j] - self.dt*kx*self.phi_x[i, j] + self.dt*self.speedOfSound**2 * (ky - kx)*dudx
                self.phi_yn[i, j] = self.phi_y[i, j] - self.dt*ky*self.phi_y[i, j] + self.dt*self.speedOfSound**2 * (kx - ky)*dudy

        self.phi_x = self.phi_xn
        self.phi_y = self.phi_yn

        self.po = self.p
        self.p = self.pn
        
        if self.debug:
            self.pressure_field_results.append(self.p)
        
        # TODO zero out f m.b instead of doing this in ard


if __name__ == "__main__":
    r = np.arange(5)
    width = 5
    kx = lambda i: 40 * ( i / width - np.sin(2*np.pi * i / width)/(2*np.pi))
    import matplotlib.pyplot as plt
    plt.plot(kx(r))
    
    