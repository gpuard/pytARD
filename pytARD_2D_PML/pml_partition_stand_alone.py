# -*- coding: utf-8 -*-
from partition import Partition
from signals import GaussianFirstDerivative
import numpy as np

class PMLPartition(Partition):
    
    def __init__(   self,
                    partition_dimensions,# in meter 
                    simulation_parameters,
                    signal = None):
        # 'LEFT' stand for the boundry on the left of AIR Partion
        
        Partition.__init__(self, simulation_parameters, partition_dimensions)
        
        # TODO i'm forcing the pml to have the width
        self.grid_shape_y = 100
        self.grid_shape_x = 100
        self.grid_shape = (self.grid_shape_y, self.grid_shape_x)

        # PRESSURE FIELD
        self.po = np.zeros(self.grid_shape) # pressure field in previous time step
        self.p = np.zeros(self.grid_shape) # pressure field in current time step
        self.pn = np.zeros(self.grid_shape) # pressure field in next time step
        
        # AUXILARY FUNCTIONS
        self.phiX = np.zeros(self.grid_shape)
        self.phiXn = np.zeros(self.grid_shape)
        
        self.phiY = np.zeros(self.grid_shape)
        self.phiYn = np.zeros(self.grid_shape)
        
        # FORCING FIELD
        self.f = np.zeros(self.grid_shape) # current time step
        self.pressure_fields = list()
        #FORCED
        self.dx = 1
        self.dy = 1
        self.wave_speed = 1
        self.src = None
        signal = None
        if signal is not None:
            signal.f0 = 0.001
            signal.t0 = 0
            signal.grid_loc_y = 50
            signal.grid_loc_x = 50
            
            signal.dt = self.dt
            signal.time_steps = np.arange(simulation_parameters.num_time_samples)
            signal.time = signal.time_steps * self.dt
            self.src = np.zeros((simulation_parameters.num_time_samples,self.grid_shape_y,self.grid_shape_x))
            self.src[:, signal.grid_loc_y, signal.grid_loc_x] = signal.generate()
            
    def simulate(self, t):
        # print(t)
        width = 3
        k_max = 0
        # kx = 0
        # ky = 0
        # f = 0.005 #Hz
        # self.f[50,50] = 2 * np.sin((t-1)*2*np.pi*f)
        self.f[50,50] = 2000 * np.sin((t-1)*np.pi/20)
        # self.f += self.src[t]
        for x in range(self.grid_shape_x)[1:-1]:
            if x <= width:
                kx = k_max*(width-x)**2/width**2
            elif x >= self.grid_shape_x - width:
                kx = k_max*(width-self.grid_shape_x+x)**2/width**2
            else:
                kx = 0
                
            for y in range(self.grid_shape_y)[1:-1]:
                if y <= width:
                    ky = k_max*(width-y)**2/width**2
                elif y >= self.grid_shape_y-width:
                    ky = k_max*(width-self.grid_shape_y+y)**2/width**2
                else:
                    ky = 0
                kx = 0
                ky = 0    
                term1 = (kx + ky)*self.dt/2
                term2 = 2 * self.p[y,x] - (1 - term1) * self.po[y,x]
         
                dp2dx2 = (self.p[y+1,x]-2*self.p[y,x]+self.p[y-1,x])/(self.dx**2)#
                dp2dy2 = (self.p[y,x+1]-2*self.p[y,x]+self.p[y,x-1])/(self.dy**2)#
                
                D2p = dp2dx2 + dp2dy2

                dphiXdx = (self.phiX[y,x]+self.phiX[y,x-1]-self.phiX[y-1,x]-self.phiX[y-1,x-1])/(2*self.dx)
                dphiYdy = (self.phiY[y,x]+self.phiY[y-1,x]-self.phiY[y,x-1]-self.phiY[y-1,x-1])/(2*self.dy)
             
                Dphi = dphiXdx + dphiYdy;
                         
                term3 = self.wave_speed**2 * D2p + Dphi;
                term4 = kx*ky*self.p[y,x]
                self.pn[y,x] = (term2 + self.dt**2*(term3 - term4))/(1+term1) + self.f[y,x]                          
                           
                term5 = 1/self.dt + kx/2;
                term6 = 1/self.dt - kx/2;
                dpdx1 = (self.pn[y+1,x]+self.pn[y+1,x+1]-self.pn[y,x]-self.pn[y,x+1])/(2*self.dx);
                dpdx2 = (self.p[y+1,x]+self.p[y+1,x+1]-self.p[y,x]-self.p[y,x+1])/(2*self.dx);
                dpdx = (dpdx1 + dpdx2)/2;
                term7 = term6*self.phiX[y,x] + self.wave_speed**2*(ky-kx)*dpdx;
                self.phiXn[y,x] = term7/term5
    
                term8 = 1/self.dt + ky/2;
                term9 = 1/self.dt - ky/2;
                dpdy1 = (self.pn[y,x+1]+self.pn[y+1,x+1]-self.pn[y,x]-self.pn[y+1,x])/(2*self.dy);
                dpdy2 = (self.p[y,x+1]+self.p[y+1,x+1]-self.p[y,x]-self.p[y+1,x])/(2*self.dy);
                dpdy = (dpdy1 + dpdy2)/2;
                term10 = term9*self.phiY[y,x] + self.wave_speed**2*(kx-ky)*dpdy;
                self.phiYn[y,x] = term10/term8

        self.phiX = self.phiXn
        self.phiY = self.phiYn
    
        self.po = self.p
        self.p = self.pn
        
        self.f = np.zeros(self.grid_shape)
        
        # print(np.max(self.p))
        self.pressure_fields.append(self.p.copy())

if __name__ == "__main__":
    r = np.arange(5)
    width = 5
    kx = lambda i,k:  k * ( i / width - np.sin(2*np.pi * i / width)/(2*np.pi))
    import matplotlib.pyplot as plt
    plt.plot(kx(r,k=40))
    plt.plot(kx(r,k=1000))

# if we zero out kx and ky, then we should get the common accoustic wave destribution.
    