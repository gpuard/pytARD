# -*- coding: utf-8 -*-
from partition import Partition
import numpy as np
from enum import Enum
from fd_coefficients import FD

class PMLType(Enum):
    LEFT = 1
    TOP = 2
    RIGHT = 3
    BOTTOM = 4
        
class PMLPartition(Partition):
    
    def __init__(   self,
                    partition_dimensions,# in meter 
                    simulation_parameters,
                    parent_part,
                    pml_type = PMLType.RIGHT):
        # 'LEFT' stand for the boundry on the left of AIR Partion
        
        Partition.__init__(self, simulation_parameters, partition_dimensions)
        self.PMLType = pml_type
        
        # TODO i'm forcing the pml to have the width
        (self.y, self.x) = partition_dimensions
        self.gridgrid_shape_y = parent_part.grid_shape_y
        self.grid_shape_x = 5
        self.grid_shape = (self.grid_shape_y, self.grid_shape_x)
        # self.coefs_central_d2_6 = np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0])/180.0
        # self.coefs_central_d1_4 = np.array([1.0, -8.0, 0.0, 8.0, -1.0])/12.0
        
        # PRESSURE FIELD
        self.po = np.zeros(self.grid_shape) # pressure field in previous time step
        self.p = np.zeros(self.grid_shape) # pressure field in current time step
        self.pn = np.zeros(self.grid_shape) # pressure field in next time step
        
        # AUXILARY FUNCTIONS
        # self.phi_x = np.zeros(self.grid_shape) # current time step
        # self.phi_xn = np.zeros(self.grid_shape) # next time step
        
        # self.phi_y = np.zeros(self.grid_shape)
        # self.phi_yn = np.zeros(self.grid_shape)
        
        self.phiX = np.zeros(self.grid_shape)
        self.phiXn = np.zeros(self.grid_shape)
        
        self.phiY = np.zeros(self.grid_shape)
        self.phiYn = np.zeros(self.grid_shape)
        
        # FORCING FIELD
        self.f = np.zeros(self.grid_shape) # current time step
               
    # def simulate(self, t):
    #     # print(t) # 778
    #     k_max = 1
    #     for x in range(self.grid_shape_x)[2:-1]:
    #         kx = 0.0 # Damping profile
    #         ky = 0.0 # Damping profile
            
    #         if self.PMLType == PMLType.RIGHT: # RIGHT
    #             width = self.grid_shape_x
    #             # kx = k_max * (width- x)**2/width**2
    #             # kx = 1
    #             kx= 0
    #             # by 1 kx = 1 all wave should be absorbed, but by lower i waves should be reflected insede the partition.
    #         for y in range(self.grid_shape_y)[2:-1]:
    #             kx = 0
    #             ky = 0    
    #             term1 = (kx + ky)*self.dt/2
    #             term2 = 2 * self.p[y,x] - (1 - term1) * self.po[y,x]
         
    #             dp2dx2 = (self.p[y+1,x]-2*self.p[y,x]+self.p[y-1,x])/(self.dx**2)#
    #             dp2dy2 = (self.p[y,x+1]-2*self.p[y,x]+self.p[y,x-1])/(self.dy**2)#
                
    #             D2p = dp2dx2 + dp2dy2

    #             dphiXdx = (self.phiX[y,x]+self.phiX[y,x-1]-self.phiX[y-1,x]-self.phiX[y-1,x-1])/(2*self.dx)
    #             dphiYdy = (self.phiY[y,x]+self.phiY[y-1,x]-self.phiY[y,x-1]-self.phiY[y-1,x-1])/(2*self.dy)
             
    #             Dphi = dphiXdx + dphiYdy;
                         
    #             term3 = self.wave_speed**2 * D2p + Dphi;
    #             term4 = kx*ky*self.p[y,x]
    #             self.pn[y,x] = (term2 + self.dt**2*(term3 - term4))/(1+term1) + self.f[y,x]                          
                           
    #             term5 = 1/self.dt + kx/2;
    #             term6 = 1/self.dt - kx/2;
    #             dpdx1 = (self.pn[y+1,x]+self.pn[y+1,x+1]-self.pn[y,x]-self.pn[y,x+1])/(2*self.dx);
    #             dpdx2 = (self.p[y+1,x]+self.p[y+1,x+1]-self.p[y,x]-self.p[y,x+1])/(2*self.dx);
    #             dpdx = (dpdx1 + dpdx2)/2;
    #             term7 = term6*self.phiX[y,x] + self.wave_speed**2*(ky-kx)*dpdx;
    #             self.phiXn[y,x] = term7/term5
    
    #             term8 = 1/self.dt + ky/2;
    #             term9 = 1/self.dt - ky/2;
    #             dpdy1 = (self.pn[y,x+1]+self.pn[y+1,x+1]-self.pn[y,x]-self.pn[y+1,x])/(2*self.dy);
    #             dpdy2 = (self.p[y,x+1]+self.p[y+1,x+1]-self.p[y,x]-self.p[y+1,x])/(2*self.dy);
    #             dpdy = (dpdy1 + dpdy2)/2;
    #             term10 = term9*self.phiY[y,x] + self.wave_speed**2*(kx-ky)*dpdy;
    #             self.phiYn[y,x] = term10/term8

    #     self.phiX = self.phiXn
    #     self.phiY = self.phiYn
    
    #     self.po = self.p
    #     self.p = self.pn
        
    #     self.f = np.zeros(self.grid_shape)
        
    #     # print(np.max(self.p))
    #     if self.debug:
    #         self.pressure_fields.append(self.p)

    def simulate(self, t):
        for x in range(self.grid_shape_x)[3:-3]:
    
            # kx = (i < 20) ? (20 - i) * kxMin/10.0
            # ky = (i < 20) ? 0.05
            kx = 40 * (width - x)**2 / width**2
            ky = 0
            for y in range(self.grid_shape_y)[3:-3]:
          
            	coefs = np.array([ 2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0 ])
            	KPx = 0.0
            	KPy = 0.0
            	for k in range(7):
            		KPx += coefs[k] * self.p[y, x+k-3]
            		KPy += coefs[k] * self.p[y+k-3, x]
                
            	KPx /= 180.0
            	KPy /= 180.0
                
            	term1 = 2 * self.p[y, x]
            	term2 = -self.po[y, x]
            	term3 = self.wave_speed**2 * (KPx + KPy + self.f[y, x])
            	term4 = -(kx + ky) * (self.p[y, x] - self.po[y, x]) / self.dt
            	term5 = -kx * ky * self.p[y, x]
            	dphiXdx = 0.0
            	dphiYdy = 0.0
            	fourthCoefs = np.array([ 1.0, -8.0, 0.0, 8.0, -1.0 ])
            	for k in range(5):
            		dphiXdx += fourthCoefs[k] * self.phiX[y, x + k - 2]
            		dphiYdy += fourthCoefs[k] * self.phiY[y + k - 2, x]
                
            	dphiXdx /= 12.0
            	dphiYdy /= 12.0
                
            	term6 = dphiXdx + dphiYdy
            	self.pn[y,x] = term1 + term2 + self.dt**2 * (term3 + term4 + term5 + term6)
                
            	dpdx = 0.0
            	dpdy = 0.0
            	for k in range(5):
            		dpdx += fourthCoefs[k] * self.pn[y, x + k - 2]
            		dpdy += fourthCoefs[k] * self.pn[y + k - 2, x]
                
            	dpdx /= 12.0
            	dpdy /= 12.0
                
            	self.phiXn[y, x] = self.phiX[y, x] - self.dt * kx * self.phiX[y, x] + self.dt * self.wave_speed**2 * (ky - kx) * dpdx
            	self.phiYn[y, x] = self.phiY[y, x] - self.dt * ky * self.phiY[y, x] + self.dt * self.wave_speed**2 * (kx - ky) * dpdy
    
        self.phiX = self.phiXn
        self.phiY = self.phiYn

        self.po = self.p
        self.p = self.pn
   
        self.f = np.zeros(self.grid_shape)
        
        # print(np.max(self.p))
        if self.debug:
            self.pressure_fields.append(self.p)
            
class PMLPartition_1D(Partition):
    
    def __init__(   self,
                    partition_dimensions,# in meter 
                    simulation_parameters,
                    parent_part,
                    pml_type = PMLType.RIGHT):
        # 'LEFT' stand for the boundry on the left of AIR Partion
        
        Partition.__init__(self, simulation_parameters, partition_dimensions)
        self.PMLType = pml_type
        
        # TODO i'm forcing the pml to have the width
        self.x, = partition_dimensions
        self.gridgrid_shape_x = parent_part.grid_shape_x
        self.grid_shape_x = 5
        self.grid_shape = (self.grid_shape_x,)
        # self.coefs_central_d2_6 = np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0])/180.0
        # self.coefs_central_d1_4 = np.array([1.0, -8.0, 0.0, 8.0, -1.0])/12.0
        
        # PRESSURE FIELD
        self.po = np.zeros(self.grid_shape) # pressure field in previous time step
        self.p = np.zeros(self.grid_shape) # pressure field in current time step
        self.pn = np.zeros(self.grid_shape) # pressure field in next time step
        
        # AUXILARY FUNCTIONS
        self.phiX = np.zeros(self.grid_shape)
        self.phiXn = np.zeros(self.grid_shape)
        
        
        # FORCING FIELD
        self.f = np.zeros(self.grid_shape) # current time step
        self.pressure_fields = [np.zeros(self.grid_shape),np.zeros(self.grid_shape)]
               
    # def simulate(self, t):
    #     for x in range(self.grid_shape_x)[3:-3]:
    
    #         # kx = (i < 20) ? (20 - i) * kxMin/10.0
    #         # ky = (i < 20) ? 0.05
    #         width = 7
    #         kx = 40 * np.power((width - x) / width, 2)
         
    #         coefs = np.array([ 2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0 ])
    #         KPx = 0.0
    #         for k in range(7):
    #          	KPx += coefs[k] * self.p[x+k-3]
               
    #         KPx /= 180.0
               
    #         term1 = 2 * self.p[x]
    #         term2 = -self.po[x]
    #         term3 = self.wave_speed**2 * (KPx + self.f[x])
    #         term4 = -kx * (self.p[x] - self.po[x]) / self.dt
    #         dphiXdx = 0.0
    #         fourthCoefs = np.array([ 1.0, -8.0, 0.0, 8.0, -1.0 ])
    #         for k in range(5):
    #          	dphiXdx += fourthCoefs[k] * self.phiX[x + k - 2]
               
    #         dphiXdx /= 12.0
               
    #         term5 = dphiXdx
    #         self.pn[x] = term1 + term2 + self.dt**2 * (term3 + term4 + term5)
               
    #         dpdx = 0.0
    #         for k in range(5):
    #          	dpdx += fourthCoefs[k] * self.pn[x + k - 2]
               
    #         dpdx /= 12.0
               
    #         self.phiXn[x] = self.phiX[x] - self.dt * kx * self.phiX[x] + self.dt * self.wave_speed**2 * (kx) * dpdx
    
    #     self.phiX = self.phiXn

    #     self.po = self.p
    #     self.p = self.pn
   
    #     self.f = np.zeros(self.grid_shape)
        
    #     # print(np.max(self.p))
    #     # if self.debug:
    #     #     self.pressure_fields.append(self.p)
    #     self.pressure_fields.append(self.p)
            
    def simulate(self, t):
        for x in range(self.grid_shape_x)[1:-1]:
            # TODO may be init value should be considered precisely.
            # kx = (i < 20) ? (20 - i) * kxMin/10.0
            # ky = (i < 20) ? 0.05
            width = 5
            kx = 40 * np.power((width - x) / width, 2)
            # kx = 0
            # kx = 1
            
            # UPDATE RULE (p field @time n+1)
            term1 = kx * self.dt/2
            term2 = 2 * self.p[x] - (1 - term1) * self.po[x]
     
            # More accurate stencil may be used
            dpdx2 = (self.p[x+1]-2*self.p[x]+self.p[x-1])/(self.dx**2)
            
            # TODO not sure about the phiX 
            # dphidx = (self.phiX[x]+self.phiX[x-1]-self.phiX[x-1]-self.phiX[x-1])/(2*self.dx)
            dphidx = (self.phiX[x+1]-self.phiX[x-1])/(2*self.dx)
                     
            term3 = self.wave_speed**2 * dpdx2 + dphidx
            # self.pn[x] = (term2 + self.dt**2 * term3)/(1+term1) + self.f[x]  
            self.pn[x] = (term2 + self.dt**2 * term3)/(1+term1) + self.dt**2 * self.f[x]/(1+term1) # TODO check if the fix is correct
            
            # UPDATE RULE (phi)
            term4 = 1/self.dt + kx/2;
            term5 = 1/self.dt - kx/2;
            # More accurate stencil may be used
            dpdy = (self.p[x+1]-self.p[x-1])/(2*self.dx);
            term6 = term5 * self.phiX[x] + self.wave_speed**2 * kx * dpdy;
            self.phiXn[x] = term6/term4
            
        self.phiX = self.phiXn

        self.po = self.p
        self.p = self.pn
   
        self.f = np.zeros(self.grid_shape)
        
        # print(np.max(self.p))
        # if self.debug:
        #     self.pressure_fields.append(self.p)
        self.pressure_fields.append(self.p)

if __name__ == "__main__":
    r = np.arange(5)
    width = 5
    kx = lambda i,k:  k * ( i / width - np.sin(2*np.pi * i / width)/(2*np.pi))
    import matplotlib.pyplot as plt
    plt.plot(kx(r,k=40))
    plt.plot(kx(r,k=1000))

# if we zero out kx and ky, then we should get the common accoustic wave destribution.
    