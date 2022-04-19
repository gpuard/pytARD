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
    #     for i in range(self.grid_shape_x):
    #         kx = 0.0 # Damping profile
    #         ky = 0.0 # Damping profile
            
    #         if self.PMLType == PMLType.RIGHT: # RIGHT
    #             # kx = (i - 20)*self.kxMax/10.0 if (i > 20) else 0.0
    #             # kx = (i - 20)*self.kxMax/10.0
    #             # ky = 0.05 if (i > 20) else 0.0
    #             width = self.grid_shape_x
    #             kx = 1000 * ( i / width - np.sin(2*np.pi * i / width)/(2*np.pi))
            
    #         # kx = (self.kxMax*i + self.kxMin*(width - 1 - i)) / width
    #         for j in range(self.grid_shape_y):
    #             # TODO fix
    #             # if True: #TOP
    #             #     ky = (20 - j)*self.kyMin / 10.0 if (j < 20) else 0.0
    #             #     kx = 0.05 if (j < 20) else 0.0
    #             # else: #BOT
    #             #     ky = (j - 20)*self.kyMax / 10.0 if (j > 20) else 0.0
    #             #     kx = 0.05 if (j > 20) else 0.0

    #             #ky = (self.kyMax*j + self.kyMin*(height - 1 - j)) / height
    #             coefs = [ 2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0 ] #/ 180.0
    #             KPx = 0.0
    #             KPy = 0.0
    #             for k in range(7): # number of points
    #                 if i + k - 3 < self.grid_shape[0]:
    #                     KPx += coefs[k] * self.p[i + k - 3, j]
    
    #                 if j + k - 3 < self.grid_shape[1]:
    #                     KPy += coefs[k] * self.p[i, j + k - 3]
                    
    #             KPx /= 180.0
    #             KPy /= 180.0

    #             term1 = 2 * self.p[i, j]
    #             term2 = -self.po[i, j]
    #             term3 = self.wave_speed**2 * (KPx + KPy + self.f[i, j])
    #             term4 = -(kx + ky)*(self.p[i, j] - self.po[i, j])/self.dt
    #             term5 = -kx*ky*self.p[i, j]
    #             dphidx = 0.0
    #             dphidy = 0.0
    #             fourthCoefs = [ 1.0, -8.0, 0.0, 8.0, -1.0 ] # first derivative; acc = 4, we need it for dphi/dt
    #             for k in range(5): # len(fourthCoefs)
    #                 if i + k - 2 < self.grid_shape[0]:
    #                     dphidx += fourthCoefs[k] * self.phi_x[i + k - 2, j]
    #                 if j + k - 2 < self.grid_shape[1]:
    #                     dphidy += fourthCoefs[k] * self.phi_y[i, j + k - 2]
   
    #             dphidx = (phi1(i,j)+phi1(i,j-1)-phi1(i-1,j)-phi1(i-1,j-1))/(2*dx);
    #             dphidy = (phi2(i,j)+phi2(i-1,j)-phi2(i,j-1)-phi2(i-1,j-1))/(2*dy);            
   
    
    #             # dphidx /= 12.0
    #             # dphidy /= 12.0

    #             term6 = dphidx + dphidy
    #             self.pn[i, j] = term1 + term2 + self.dt**2 * (term3 + term4 + term5 + term6)

    #             dudx = 0.0
    #             dudy = 0.0
    #             # for k in range(5):
    #             #     if i + k - 2 < self.grid_shape[1]:
    #             #         dudx += fourthCoefs[k] * self.pn[i + k - 2, j]
    #             #     if j + k - 2 < self.grid_shape[0]:                        
    #             #         dudy += fourthCoefs[k] * self.pn[i, j + k - 2]
 
    #             dudx = (self.pn[j,i+1]+self.pn[j+1,i+1]-self.pn[j,i]-self.pn[j+1,i])/(2*self.dx)
    #             #TODO chekck if stepe is ok
    #             dudy = (self.p[j,i+1]+self.p[j+1,i+1]-self.p[j,i]-self.p[j+1,i])/(2*self.dy);               
 
    #             # dudx /= 12.0
    #             # dudy /= 12.0

    #             self.phi_xn[i, j] = self.phi_x[i, j] - self.dt*kx*self.phi_x[i, j] + self.dt*self.wave_speed**2 * (ky - kx)*dudx
    #             self.phi_yn[i, j] = self.phi_y[i, j] - self.dt*ky*self.phi_y[i, j] + self.dt*self.wave_speed**2 * (kx - ky)*dudy

    #     self.phi_x = self.phi_xn
    #     self.phi_y = self.phi_yn

    #     self.po = self.p
    #     self.p = self.pn
        
    #     if self.debug:
    #         self.pressure_field_results.append(self.p)
        
    def simulate(self, t):
        # print(t) # 778
        k_max = 1
        for x in range(self.grid_shape_x)[2:-1]:
            kx = 0.0 # Damping profile
            ky = 0.0 # Damping profile
            
            if self.PMLType == PMLType.RIGHT: # RIGHT
                width = self.grid_shape_x
                # kx = k_max * (width- x)**2/width**2
                # kx = 1
                kx= 0
                # by 1 kx = 1 all wave should be absorbed, but by lower i waves should be reflected insede the partition.
            for y in range(self.grid_shape_y)[2:-1]:
                # ky = 1
                term1 = (kx + ky)*self.dt/2
                term2 = 2 * self.p[y,x] - (1 - term1)*self.po[y,x]
         
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
                dpdy1 = (self.pn[y+1,x]+self.pn[y+1,x+1]-self.pn[y,x]-self.pn[y,x+1])/(2*self.dy);
                dpdy2 = (self.p[y+1,x]+self.p[y+1,x+1]-self.p[y,x]-self.p[y,x+1])/(2*self.dy);
                dpdy = (dpdy1 + dpdy2)/2;
                term10 = term9*self.phiY[y,x] + self.wave_speed**2*(kx-ky)*dpdy;
                self.phiYn[y,x] = term10/term8

        self.phiX = self.phiXn
        self.phiY = self.phiYn

        self.po = self.p
        self.p = self.pn
        
        self.f = np.zeros(self.grid_shape)
        
        # print(np.max(self.p))
        if self.debug:
            self.pressure_fields.append(self.p)

if __name__ == "__main__":
    r = np.arange(5)
    width = 5
    kx = lambda i,k:  k * ( i / width - np.sin(2*np.pi * i / width)/(2*np.pi))
    import matplotlib.pyplot as plt
    plt.plot(kx(r,k=40))
    plt.plot(kx(r,k=1000))

# if we zero out kx and ky, then we should get the common accoustic wave destribution.
    