# -*- coding: utf-8 -*-
import numpy as np

class FTDT:
    FTDT_COEFFICIENTS = {1 : {  2 : np.array([-1/2, 0, 1/2]),
                                4 : np.array([1/12, -2/3, 0, 2/3, -1/12]),
                                6 : np.array([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60]),
                                8 : np.array([1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280 ])
                                },
                        2 : {   2 : np.array([1, -2, 1]),
                                4 : np.array([-1/12, 4/3, -5/2, 4/3, -1/12]),
                                6 : np.array([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]),
                                8 : np.array([-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560])
                                }
                        }
    @staticmethod
    def get_ftdt_coefficients(derivative, accuracy):
        return FTDT.FTDT_COEFFICIENTS[derivative][accuracy]
   
    @staticmethod
    def get_laplacian_matrix():
        coefs = FTDT.get_ftdt_coefficients(2,6)
        k1 = coefs[0]
        k2 = coefs[1]
        k3 = coefs[2]
        # K = np.array([0,0,-k1,k1,0,0,0,-k1,k2,-k2,k1,0,-k1,k2,-k3,k3,-k2,k1]).reshape(3,6)
        # K = (np.vstack([K,-np.flipud(K)]))
        K = np.array([k1,0,0,k2,k1,0,k3,k2,k1]).reshape(3,3)
        K = (np.vstack([K,-np.flipud(K)]))
        K = (np.hstack([-np.fliplr(K),K]))
        return K
# -0	-0	-0.0111111	0.0111111	0	0
# -0	-0.0111111	0.15	-0.15	0.0111111	0
# -0.0111111	0.15	-1.5	1.5	-0.15	0.0111111
# 0.0111111	-0.15	1.5	-1.5	0.15	-0.0111111
# 0	0.0111111	-0.15	0.15	-0.0111111	-0
# 0	0	0.0111111	-0.0111111	-0	-0
    
    def number_of_points():
        pass
        
class Interface():
      
    def __init__(self, partL, partR, simulation_parameters):
        self.wave_speed = simulation_parameters.wave_speed
        self.dx = simulation_parameters.dx
        self.partL = partL
        self.partR = partR
        self.K = FTDT().get_laplacian_matrix() 
        
    def preprocess(self):
        self.K =  np.power((self.wave_speed/self.dx), 2) * self.K
    
class X_Interface(Interface):
    # along y axis
    def __init__(self, partL, partR, simulation_parameters):
        Interface.__init__(self, partL, partR, simulation_parameters)
        num_points = 6 # sim parameters
        
        if partL.grid_shape == 2:
            self.grid_shape_y = min(partL.grid_shape_y,partR.grid_shape_y)
            
        self.grid_shape_x = num_points
        if partL.grid_shape == 2:
            self.grid_shape  = (self.grid_shape_y, self.grid_shape_x)
        else:
            self.grid_shape  = (self.grid_shape_x,)  
            
    def simulate(self):
        
        # cannot reference because of update rule
        p_left = self.partL.p[:,-3:]
        p_right = self.partR.p[:,:3]
        
        p = np.hstack((p_left, p_right))
        # Add everything together
        f = np.matmul(p,self.K)
            
        self.partL.f[:,-3:] += f[:,:3]
        self.partR.f[:,:3] += f[:,-3:]
        
class X_Interface_1D (Interface):
    # along y axis
    def __init__(self, partL, partR, simulation_parameters):
        Interface.__init__(self, partL, partR, simulation_parameters)
        num_points = 6 # sim parameters
        
        if partL.grid_shape == 2:
            self.grid_shape_y = min(partL.grid_shape_y,partR.grid_shape_y)
            
        self.grid_shape_x = num_points
        if partL.grid_shape == 2:
            self.grid_shape  = (self.grid_shape_y, self.grid_shape_x)
        else:
            self.grid_shape  = (self.grid_shape_x,)  
            
    def simulate(self):
        
        # cannot reference because of update rule
        p_left = self.partL.p[-3:]
        p_right = self.partR.p[:3]
        
        p = np.hstack((p_left, p_right))
        # Add everything together
        f = np.matmul(p,self.K)
            
        self.partL.f[-3:] += f[:3]
        self.partR.f[:3] += f[-3:]

    # def simulate(self):
    #     for y in range(self.grid_shape_y):
    #         p = np.zeros(shape=[6])
        
    #         # Left room
    #         p[0 : 3] = self.partL.p[y, -3 : ].copy()
        
    #         # Right top room
    #         p[3 : ] = self.partR.p[y, : 3].copy()
        
    #         # Calculate new forces transmitted into room
    #         f = self.K.dot(p)
    #         # f = np.matmul(p,self.K)
    #         # Add everything together
    #         self.partL.f[y, -3] += f[0]
    #         self.partL.f[y, -2] += f[1]
    #         self.partL.f[y, -1] += f[2]
    #         self.partR.f[y, 0] += f[3]
    #         self.partR.f[y, 1] += f[4]
    #         self.partR.f[y, 2] += f[5]

    # def simulate(self):
    #     s = np.array([2, -27, 270, -490, 270, -27, 2]) / (180 * self.dx ** 2)
    #     pi_left = self.partL.p[:,-3:]
    #     pi_right = self.partR.p[:,:3]
                 
    #     pi = np.hstack((pi_left,pi_right))
        
    #     fi = np.zeros((self.grid_shape_y,3)) # forcing term produced" by interface
    #     for il in range(self.grid_shape_y): # all y values (column)
    #         for j in [0,1,2]:# layer                  
    #             for i in range(j-3,-1+1):
    #                 fi[il,j] += pi[il,i+3] * s[j-i+3]
    #                 # fi += pi[j:3,j] * s[j-i+3]
    #             for i in range(0,2-j+1):
    #                 fi[il,j] -= pi[il,i+3] * s[i+j+1+3]     
    #     self.partR.f[:,:3] += self.wave_speed**2 * fi       

if __name__=="__main__":
    coefs = FTDT.get_ftdt_coefficients(2, 6)
    k1 = coefs[0]
    k2 = coefs[1]
    k3 = coefs[2]
    
    K = np.array([k1,0,0,k2,k1,0,k3,k2,k1]).reshape(3,3)
    K = (np.vstack([K,-np.flipud(K)]))
    K = (np.hstack([-np.fliplr(K),K]))
    
    # K_test = np.array(
    #     [
    #         [-0.,         -0.,         -0.01111111,  0.01111111,  0.,          0.        ],
    #         [-0.,         -0.01111111,  0.15,       -0.15,        0.01111111,  0.        ],
    #         [-0.01111111,  0.15,       -1.5,         1.5,        -0.15,        0.01111111],
    #         [ 0.01111111, -0.15,        1.5,        -1.5,         0.15,       -0.01111111],
    #         [ 0.,          0.01111111, -0.15,        0.15,       -0.01111111, -0.        ],
    #         [ 0.,          0.,          0.01111111, -0.01111111, -0.,         -0.        ]
    #     ])
    # print(K == K_test )
        