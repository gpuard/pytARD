# -*- coding: utf-8 -*-
import numpy as np
class FD:
    FD_COEFFICIENTS = {1 : {  2 : np.array([-1/2, 0, 1/2]),
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
    def get_fd_coefficients(derivative, accuracy):
        return FD.FD_COEFFICIENTS[derivative][accuracy]
   
    @staticmethod
    def get_laplacian_matrix():
        coefs = FD.get_fd_coefficients(2,6)
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