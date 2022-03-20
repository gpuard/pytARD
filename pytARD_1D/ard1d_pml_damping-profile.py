# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 22:37:09 2022

@author: smailnik@students.zhaw.ch
"""

import numpy as np
from numpy import log, sin, pi, absolute
import matplotlib.pyplot as plt


# The purpose of this small example is to visualize the damping profile.

# Source: Efficient PML for the wave equation Marcus J. Grote1, and Imbo Sim2

# Assumption:   1d rooom of size loc_size_x, 
#               and pml-layer of pml_thickness_x, located at loc_ax
          

c = 330 # speed of sound in m/s
loc_size_x = 50 # m
grid_x = np.arange(0,loc_size_x,0.01)

the_constant = [10,25,40,100]

loc_ax = 15 # location of pml layer
pml_thickness_x = 20

def damping_profile(x, the_constant, loc_ax=loc_ax, pml_thickness_x=pml_thickness_x):
    if x < loc_ax:
        # inside of air-partition - no damping
        return 0
    elif x <= loc_ax + pml_thickness_x:
        # inside of pml-layer
        return the_constant*(absolute(x-loc_ax) / pml_thickness_x - sin(2*pi*absolute(x-loc_ax) / pml_thickness_x) / (2*pi) )
    else:
        return 0

fig, ax = plt.subplots()

dp = list()
for k in the_constant:   
    dp.append( [damping_profile(x, k) for x in grid_x] )

# ax.scatter(grid_x, dp,color='r',marker='.')
for i in range(len(dp)):
    ax.plot(grid_x, dp[i], label=the_constant[i])
    
ax.axvline(loc_ax,color='k', linestyle='--', label="pml-borders")
ax.axvline(loc_ax+pml_thickness_x,color='k', linestyle='--')
ax.legend()