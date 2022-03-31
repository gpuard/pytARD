# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 06:55:29 2022

@author: smailnik@students.zhaw.ch
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import idctn, dctn
from common.microphone import Microphone as Mic


class ARDSimulator:
    '''
    ARD Simulation class. Creates and runs ARD simulator instance.
    '''

    def __init__(self, sim_parameters, air_partitions, pml_partitions):
        '''
        Create and run ARD simulator instance.

        Parameters
        ----------
        sim_parameters : SimulationParameters
            Instance of simulation parameter class.
        air_partitions : list
            List of PartitionData objects.
        '''

        # Parameter class instance (SimulationParameters)
        self.sim_param = sim_parameters

        # List of partition data (PartitionData objects)
        self.air_partitions = air_partitions
        
        # List containing PML-Partitions
        self.pml_partitions = pml_partitions


        # 2D FDTD coefficents array. Normalize FDTD coefficents with space divisions and speed of sound. 
        fdtd_coeffs_not_normalized = np.array(
            [
                [-0.,         -0.,         -0.01111111,  0.01111111,  0.,          0.        ],
                [-0.,         -0.01111111,  0.15,       -0.15,        0.01111111,  0.        ],
                [-0.01111111,  0.15,       -1.5,         1.5,        -0.15,        0.01111111],
                [ 0.01111111, -0.15,        1.5,        -1.5,         0.15,       -0.01111111],
                [ 0.,          0.01111111, -0.15,        0.15,       -0.01111111, -0.        ],
                [ 0.,          0.,          0.01111111, -0.01111111, -0.,         -0.        ]
            ]
        )

        # TODO: Unify h of partition data, atm it's hard coded to first partition
        self.FDTD_COEFFS = fdtd_coeffs_not_normalized * ((sim_parameters.c / air_partitions[0].h_x) ** 2)

        # FDTD kernel size.
        self.FDTD_KERNEL_SIZE = int((len(fdtd_coeffs_not_normalized[0])) / 2) 

        # Initialize & position mics. 
        # self.mic1 = Mic([int(air_partitions[0].dimensions[0] / 2), int(air_partitions[0].dimensions[1] / 2)], sim_parameters, "left")
        # self.mic2 = Mic([int(air_partitions[1].dimensions[0] / 2), int(air_partitions[1].dimensions[1] / 2)], sim_parameters, "right")


    def preprocessing(self):
        '''
        Preprocessing stage. Refers to Step 1 in the paper.
        '''

        for i in range(len(self.air_partitions)):
            self.air_partitions[i].preprocessing()

        
    def simulation(self):
        '''
        Simulation stage. Refers to Step 2 in the paper.
        '''
        
        s = np.array([2, -27, 270, -490, 270, -27, 2]) / (180 * self.sim_param.dx ** 2)
        
        for t_s in range(1, self.sim_param.number_of_time_samples):
            
            ###########################
            # RESETING FORCING FIELDS #
            ###########################

            for part in self.pml_partitions:
                part.f = np.zeros_like(part.f)
                
            for part in self.air_partitions:
                part.new_forces = np.zeros_like(part.new_forces)
                
            ###############################
            # STEP 1: HANDLING INTERFACES #
            ###############################

            #############
            # PML - AIR #
            #############
            
            ### LEFT(AIR) -> RIGHT(PML) ###
            
            pi_left = self.air_partitions[0].pressure_field[:,-3:]
            pi_right = self.pml_partitions[0].p[:,:3]
                     
            pi = np.hstack((pi_left,pi_right))
            
            interface_length = self.air_partitions[0].grid_shape[0]
            fi = np.zeros((interface_length,3)) # forcing term produced" by interface
            for il in range (interface_length): # all y values (column)
                for j in [0,1,2]:# layer                  
                    for i in range(j-3,-1+1):
                        fi[il,j] += pi[il,i+3] * s[j-i+3]
                        # fi += pi[j:3,j] * s[j-i+3]
                    for i in range(0,2-j+1):
                        fi[il,j] -= pi[il,i+3] * s[i+j+1+3]     
            self.pml_partitions[0].f[:,:3] += self.sim_param.c**2 * fi        
           
            ### LEFT(AIR) <- RIGHT(PML) ###
            
            pi_left = self.air_partitions[0].pressure_field[:,-3:]
            pi_right = self.pml_partitions[0].p[:,:3]
                     
            pi = np.hstack((pi_left,pi_right))
            
            interface_length = self.air_partitions[0].grid_shape[0]
            fi = np.zeros((interface_length,3)) # forcing term produced" by interface
            for il in range (interface_length): # all y values (column)
                for j in [0,1,2]:# layer                  
                    for i in range(j-3,-1+1):
                        fi[il,2-j] -= pi[il,i+3] * s[j-i+3]
                        # fi += pi[j:3,j] * s[j-i+3]
                    for i in range(0,2-j+1):
                        fi[il,2-j] += pi[il,i+3] * s[i+j+1+3]
            self.air_partitions[0].new_forces[:,-3:] += self.sim_param.c**2 * fi           
            # self.air_partitions[0].new_forces[:,:3] += self.sim_param.c**2 * fi  # uncomment for education purposes

            ### SECOND X INTERFACE ###

            ### LEFT(PML) -> RIGHT(AIR) ###
            
            pi_left = self.pml_partitions[0].p[:,-3:]
            pi_right = self.air_partitions[1].pressure_field[:,:3]
                     
            pi = np.hstack((pi_left,pi_right))
            
            interface_length = self.air_partitions[0].grid_shape[0]
            fi = np.zeros((interface_length,3)) # forcing term produced" by interface
            for il in range (interface_length): # all y values (column)
                for j in [0,1,2]:# layer                  
                    for i in range(j-3,-1+1):
                        fi[il,j] += pi[il,i+3] * s[j-i+3]
                        # fi += pi[j:3,j] * s[j-i+3]
                    for i in range(0,2-j+1):
                        fi[il,j] -= pi[il,i+3] * s[i+j+1+3]     
            self.air_partitions[1].new_forces[:,:3] += self.sim_param.c**2 * fi        
           
            ### LEFT(AIR) <- RIGHT(PML) ###
            
            pi_left = self.pml_partitions[0].p[:,-3:]
            pi_right = self.air_partitions[1].pressure_field[:,:3]
                     
            pi = np.hstack((pi_left,pi_right))
            
            interface_length = self.air_partitions[0].grid_shape[0]
            fi = np.zeros((interface_length,3)) # forcing term produced" by interface
            for il in range (interface_length): # all y values (column)
                for j in [0,1,2]:# layer                  
                    for i in range(j-3,-1+1):
                        fi[il,2-j] -= pi[il,i+3] * s[j-i+3]
                        # fi += pi[j:3,j] * s[j-i+3]
                    for i in range(0,2-j+1):
                        fi[il,2-j] += pi[il,i+3] * s[i+j+1+3]
            self.pml_partitions[0].f[:,-3:] += self.sim_param.c**2 * fi           
    
            #####################
            # INJECTING SOURCES #
            #####################
 
            for i in range(len(self.air_partitions)):
                self.air_partitions[i].new_forces += self.air_partitions[i].impulses[t_s].copy()
  
              
            ###################################
            # STEP 2: AIR PARTITIONS HANDLING #
            ###################################
            
            for i in range(len(self.air_partitions)):
                
                # Execute DCT for next sample
                self.air_partitions[i].forces = dctn(self.air_partitions[i].new_forces, type=1)

                # Updating mode using the update rule in equation 8.
                # Relates to (2 * F^n) / (ω_i ^ 2) * (1 - cos(ω_i * Δ_t)) in equation 8.
                term1 = 2 * self.air_partitions[i].forces / (self.air_partitions[i].omega_i ** 2) * (1 - np.cos(self.air_partitions[i].omega_i * self.sim_param.delta_t))

                # Relates to M^(n+1) in equation 8.
                self.air_partitions[i].M_next = 2 * self.air_partitions[i].M_current * \
                np.cos(self.air_partitions[i].omega_i * self.sim_param.delta_t) - self.air_partitions[i].M_previous + term1
                
                # Convert modes to pressure values using inverse DCT.
                # self.air_partitions[i].pressure_field = idctn(self.air_partitions[i].M_next) 
                self.air_partitions[i].pressure_field = idctn(self.air_partitions[i].M_current) 
                
                self.air_partitions[i].pressure_field_results.append(self.air_partitions[i].pressure_field.copy())
                
                # if i == 0:
                #     self.mic1.record(self.air_partitions[0].pressure_field[int(self.air_partitions[0].space_divisions_y * (self.mic1.location[1] / self.air_partitions[0].dimensions[1]))][int(self.air_partitions[0].space_divisions_x * (self.mic1.location[0] / self.air_partitions[0].dimensions[0]))], t_s)
                # if i == 1:
                #     self.mic2.record(self.air_partitions[1].pressure_field[int(self.air_partitions[1].space_divisions_y * (self.mic2.location[1] / self.air_partitions[1].dimensions[1]))][int(self.air_partitions[1].space_divisions_x * (self.mic2.location[0] / self.air_partitions[1].dimensions[0]))], t_s)
                
                # Update time stepping to prepare for next time step / loop iteration.
                self.air_partitions[i].M_previous = self.air_partitions[i].M_current.copy()
                self.air_partitions[i].M_current = self.air_partitions[i].M_next.copy()

            ###################################
            # STEP 3: PML PARTITIONS HANDLING #
            ###################################
            
            for pml_part in self.pml_partitions:
                pml_part.simulate(t_s)
   
        # Microphones. TODO: Make mics dynamic
        # self.mic1.write_to_file(self.sim_param.Fs)
        # self.mic2.write_to_file(self.sim_param.Fs)