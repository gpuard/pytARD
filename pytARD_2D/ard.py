import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import idctn, dctn
from common.microphone import Microphone as Mic


class ARDSimulator:
    '''
    ARD Simulation class. Creates and runs ARD simulator instance.
    '''

    def __init__(self, sim_parameters, part_data, pml_partitions):
        '''
        Create and run ARD simulator instance.

        Parameters
        ----------
        sim_parameters : SimulationParameters
            Instance of simulation parameter class.
        part_data : list
            List of PartitionData objects.
        '''

        # Parameter class instance (SimulationParameters)
        self.sim_param = sim_parameters

        # List of partition data (PartitionData objects)
        self.part_data = part_data
        
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
        self.FDTD_COEFFS = fdtd_coeffs_not_normalized * ((sim_parameters.c / part_data[0].h_x) ** 2)

        # FDTD kernel size.
        self.FDTD_KERNEL_SIZE = int((len(fdtd_coeffs_not_normalized[0])) / 2) 

        # Initialize & position mics. 
        # self.mic1 = Mic([int(part_data[0].dimensions[0] / 2), int(part_data[0].dimensions[1] / 2)], sim_parameters, "left")
        # self.mic2 = Mic([int(part_data[1].dimensions[0] / 2), int(part_data[1].dimensions[1] / 2)], sim_parameters, "right")


    def preprocessing(self):
        '''
        Preprocessing stage. Refers to Step 1 in the paper.
        '''

        for i in range(len(self.part_data)):
            self.part_data[i].preprocessing()

        
    def simulation(self):
        '''
        Simulation stage. Refers to Step 2 in the paper.
        '''
        
        s = np.array([2, -27, 270, -490, 270, -27, 2]) / (180 * self.sim_param.dx ** 2)
        
        for t_s in range(1, self.sim_param.number_of_time_samples):
            
            # Interface handing (step 1)
            # Interface handing AIR<-> AIR TOP -> DOWN
            
            pi_top = self.part_data[0].pressure_field[-3:,:]
            pi_bot = self.part_data[1].pressure_field[:3,:]
            
            pi = np.concatenate((pi_top,pi_bot))
            
            interface_length = self.part_data[0].grid_shape[1]
            fi = np.zeros((3,interface_length)) # forcing term produced" by interface
            for il in range (interface_length): # all y values (column)
                for j in [0,1,2]:# layer                  
                    for i in range(j-3,-1+1):
                        fi[j,il] += pi[i+3,il] * s[j-i+3]
                        # fi += pi[j:3,j] * s[j-i+3]
                    for i in range(0,2-j+1):
                        fi[j,il] -= pi[i+3,il] * s[i+j+1+3]
            self.part_data[1].new_forces = np.zeros_like(self.part_data[1].new_forces)
            self.part_data[1].new_forces[:3,:] = self.sim_param.c**2 * fi
 
            # Interface handing AIR<-> AIR TOP <- DOWN
                       
            fi = np.zeros((3,interface_length)) # forcing term produced" by interface
            
            for il in range (interface_length): # all y values (column)
                for j in [0,1,2]:# layer                  
                    for i in range(j-3,-1+1):
                        fi[2-j,il] -= pi[i+3,il] * s[j-i+3]
                        # fi += pi[j:3,j] * s[j-i+3]
                    for i in range(0,2-j+1):
                        fi[2-j,il] += pi[i+3,il] * s[i+j+1+3]
            self.part_data[0].new_forces = np.zeros_like(self.part_data[0].new_forces)
            self.part_data[0].new_forces[-3:,:] = self.sim_param.c**2 * fi
           
            # Interface handing PML <-> AIR 
            # Interface handing AIR -> PML
            
            # pi_left = self.part_data[0].pressure_field[:,-3:]
            # pi_right = self.pml_partitions.p[:,:3]
                     
            # pi = np.concatenate((pi_left,pi_right))
            
            # interface_length = self.part_data[0].grid_shape[1]
            # fi = np.zeros((3,interface_length)) # forcing term produced" by interface
            # for il in range (interface_length): # all y values (column)
            #     for j in [0,1,2]:# layer                  
            #         for i in range(j-3,-1+1):
            #             fi[j,il] += pi[i+3,il] * s[j-i+3]
            #             # fi += pi[j:3,j] * s[j-i+3]
            #         for i in range(0,2-j+1):
            #             fi[j,il] -= pi[i+3,il] * s[i+j+1+3]
            # self.part_data[1].new_forces = np.zeros_like(self.part_data[1].new_forces)
            # self.part_data[1].new_forces[:3,:] = self.sim_param.c**2 * fi        
           
 
            # for i in range(len(self.part_data)):
            #     # Update forcing term (add source)
            #     self.part_data[i].new_forces += self.part_data[i].impulses[t_s].copy()
            #     # self.part_data[i].new_forces += self.part_data[i].impulses[t_s].copy()
            
            self.part_data[0].new_forces += self.part_data[0].impulses[t_s].copy()
            self.part_data[1].new_forces += self.part_data[1].impulses[t_s].copy() 
                    
            # AIR-Partitions (step 2)
            for i in range(len(self.part_data)):
                #print(f"nu forces: {self.part_data[i].new_forces}")
                # Execute DCT for next sample
                self.part_data[i].forces = dctn(self.part_data[i].new_forces, type=1)

                # Updating mode using the update rule in equation 8.
                # Relates to (2 * F^n) / (ω_i ^ 2) * (1 - cos(ω_i * Δ_t)) in equation 8.
                term1 = 2 * self.part_data[i].forces / (self.part_data[i].omega_i ** 2) * (1 - np.cos(self.part_data[i].omega_i * self.sim_param.delta_t))

                # Relates to M^(n+1) in equation 8.
                self.part_data[i].M_next = 2 * self.part_data[i].M_current * \
                np.cos(self.part_data[i].omega_i * self.sim_param.delta_t) - self.part_data[i].M_previous + term1
                
                # Convert modes to pressure values using inverse DCT.
                # self.part_data[i].pressure_field = idctn(self.part_data[i].M_next) 
                self.part_data[i].pressure_field = idctn(self.part_data[i].M_current) 
                
                self.part_data[i].pressure_field_results.append(self.part_data[i].pressure_field.copy())
                
                # if i == 0:
                #     self.mic1.record(self.part_data[0].pressure_field[int(self.part_data[0].space_divisions_y * (self.mic1.location[1] / self.part_data[0].dimensions[1]))][int(self.part_data[0].space_divisions_x * (self.mic1.location[0] / self.part_data[0].dimensions[0]))], t_s)
                # if i == 1:
                #     self.mic2.record(self.part_data[1].pressure_field[int(self.part_data[1].space_divisions_y * (self.mic2.location[1] / self.part_data[1].dimensions[1]))][int(self.part_data[1].space_divisions_x * (self.mic2.location[0] / self.part_data[1].dimensions[0]))], t_s)
                
                # Update time stepping to prepare for next time step / loop iteration.
                self.part_data[i].M_previous = self.part_data[i].M_current.copy()
                self.part_data[i].M_current = self.part_data[i].M_next.copy()


            # # PML-Partitions (step 3)    
            # for part in self.pml_partitions:
            #     part.simulate(t_s)

            # # Interface handling AIR <-> AIR in ??x
            # for y in range(self.part_data[i].grid_shape[1]):
            #     pressure_field_around_interface = np.zeros(shape=[2 * self.FDTD_KERNEL_SIZE, 1])

            #     # Top room
            #     pressure_field_around_interface[0 : self.FDTD_KERNEL_SIZE] = self.part_data[0].pressure_field[-self.FDTD_KERNEL_SIZE :,y].copy().reshape([self.FDTD_KERNEL_SIZE, 1])

            #     # Down room
            #     pressure_field_around_interface[self.FDTD_KERNEL_SIZE : 2 * self.FDTD_KERNEL_SIZE] = self.part_data[1].pressure_field[0 : self.FDTD_KERNEL_SIZE,y].copy().reshape(self.FDTD_KERNEL_SIZE, 1)

            #     new_forces_from_interface = self.FDTD_COEFFS.dot(pressure_field_around_interface)

            #     self.part_data[0].new_forces[-3,y] += new_forces_from_interface[0]
            #     self.part_data[0].new_forces[-2,y] += new_forces_from_interface[1]
            #     self.part_data[0].new_forces[-1,y] += new_forces_from_interface[2]
            #     self.part_data[1].new_forces[ 0,y] += new_forces_from_interface[3]
            #     self.part_data[1].new_forces[ 1,y] += new_forces_from_interface[4]
            #     self.part_data[1].new_forces[ 2,y] += new_forces_from_interface[5]
                
            # # Interface handling AIR <-> PML ::: along X -> therefore for all y's
            # for y in range(self.part_data[0].grid_shape[0]): 
            #     assert self.part_data[0].grid_shape[0] == self.pml_partitions[0].grid_shape[0]
            #     # TODO interfaces need to know which partitions are connected how to solve this? 
            #     # Is it solution in both directions?
                
            #     # Assumption: source is in the room on the left. sound propagets in x to pml room on the right
            #     # pi - pressure_field_interface
            #     pi = np.zeros(shape=[6, 1])

            #     # Left  - get p from the left room
            #     pi[0:3] = self.part_data[0].pressure_field[y,-3:].copy().reshape(pi[0:3].shape)

            #     # Right PML-Parition 
            #     # Note the thickness of PML should be chosen to be at least 3 voxels?
            #     pi[3:6] = self.pml_partitions[0].p[y,:3].copy().reshape(pi[3:].shape)

            #     forcing_value = self.FDTD_COEFFS.dot(pi)

            #     self.part_data[0].new_forces[y,-3] += forcing_value[0]
            #     self.part_data[0].new_forces[y,-2] += forcing_value[1]
            #     self.part_data[0].new_forces[y,-1] += forcing_value[2]
            #     self.pml_partitions[0].f[y,0] += forcing_value[3]
            #     self.pml_partitions[0].f[y,1] += forcing_value[4]
            #     self.pml_partitions[0].f[y,2] += forcing_value[5]
        
        
        # Microphones. TODO: Make mics dynamic
        # self.mic1.write_to_file(self.sim_param.Fs)
        # self.mic2.write_to_file(self.sim_param.Fs)



'''
0NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNXKOkkkkkkkOKXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNXOo;´...  ....´;cokKXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNNNNNNNNNNNNNNNNXOc..               .,lkXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNNNNNNNNNNNNNNNXx´       ...............lKNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNNNNNNNNNNNNNNXx´.   ....´,,,;:::::;,´...dXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNNNNNNNNNNNNNXx´.....,;:ccllooooolcc:;´..:0NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNNNNNNNNNNNNN0;  ...´;:cloddxxxddolcc:,...oXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNNNNNNNNNNNNNO´  ..´,;:clooddddoooolc:,...´dKNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNNNNNNNNNNNNXd.   ..,;:ccloodddddoc:;;,´...:dKNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNNNNNNNNNNNNXd.    ..,;::cccloddolc:;;;;;,;clkXXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNNNNNNNNNNNNN0:     ..´,;;,:lllolcccccc:::;;lkKXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNNNNNNNNNNNNNNO;......,;;;;cllcccc:clolcc:;,:kXXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNNNNNNNNNNNNNNNk:;:;´´;::clllllc:ccclddolc:,l0NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNNNNNNNNNNNNNNNXOlcc:´,:ccllodoc:llc:looolc;oXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNNNNNNNNNNNNNNNNN0dc:,´;:cloddocccll:::cllc:dXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNKkl:´´;:loooc::clc:::cll:;oKNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNKOx;´;clllc:ccccccccc:;,;oxOKXXXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNO:.´;:llcccclllc::;,´´,,;:ccclodxO0KNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNXK0kdc,..´,;::;::clllc;´´,,.´,;;,.......,cokKXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNNNNNNNNNNXKKOOxo:,´,;;;,,,,,,´,,;:c::;´´,;;,,;;:;....     ..´;cox0XNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNNNNNNXKko:,´.......;;;;;;:::,,´´,,,,,,,;:c;.´;::,.   .........´´´;ckKNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNNNNXkc,..   ......´;;;;:::cccc:;;;;;::c:;´..´;:;....´´,,,,,,,´´....´lOXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNNNXx;............´;;;;;;:c;,,;:::::;;,´..  .´::;´´,,,,,´............,:dKNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNNNXOl´..............,;;;;;:ll,..........   ..,,;;,,´.................,;..l0NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNNN0l´...............,,´´,;;;:ol,........ ...´;,´´´...................´,....;kXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNNN0:........ ..............´;:cdxl;..... ..´´..´´............................´dXNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNN0c.´´´,,...................´;:okOxc´.....´..´´´...............................oKNNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNXo´.....´´....................´,:lddc,´..´´´,,´...............................´;oKNNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNNNO;..........´´....................´;loc,´´,,´´´..............................,,.´oKNNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNNN0c...................................´lkd;´,,´..´............´´´,,,,;;;,....;;´...´oKNNNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNN0c....................................´ckkl;;;:ll;´´´,,;;;,,,,,;,,´´.......,;´.......l0NNNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNNKl´´´.............´;;;;;;;;,´........´´,lO0xooxO0Odc;;,,,´...........  ...´,´..........c0NNNNNNNNNNNNNNNNNNNNNNN
0NNNNNNXd,´´.....................´´´.´,;;;;;;,,ckKKOkkxxO00o,......  ...´´..  ..................cOXNNNNNNNNNNNNNNNNNNNNN
0NNNNNXo.............´´........................,x00OOkxkOOkl,.  ..  .............................,xXNNNNNNNNNNNNNNNNNNNN
0NNNNNO;...´´.....   ..................   ......,::;:cc:,''...   ..    .....','...................;kXNNNNNNNNNNNNNNNNNNN
0NNNNNk;';;;,......... ...     .......     ........              ......',,,,'......................'dXNNNNNNNNNNNNNNNXXX
0NNNNXx;''...................  .. ..            .......',',,,,,,,,,,,,,,,'....................''....;ONNNNNNNXXXXXXXXXXX
0NNNNk;......................'','...........'',,,,,,;;;;;,,,,,,,','..............'c:......''.....   .xXNNXK0OOOOO000KKKK
0NNNKc...............,,'......,c:,',;;;;,,,,,,''.................................,kKd,......      ...ck000OOOOOOOOOOO00K
0NNNk,............';:,........;;'................................................:0NXd.        ..,;ldkOO0000000000000000
0NNXo............,:,.........;:'................................................'oXNNO,    ...'cdxkO000KKKKKK00000000000
0NNXo.........'.............'c:...............'............''''''....'......... .xNNNXx,....;oxkkOO000KKK0KKK00KKKKK0000
0NNXo........,;'............;l,...........''..............':ccc::::::;;,,,'..   .dXNNNXx;.'lk0OkkO000KKK00000KKKKKKKKKK0
0NNXx'.....'',::;;,'........cc'...........................;oddolllcccccclll:,.  .oXNNNNXx:lk00OxkO00000000000KKKKKKKKKKK


'''
