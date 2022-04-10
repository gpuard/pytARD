import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import idctn, dctn

from pytARD_2D.interface import Interface2D

class ARDSimulator:
    '''
    ARD Simulation class. Creates and runs ARD simulator instance.
    '''

    def __init__(self, sim_param, part_data, interface_data=[], mics=[]):
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
        self.sim_param = sim_param

        # List of partition data (PartitionData objects)
        self.part_data = part_data

        # List of interfaces (InterfaceData objects)
        self.interface_data = interface_data
        self.interfaces = Interface2D(sim_param, part_data)

        # Initialize & position mics. 
        self.mics = mics


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
        if self.sim_param.verbose:
            print(f"Simulation started.")
        
        for t_s in range(2, self.sim_param.number_of_samples):
            for i in range(len(self.part_data)):
                #print(f"nu forces: {self.part_data[i].new_forces}")
                # Execute DCT for next sample
                self.part_data[i].forces = dctn(self.part_data[i].new_forces, type=1)

                # Updating mode using the update rule in equation 8.
                # Relates to (2 * F^n) / (ω_i ^ 2) * (1 - cos(ω_i * Δ_t)) in equation 8.
                self.part_data[i].force_field = ((2 * self.part_data[i].forces.reshape([self.part_data[i].space_divisions_y, self.part_data[i].space_divisions_x, 1])) / (
                    (self.part_data[i].omega_i) ** 2)) * (1 - np.cos(self.part_data[i].omega_i * self.sim_param.delta_t))

                self.part_data[i].force_field[0, 0, 0] = 0

                # Relates to M^(n+1) in equation 8.
                self.part_data[i].M_next = 2 * self.part_data[i].M_current * \
                np.cos(self.part_data[i].omega_i * self.sim_param.delta_t) - self.part_data[i].M_previous + self.part_data[i].force_field
                
                # Convert modes to pressure values using inverse DCT.
                self.part_data[i].pressure_field = idctn(self.part_data[i].M_next.reshape(
                    self.part_data[i].space_divisions_y, self.part_data[i].space_divisions_x), type=1) 
                
                self.part_data[i].pressure_field_results.append(self.part_data[i].pressure_field.copy())
                
                # Loop through microphones and record pressure field at given position
                for m_i in range(len(self.mics)):
                    p_num = self.mics[m_i].partition_number
                    pressure_field_y = int(self.part_data[p_num].space_divisions_y * (self.mics[m_i].location[1] / self.part_data[p_num].dimensions[1]))
                    pressure_field_x = int(self.part_data[p_num].space_divisions_x * (self.mics[m_i].location[0] / self.part_data[p_num].dimensions[0]))
                    
                    self.mics[m_i].record(self.part_data[p_num].pressure_field.copy().reshape(
                        [self.part_data[p_num].space_divisions_y, self.part_data[p_num].space_divisions_x, 1])[pressure_field_y][pressure_field_x], t_s)

                # Update time stepping to prepare for next time step / loop iteration.
                self.part_data[i].M_previous = self.part_data[i].M_current.copy()
                self.part_data[i].M_current = self.part_data[i].M_next.copy()

                # Update impulses
                self.part_data[i].new_forces = self.part_data[i].impulses[t_s].copy()

            # Interface monke go here
            for interface in self.interface_data:
                self.interfaces.handle_interface(interface)

        if self.sim_param.verbose:
            print(f"Simulation completed successfully.\n")
        


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
