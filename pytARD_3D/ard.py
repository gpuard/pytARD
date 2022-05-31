from pytARD_3D.interface import Interface3D
from common.microphone import Microphone as Mic

import numpy as np
from scipy.fft import idctn, dctn
from tqdm import tqdm


class ARDSimulator:
    '''
    ARD Simulation class. Creates and runs ARD simulator instance.
    '''

    def __init__(self, sim_param, part_data, normalization_factor, interface_data=[], mics=[]):
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
        self.interfaces = Interface3D(sim_param, part_data)

        # Initialize & position mics.
        self.mics = mics

        self.normalization_factor = normalization_factor

    def preprocessing(self):
        '''
        Preprocessing stage. Refers to Step 1 in the paper.
        '''

        for i in range(len(self.part_data)):
            self.part_data[i].preprocessing()


    def record_to_mic(self, mic, t_s):
        '''
        TODO: Doc
        '''
        pressure_field_z = int(self.part_data[mic.partition_number].space_divisions_z * (mic.location[2] / self.part_data[mic.partition_number].dimensions[2]))
        pressure_field_y = int(self.part_data[mic.partition_number].space_divisions_y * (mic.location[1] / self.part_data[mic.partition_number].dimensions[1]))
        pressure_field_x = int(self.part_data[mic.partition_number].space_divisions_x * (mic.location[0] / self.part_data[mic.partition_number].dimensions[0]))
        mic.record(self.part_data[mic.partition_number].pressure_field[pressure_field_z][pressure_field_y][pressure_field_x], t_s)

        
    def simulation(self):
        '''
        Simulation stage. Refers to Step 2 in the paper.
        '''
        if self.sim_param.verbose:
            print(f"Simulation started.")

        for t_s in tqdm(range(2, self.sim_param.number_of_samples)):
            # Interface Handling
            for interface in self.interface_data:
                self.interfaces.handle_interface(interface)

            for i in range(len(self.part_data)):
                #print(f"nu forces: {self.part_data[i].new_forces}")
                # Execute DCT for next sample
                self.part_data[i].forces = dctn(self.part_data[i].new_forces, 
                type=2, 
                s=[ # TODO This parameter may be unnecessary
                    self.part_data[i].space_divisions_z, 
                    self.part_data[i].space_divisions_y, 
                    self.part_data[i].space_divisions_x
                ])

                # Updating mode using the update rule in equation 8.
                # Relates to (2 * F^n) / (ω_i ^ 2) * (1 - cos(ω_i * Δ_t)) in equation 8.
                self.part_data[i].force_field = (
                    (2 * self.part_data[i].forces) / ((self.part_data[i].omega_i) ** 2)) * (
                        1 - np.cos(self.part_data[i].omega_i * self.sim_param.delta_t))

                # Edge case for first iteration according to Nikunj Raghuvanshi. p[n+1] = 2*p[n] – p[n-1] + (\delta t)^2 f[n], while f is impulse and p is pressure field.
                self.part_data[i].force_field[0, 0, 0] = 2 * self.part_data[i].M_current[0, 0, 0] - self.part_data[i].M_previous[0, 0, 0] + \
                    self.sim_param.delta_t ** 2 * \
                        self.part_data[i].impulses[t_s][0, 0, 0]

                # Relates to M^(n+1) in equation 8.
                self.part_data[i].M_next = 2 * self.part_data[i].M_current * \
                np.cos(self.part_data[i].omega_i * self.sim_param.delta_t) - self.part_data[i].M_previous + self.part_data[i].force_field
                
                # Convert modes to pressure values using inverse DCT.
                self.part_data[i].pressure_field = idctn(
                    self.part_data[i].M_next.reshape(
                        self.part_data[i].space_divisions_z, 
                        self.part_data[i].space_divisions_y, 
                        self.part_data[i].space_divisions_x
                    ), type=2,
                s=[ # TODO This parameter may be unnecessary
                    self.part_data[i].space_divisions_z, 
                    self.part_data[i].space_divisions_y, 
                    self.part_data[i].space_divisions_x
                ])
                
                self.part_data[i].pressure_field_results.append(self.part_data[i].pressure_field.copy())
                
                # Loop through microphones and record pressure field at given position
                for m_i in range(len(self.mics)):
                    p_num = self.mics[m_i].partition_number
                    pressure_field_z = int(self.part_data[p_num].space_divisions_z * (
                        self.mics[m_i].location[2] / self.part_data[p_num].dimensions[2]))
                    pressure_field_y = int(self.part_data[p_num].space_divisions_y * (
                        self.mics[m_i].location[1] / self.part_data[p_num].dimensions[1]))
                    pressure_field_x = int(self.part_data[p_num].space_divisions_x * (
                        self.mics[m_i].location[0] / self.part_data[p_num].dimensions[0]))

                    self.mics[m_i].record(self.part_data[p_num].pressure_field.copy().reshape(
                        [
                            self.part_data[p_num].space_divisions_z, 
                            self.part_data[p_num].space_divisions_y, 
                            self.part_data[p_num].space_divisions_x, 1
                        ])[pressure_field_z][pressure_field_y][pressure_field_x], t_s)

                # Update time stepping to prepare for next time step / loop iteration.
                self.part_data[i].M_previous = self.part_data[i].M_current.copy()
                self.part_data[i].M_current = self.part_data[i].M_next.copy()

                # Update impulses
                self.part_data[i].new_forces = self.part_data[i].impulses[t_s].copy()

        if self.sim_param.verbose:
            print(f"Simulation completed successfully.\n")


'''
********************************************°*°°°************°°°°°°°°...°°°°°°***°**************************************
*********************************************°°°°°°°°°°°°°°°...             ...°°°°*************************************
*************************************°***°°*°°°°°°°°°.....                 ....  ....°°°°*******************************
************************************°°**°°*°°°°°....                                  ....°°°***************************
*************************************°°°°°°°°...                                          ....°°°°**********************
*************************************°°°°°..                             .  ..                 ..°°*********************
************************************°°°...                      ...                .             ..°°°**°***************
**********************************°°....              .........°°°.......        ..     .           ..°°°***************
*********************************°°...           ...°°°*************°°°°°.......... ......         . ...°°°°************
******************************°°°...          .°°****ooooooooooooooooo****°°°°°°°°°°....                ..°°°°°*********
****************************°°°..           ..***ooooooooooooooooooooooooooooo******°°°..                ...°°°°°*******
**************************°°°..           ..°**oooooooOooooooOOOOOOOOOOoOOOOooooooooooo**°...             ....°°°°******
*************************°°°..           ..°*oooooooooOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOoo°.         ..   ...°°°°*****
**********************°°°°°...          ..°*oooooooooooOOooOoOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOoo*°.        .   ..°°°°*****
*********************°°°°°...           .°*ooooooooooooooooooOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOoo*°.           ..°°°*****
*******************°°°°°°°....         ..°*ooooooooooooOooooOOoOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOoo°..          ....°°****
*******************°°°°°°... .        ...°*ooooooooooooOOooOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOoo*°..       .. ...°°****
*******************°°°°°°°..           ..°*ooooooooOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO*°°..          ....°°***
********************°°°°°...     ..... ..°oooooooooo********ooOOOOOOOOOOOOOoooooooOOOOOOOOOOOOOOo°°...       .......°°**
**********************°°°..       °°°°...*ooooooo*****°°****ooooooOOOOOOOOooo****°***ooooooOOOOOo°°...  .°*°.........°°*
******************°°°°°°...       .°°°..°oooooooo****°°°°°****ooooOOOOOOOooooo*******oooooOOOOOOO*°....*****. .......°°°
****************°°°°°°°°...        °°*.°*oooooo**°°*o*...°o*°***ooOOOOOOoo****o*°.°°***oooOOOOOOOo°°°°*oo***..........°°
***************°°°*°°... ..        .**°°oooooooooooooo*********oooooOOOOooo***oo°°°*o***oooOOOOOOO*°°**oo*o° ..........°
*****************°°°...   .         **°*oooooooooooooooo*****oooooooOOOOOOoo***ooooooOOOOoOOOOOOOOo°°**oo*o  .. ........
****************°°°...              **°*ooooooooooooooooooooooooooOoOOOOOOOOOooooooOOOOOOoOOOOOOOOOoo***oo° ... ........
*************°°°°°.                 ***oooooooooooooooooooooooOoooooOOOOOOOOOOOOooooooooOOOOOOOOOOOOo**oO*  ............
************°°°°..   .             .*ooooooooooooooooooooooOoooooooOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO*oOoo. .............
**********°°°°°..                  .o**oooooooooooooooooooooooooooOOOOOOOOoOOOOOOOOOOOOOOOOOOOOOOOOOOOOO* ...   ........
*******°°°°°...                     °*oooooooOoooooooOoOOoo*oooooOOOOOOOOOOooOOOOOOOOOOOOOOOOOOOOOOOOOOO. ..............
******°°°......                        *ooooOOOOOOoooooooooooo**ooooooo**ooooooOOOOOOOOOOOOOOOOOOOOOOOO° ..............°
*****°°...°...    .                    °ooooOOOOOOooooooooooooooooooooooooOOOoooOOOOOOOOOOOOOOOOOOo**°. ........ .......
*****°......                           .oooooOOOooooooooooooooooooooooOOOOOOOOOooOOOOOOOOOOOOOOOOO.   .. ...............
***°°........                           °OoooOOOooooo*************oooooooooooooooooOOOOOOOOOOOOO#°   ... ........  .....
*°°°.......                              *ooOOOooooooooooooooooooooooooooooooooooooOOOOOOOOOOOOO*  ....°...... .    ....
°.°°°..                                  °ooOOOOOooooooooooooooooooooooOOOOOOOOOooOOOOOOOOOOOOOo .......°.....      ....
 ....          .  .                      °oooooOOOooooooooooooooooooOOOOOOOOOOOOOOOOOOOOOOOOOOO* ........... .     .....
.                     .....              *oooooooooooooooooooooooOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOo.....°°°°°°.°..        .
                     ..°°.°...        .°°oooooooooooooooooooooooOOOOOOOOOJULAYYOOOOOOOOOOOOOOOOO°.....°oOOOoOOOOoo*°°...
                    ..°°°°...........°***oooooooooooooooooooooooooooOOOOOOOOOOOOOOOOOOOOOOOOOO#Oo**°°°°.o#####O#######OO
            .  ....°***oo°....°°°°.°*o***oOOooooooooooooooooooooooooOOOOOOOOOOOOOOOOOOOOOOOOOO#Ooooo***° *##############
..°°°°°********oo*oOOOOOO..°°*****°oooo**ooOoooooooooooooooooooooooooOOOOOOOOOOOOOOOOOOOOOOO###OoOOOo***. *#############
ooOOOoOOOOOOOOOOOOOOOOOOO*.°°*****oooooooooOOOooooooooooooooooooooOOOoooooOOOOOOOOOOOOOOOOOOOOOOOOO#o°**..o#############
OOOOOOOOOOOOOO#OOO##O##OOO*.°****ooOOoooooooOOOooooooooooooooooooooOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO##O**°°O###@##########
O#OO##OOOOOOO###########OOOo°°**ooOOOOOooooooOOOoooooooooooooooooOOOOOOOOOOOoOOOOOOOOOOOOOOOOOOOO####*.*################
#########OOOOOO#############Oo°°oOOOOOOOooooooOOOOoooooooooooooooooooOOoooooooOOOOOOOOOO#OOOOOOO###Oo*O#################
###########OOOOOO###########O#OoooOOOOOOOOooooooOOOoooooooo***oooooooooooooooooOOOOOOOOOOOOOOOO##OOOO###################
###############OOOOO#########OO#OOooOOOOOOOOOoooooooooooooooooo****ooooooooOOoOOOOOOOOOOOOOOOOOOOO######################
##################OOOOO########OOOOOOoOOOOOOOOOOoooooooooooooooooooooooooOOOOOOOOOOOOOOOOOOOOOO#########################
#####################OOOOOO#######OOOOOoooOOOOOOOOoooooooooooooooooooooOOOOOOOOOOOOOOOOOOOOO############################
########################OOOOOO#####OOOOOOOoooOOOOOOOOOooooooooooooooOOOOOOOOOOOOOOOOOOOO################################



'''
