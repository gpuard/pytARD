from common.microphone import Microphone as Mic

import numpy as np
from scipy.fft import idctn, dctn
from tqdm import tqdm


class ARDSimulator:
    '''
    ARD Simulation class. Creates and runs ARD simulator instance.
    '''

    def __init__(self, sim_parameters, part_data):
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
        # Important: For each direction the sound passes through an interface, the according FDTD coeffs should be used.
        self.FDTD_COEFFS_X = fdtd_coeffs_not_normalized * ((sim_parameters.c / part_data[0].h_x) ** 2)
        self.FDTD_COEFFS_Y = fdtd_coeffs_not_normalized * ((sim_parameters.c / part_data[0].h_y) ** 2)
        self.FDTD_COEFFS_Z = fdtd_coeffs_not_normalized * ((sim_parameters.c / part_data[0].h_z) ** 2)

        # FDTD kernel size.
        self.FDTD_KERNEL_SIZE = int((len(fdtd_coeffs_not_normalized[0])) / 2) 

        # Initialize & position mics.
        # TODO: Instantiate mics in example script
        self.mic1 = Mic(
            0,
            [
                int(part_data[0].dimensions[0] / 2), 
                int(part_data[0].dimensions[1] / 2), 
                int(part_data[0].dimensions[2] / 2)
            ], sim_parameters, "left"
        )

        self.mic2 = Mic(
            1,
            [
                int(part_data[1].dimensions[0] / 2), 
                int(part_data[1].dimensions[1] / 2), 
                int(part_data[1].dimensions[2] / 2)
            ], sim_parameters, "right"
        )

        self.mic3 = Mic(
            2,
            [
                int(part_data[2].dimensions[0] / 2), 
                int(part_data[2].dimensions[1] / 2), 
                int(part_data[2].dimensions[2] / 2)
            ], sim_parameters, "bottom"
        )

        self.mics = [self.mic1, self.mic2, self.mic3]



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

        for t_s in tqdm(range(2, self.sim_param.number_of_samples)):
            for i in range(len(self.part_data)):
                #print(f"nu forces: {self.part_data[i].new_forces}")
                # Execute DCT for next sample
                self.part_data[i].forces = dctn(self.part_data[i].new_forces, type=1)

                # Updating mode using the update rule in equation 8.
                # Relates to (2 * F^n) / (ω_i ^ 2) * (1 - cos(ω_i * Δ_t)) in equation 8.
                self.part_data[i].force_field = (
                    (2 * self.part_data[i].forces.reshape(
                        [
                            self.part_data[i].space_divisions_z, 
                            self.part_data[i].space_divisions_y, 
                            self.part_data[i].space_divisions_x, 
                            1
                        ])) / ((self.part_data[i].omega_i) ** 2)) * (1 - np.cos(self.part_data[i].omega_i * self.sim_param.delta_t))

                # Relates to M^(n+1) in equation 8.
                self.part_data[i].M_next = 2 * self.part_data[i].M_current * \
                np.cos(self.part_data[i].omega_i * self.sim_param.delta_t) - self.part_data[i].M_previous + self.part_data[i].force_field
                
                # Convert modes to pressure values using inverse DCT.
                self.part_data[i].pressure_field = idctn(
                    self.part_data[i].M_next.reshape(
                        self.part_data[i].space_divisions_z, 
                        self.part_data[i].space_divisions_y, 
                        self.part_data[i].space_divisions_x
                    ), type=1) 
                
                self.part_data[i].pressure_field_results.append(self.part_data[i].pressure_field.copy())
                
                for mic in self.mics:
                    if i == mic.partition_number:
                        self.record_to_mic(mic, t_s)

                # Update time stepping to prepare for next time step / loop iteration.
                self.part_data[i].M_previous = self.part_data[i].M_current.copy()
                self.part_data[i].M_current = self.part_data[i].M_next.copy()

                # Update impulses
                self.part_data[i].new_forces = self.part_data[i].impulses[t_s].copy()

            
            # Interface handling Left -> Right (through y axis)
            for z in range(self.part_data[i].space_divisions_z):
                for y in range(self.part_data[i].space_divisions_y):
                    pressure_field_around_interface_y = np.zeros(shape=[2 * self.FDTD_KERNEL_SIZE, 1])

                    # Left room
                    pressure_field_around_interface_y[0 : self.FDTD_KERNEL_SIZE] = self.part_data[0].pressure_field[z, y, -self.FDTD_KERNEL_SIZE : ].copy().reshape([self.FDTD_KERNEL_SIZE, 1])

                    # Right top room
                    pressure_field_around_interface_y[self.FDTD_KERNEL_SIZE : 2 * self.FDTD_KERNEL_SIZE] = self.part_data[1].pressure_field[z, y, 0 : self.FDTD_KERNEL_SIZE].copy().reshape(self.FDTD_KERNEL_SIZE, 1)

                    # Calculate new forces transmitted into room. Use X coeffs, because we pass the interface in X direction.
                    new_forces_from_interface_y = self.FDTD_COEFFS_X.dot(pressure_field_around_interface_y)

                    # Add everything together
                    self.part_data[0].new_forces[z, y, -3] += new_forces_from_interface_y[0]
                    self.part_data[0].new_forces[z, y, -2] += new_forces_from_interface_y[1]
                    self.part_data[0].new_forces[z, y, -1] += new_forces_from_interface_y[2]
                    self.part_data[1].new_forces[z, y, 0] += new_forces_from_interface_y[3]
                    self.part_data[1].new_forces[z, y, 1] += new_forces_from_interface_y[4]
                    self.part_data[1].new_forces[z, y, 2] += new_forces_from_interface_y[5]

            # Interface handling Right -> Bottom (through x axis)
            for z in range(self.part_data[i].space_divisions_z):
                for x in range(self.part_data[i].space_divisions_x):
                    pressure_field_around_interface_y = np.zeros(shape=[2 * self.FDTD_KERNEL_SIZE, 1])

                    # Right top room
                    pressure_field_around_interface_y[0 : self.FDTD_KERNEL_SIZE] = self.part_data[1].pressure_field[z, -self.FDTD_KERNEL_SIZE : , x].copy().reshape([self.FDTD_KERNEL_SIZE, 1])

                    # Right bottom room
                    pressure_field_around_interface_y[self.FDTD_KERNEL_SIZE : 2 * self.FDTD_KERNEL_SIZE] = self.part_data[2].pressure_field[z, 0 : self.FDTD_KERNEL_SIZE, x].copy().reshape(self.FDTD_KERNEL_SIZE, 1)

                    # Calculate new forces transmitted into room
                    new_forces_from_interface_y = self.FDTD_COEFFS_X.dot(pressure_field_around_interface_y)

                    # Add everything together
                    self.part_data[1].new_forces[z, -3, x] += new_forces_from_interface_y[0]
                    self.part_data[1].new_forces[z, -2, x] += new_forces_from_interface_y[1]
                    self.part_data[1].new_forces[z, -1, x] += new_forces_from_interface_y[2]
                    self.part_data[2].new_forces[z, 0, x] += new_forces_from_interface_y[3]
                    self.part_data[2].new_forces[z, 1, x] += new_forces_from_interface_y[4]
                    self.part_data[2].new_forces[z, 2, x] += new_forces_from_interface_y[5]


        # Microphones. TODO: Make mics dynamic
        for mic in self.mics:
            mic.write_to_file(self.sim_param.Fs)


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
