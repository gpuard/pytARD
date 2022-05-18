from common.microphone import Microphone
from common.notification import Notification
from common.parameters import SimulationParameters

from pytARD_3D.interface import Interface3DLooped, Interface3D

from tqdm import tqdm

class ARDSimulator3D:
    '''
    ARD Simulation class. Creates and runs ARD simulator instance.
    '''

    def __init__(
        self, 
        sim_param: SimulationParameters, 
        partitions: list, 
        normalization_factor: float, 
        interface_data: list=[], 
        mics: list=[]
    ):
        
        '''
        Create and prepare ARD simulator instance.

        Parameters
        ----------
        sim_param : SimulationParameters
            Instance of simulation parameter class.
        partitions : list
            List of Partition objects. All partitions of the domain are collected here.
        normalization_factor : float
            Normalization multiplier to harmonize amplitudes between partitions.
        interface_data : list
            List of Interface objects. All interfaces of the domain are collected here.
        mics : list
            List of Microphone objects. All microphones placed within the domain are collected here.
        '''

        # Parameter class instance (SimulationParameters)
        self.sim_param = sim_param

        # List of partition data (PartitionData objects)
        self.part_data = partitions

        # List of interfaces (InterfaceData objects)
        self.interface_data = interface_data
        
        if interface_data[0].looped:
            self.interfaces = Interface3DLooped(sim_param, partitions)
        else:
            self.interfaces = Interface3D(sim_param, partitions)

        # Initialize & position mics.
        self.mics = mics

        self.normalization_factor = normalization_factor

    def preprocessing(self):
        '''
        Preprocessing stage. Refers to Step 1 in the paper.
        '''

        for i in range(len(self.part_data)):
            self.part_data[i].preprocessing()


    def record_to_mic(self, mic: Microphone, t_s: int):
        '''
        Records a given position in the domain to a given microphone.

        Parameters
        ----------
        mic : Microphone
            Microphone object instance which is placed inside the domain.
        t_s : float
            Current time step of the ARD simulation.
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

        Notification.notify("The ARD Simulation has started. Check terminal for progress end ETA.", "pytARD: Simulation started")

        for t_s in tqdm(range(self.sim_param.number_of_samples)):
            # Interface Handling
            for interface in self.interface_data:
                self.interfaces.handle_interface(interface)

            # Reverberation generation sensation
            for i in range(len(self.part_data)):
                self.part_data[i].simulate(t_s, self.normalization_factor)

                # Microphone handling
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

        if self.sim_param.verbose:
            print(f"Simulation completed successfully.\n")
        
        Notification.notify("The ARD Simulation has completed successfully.", "pytARD: Simulation completed")



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
