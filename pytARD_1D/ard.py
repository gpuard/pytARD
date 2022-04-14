import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import idct, dct
from common.microphone import Microphone as Mic

from pytARD_1D.interface import Interface1D


class ARDSimulator:
    '''
    ARD Simulation class. Creates and runs ARD simulator instance.
    '''

    def __init__(self, sim_param, part_data, interface_data=[], mics=[]):
        '''
        Create and run ARD simulator instance.

        Parameters
        ----------
        sim_param : SimulationParameters
            Instance of simulation parameter class.
        part_data : list
            List of PartitionData objects.
        '''

        # Parameter class instance (SimulationParameters)
        self.sim_param = sim_param

        # List of partition data (PartitionData objects)
        self.part_data = part_data

        self.interface_data = interface_data
        self.interfaces = Interface1D(sim_param, part_data)

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
        for t_s in range(2, self.sim_param.number_of_samples):
            for interface in self.interface_data:
                self.interfaces.handle_interface(interface)

            for i in range(len(self.part_data)):
                #print(f"nu forces: {self.part_data[i].new_forces}")
                # Execute DCT for next sample
                self.part_data[i].forces = dct(self.part_data[i].new_forces, n=self.part_data[i].space_divisions, type=1)

                # Updating mode using the update rule in equation 8.
                # Relates to (2 * F^n) / (ω_i ^ 2) * (1 - cos(ω_i * Δ_t)) in equation 8.
                self.part_data[i].force_field = ((2 * self.part_data[i].forces.reshape([self.part_data[i].space_divisions, 1])) / (
                    (self.part_data[i].omega_i) ** 2)) * (1 - np.cos(self.part_data[i].omega_i * self.sim_param.delta_t))

                # Relates to M^(n+1) in equation 8.
                self.part_data[i].M_next = 2 * self.part_data[i].M_current * \
                np.cos(self.part_data[i].omega_i * self.sim_param.delta_t) - self.part_data[i].M_previous + self.part_data[i].force_field
                
                # Convert modes to pressure values using inverse DCT.
                self.part_data[i].pressure_field = idct(self.part_data[i].M_next.reshape(
                    self.part_data[i].space_divisions), n=self.part_data[i].space_divisions, type=1)
                
                # Record signal with mics
                for m_i in range(len(self.mics)):
                    p_num = self.mics[m_i].partition_number                    
                    self.mics[m_i].record(self.part_data[p_num].pressure_field[int(self.part_data[p_num].space_divisions * (self.mics[m_i].location / self.part_data[p_num].dimensions))], t_s)
                
                # Update time stepping to prepare for next time step / loop iteration.
                self.part_data[i].M_previous = self.part_data[i].M_current.copy()
                self.part_data[i].M_current = self.part_data[i].M_next.copy()

                # Update impulses
                self.part_data[i].new_forces = self.part_data[i].impulses[t_s].copy()



'''
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMWWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMWKk0XWWWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWWMMMMMMMMMMMM
MMMMWk'..,;coONWWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWWMMMMMMMMMMMMM
MMMMWXc      cKNNNWWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM0lkWMMMMMMMMMMMM
MMMMMM0'    .xNNNNNNNWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWMMWWWO' .kWMMMMMMMMMMM
MMMMMMWx.   :KNNNNNNXXNWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWWWKo.   .kWMMMMMMMMMM
MMMMMMMWd. .xNNNNNNNNNXNWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWXxc.      ,KMMMMMMMMMM
MMMMMMMMNd:kXNNNNNNNNNNXXWWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWWWNKo.        .xMMMMMMMMMM
MMMMMMMMWNXXNNNNNNNNNNNNNNNWWWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWNNNNNNNX0kdl:,..  lNWMMMMMMMM
MMMMMMMMMMWXXNNNNNNNNNNNNNNNNNWWMMMMMMMMMMMMMMMMMMMMMMMWWNNNNNWWWMMMMMMMMMMMMMMMMMMMMMMMWWNXXNNNNNNNNNNNXKOdcdNWMMMMMMMM
MMMMMMMMMMWWNXXNNNNNNNNNNNNNNXNNNWWMMMMMMMMMMMMMMMMMMMWNXXXXXXXXNNNWWMMMMMMMMMMMMMMMMMMMNNNNNNNNNNNNNNNNNNNNXNWWMMMMMMMM
MMMMMMMMMMMMWWNNNNNNNNNNNNNNNNNNNNNNWWMMMMMMMMMMMMMMMWNNNNNNNNNNNXXNNNWMMMMMMMMMMMMMMMMWNNNNNNNNNNNNNNNNNNNNNWWWMMMMMMMM
MMMMMMMMMMMMMMWWNXNNNNNNNNNNNNNNNNNNNNWWMMWWWNNNNNNNNNXNNNNNNNNNNNNNNXNWMMMMMMMMMMMMMMWNXNNNNNNNNNNNNNNNNNNNWWMMMMMMMMMM
MMMMMMMMMMMMMMMMWNXNNNNNNNNNNNNNNNNNNNNNWWNXXNNXNNNNNNNNNNNNNNNNNNNNNNNNWMMMMMMMMMMMWWNNNNNNNNNNNNNNNNNNNNNWMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMWWWNNXXNNNNNNNNNNNNNNNXXXXNNNNNNNNNNNNNNNNNNNNNNNNNNNNXNNWMMMMMMMMWNNNNNNNNNNNNNNNNNNNNNNWMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMWWWNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNWWWWWWNXNNNNNNNNNNNNNNNNNNXNWMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMWNXXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNXNNNNNNNNNNNNNNNNNNNNNNNNNXNMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMWWWNNNNXXXXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNXXXNNNNNNNNNNNNNNNNNNNNXNWMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMWWNXXNNNNNXXKXNNNNNNNNNNXXXXXXNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNXNWMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMWNNXNNNNXXXXXXXXNNNNNNNNNXXXXXXXXXXXNNNNNNNNNNNXXXXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNXXXNNNWWWMMMMMMMMMMM
MMMMMMMMMMMMMMMMMWNXNNNXXXXNNXXXXNNNNNNNNNXNNNWWNXXNNXXXNNNNNNNXXXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNXXXNNNNXXNNNWWWMMMMMM
MMMMMMMMMMMMMMMMWNNNNNNXXXXXXXXNNNNNNNNNNNNWWMMMMWWNNNXXXNNNNNXXNWMMMMMMMMWWWNNNNNNNNNNNNNNNNNNNNNNXXXXNNNNNNNNNNNWMMMMM
MMMMMMMMMMMMMMMWNXXNNXXXXXXXNNNNNNNNNNNNWWMMMMMMMMMWWNXKKNNNXXKXWMMMMMMMMMMMMMWNNNNNNNNNNNNNNNNNNNNNXXXXNNNNNNNNXXNWMMMM
MMMMMMMMMMMMMMMNXNNNXXKXNNNNNNNNNNNNNWWMWMWMMWWNXNWMMWWXXNNNNXXNMMMMMMMMMMMMMMMMWNNNNNNNNNNNNNNNNNNNNNXXXXNNNNNNNNNNWMMM
MMMMMMMMMMMMMMWNXNNXXXXNNNNNNNNNNNNWMMMMWWWNKOkkxONMMMMNXNNNNNXWMWWNXKXWMMMMMMMMWWNNNNNNNNNNNNNNNNNNNNNXXXXXNNNNNNNNWMMM
MMMMMMMMMMMMMWNXNNXXXKXNNNNNNNNNXNWMMMMMWNKkocloxkXMMMMWXXNNNXXWWKOkkxxOKNWMMMMMMWNNNNNNNNNNNNNNNNNNNNNXXNNXXNNNNNNNWMMM
MMMMMMMMMMMMWNXNNXXXNXXNNNNNNNNXNWMMMMMMN0o,.  .lkKWMMMWNXNNNNNMWOxxo;,lxOXWMMMMMWNNNNNNNNNNNNNNNNNNNNNXXNNNXXXNNNNNWMMM
MMMMMMMMMWMWNXNNXXNNXXXNNNNNNNNXNWMMMMMWKx;    .lkKWMMMMWNNNNNWMXkxx;  .lxkXWWMMMWNNNNNNNNNNNNNNNNNNNNNNXXNNNXXXNNNNNMMM
MMMMMMMMWWMNXNNXXNNNXXNNNNNNNNNNNWMMMMMW0d'    :xkKWWMMMMMMMMMMMNkxx,  .cxxKWMMMMMNXNNNNNNNNNNNNNNNNNNNNXXNNNNXXNNNXNMMM
MMMMMMMMMMWNXNXXNNNNXXNNNNNNNNNNNWMMMMMWKx;   'oxkXWMMMMMMMMMMMMNOxx:  'oxkKWMMMMMNXNNNNNNNNNNNNNNNNNNNNXXNNNNXXNNNXNMMM
MMMMMMMMMWNXNNXXNNNNXXNNNNNNNNNNNWMMMMMMXOdl;:oxkXWMMMMMMMMMMMMMW0xxdc:oxxONWMMMMMNXNNNNNNNNNNNNNNNNNNNNXXNNNNNXXNNXNMMM
MMMMMMMMMWNNNNXXNNNNXKNNNNNNNNNXXWMMMMMMWNXKOkO0NWWMMWWMMMMMMMMMWXOkkxxxkOXWMMMMMMNNNNNNNNNNNNNNNNNNNNNNXXNNNNNXXNNXNMMM
MMMMMMMMMWNNNNXXNNNNXKXXOxdxk0KXXNWWMMMMMMMMWWWWWXOdoc:c:lOWMMWWMWWNNXXXNWMMMMMMMWNXKOxooodkKNNNNNNNNNNNXXNNNNNXXNNXNMMM
MMMMMMMMMWXNNNXXNNNNXXXx:;,;;:cd0NWMWMMMMMMMMMM0c.        .lx0NWMMMMMMMMMMMMMMMMMWNXx:;;,;;:kNNNNNNNNNNNXXNNNNNXXNNXNMMM
MMMMMMMMMWNNNNXXNNNNXKKd;;;;;;;l0XNWWWMWWNXNNNNx'            .:ONNNNNNNNNNNWWWMWWNXKd;;;;;;;xXNNNNNNNNNXXXNNNNNXXNNXNMMM
MMMMMMMMMWXNNNXXXNNNXXX0o:,;;;:xXNXNNXXK0OOOOOOOxolc;,.      .cxOOOOOOOOOOOO00KKKKKXOc;;;,,:kXNNNNNNNNNXXNNNNNNXXNNXNMMM
MMMMMMMMMWNXNNNXXNNNNXXXKkdoookXXXK0OOOOOOOOOOOOOOOOOOkoc;;ldxOOOOOOOOOOOOOOOOOOOkO000kdoodOXNNNNNNNNNXXXNNNNXXXNNNNWMMM
MMMMMMMMMWWNXNNNXXXXNNNXXNNNXXXX0OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOkxddddkkkOOOOOOOO0XNNXXNNNNNNNNNXXXXNNNNNXXXXNNXNMMMM
MMMMMMMMMMMWNNNNXXNNNNNXWWWWNNX0OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOkdoodxkxxxkOOOOOOKNNXXNNNNNNNNNXXXNNNNNNWNNNNXNWMMMM
MMMMMMMMMMMMMNXXNWWWNXXNWMMMWWX0kOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOkkxxkOOOOOkkkOOOOOOKNNNNXXXXNNNNWWNXNNNWWWWNNXNNWMMMMM
MMMMMMMMMMMMMWNWMMMMWNWMMMMMMMMNKOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOkxxxkOOOOOOOOOOOOOOO0KNNNNNNNWWWMMMMWWWMMMMMMWWNWWMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWN0OOOOOOOOOOOOOOOOOOOOOOOOOOOOOkkOOOOOOOOO000KKXNNNWWWWWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWNXXKK00OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOKNNWWWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWWNXK00OOOOOOOOOOOOOOOOOOOO000KXNWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWNNXXKXXXXXXXXXXNNNNNNWWWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM

'''
