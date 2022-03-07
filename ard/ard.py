import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import idct, dct


class ARDSimulator:
    # TODO Maybe create parameter class?
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

        self.coeffs = [
            [  0.0,   0.0,   -2.0,    2.0,   0.0,  0.0 ],
            [  0.0,  -2.0,   27.0,  -27.0,   2.0,  0.0 ],
            [ -2.0,  27.0, -270.0,  270.0, -27.0,  2.0 ],
            [  2.0, -27.0,  270.0, -270.0,  27.0, -2.0 ],
            [  0.0,   2.0,  -27.0,   27.0,  -2.0,  0.0 ],
            [  0.0,   0.0,    2.0,   -2.0,   0.0,  0.0 ] ]

        

    def preprocessing(self):
        '''
        Preprocessing stage. Refers to Step 1 in the paper.

        Parameters
        ----------

        Returns
        -------
        '''
        for i in range(len(self.part_data)):
            self.part_data[i].preprocessing()
            
    def s(self,j,h):
        coef = np.array([2, -27, 270, -490, 270, -27, 2])
        return 1/(180*h**2) * coef[j+3]
        
    def simulation(self):
        '''
        Simulation stage. Refers to Step 2 in the paper.

        Parameters
        ----------

        Returns
        -------
        '''

        for t_s in range(2, self.sim_param.number_of_samples):
            for i in range(len(self.part_data)):
               
                # Updating mode using the update rule in equation 8.
                # Relates to (2 * F^n) / (ω_i ^ 2) * (1 - cos(ω_i * Δ_t)) in equation 8.
                                
                self.part_data[i].force_field = ((2 * self.part_data[i].forces.reshape([self.part_data[i].space_divisions, 1])) / (
                (self.part_data[i].omega_i + 0.00000001) ** 2)) * (1 - np.cos(self.part_data[i].omega_i * self.sim_param.delta_t))
                # TODO Perhaps set zero element to zero in force field if something goes horribly wrong
                
                # Relates to M^(n+1) in equation 8.
                self.part_data[i].M_next = 2 * self.part_data[i].M_current * \
                np.cos(self.part_data[i].omega_i * self.sim_param.delta_t) - self.part_data[i].M_previous + self.part_data[i].force_field
                
                # Convert modes to pressure values using inverse DCT.
                self.part_data[i].pressure_field = idct(self.part_data[i].M_next.reshape(
                self.part_data[i].space_divisions), n=self.part_data[i].space_divisions, type=1)
                
                self.part_data[i].pressure_field_results.append(self.part_data[i].pressure_field.copy())
                #self.mic[t_s] = self.part_data[i].pressure_field[int(self.part_data[i].space_divisions * .75)]
                
                # Update time stepping to prepare for next time step / loop iteration.
                self.part_data[i].M_previous = self.part_data[i].M_current.copy()
                self.part_data[i].M_current = self.part_data[i].M_next.copy()


            # INTERFACE HANDLING
            for i in range(-3,3):
                left_sum = 0.
                right_sum = 0.
                for l in range(3):
                    left_sum += self.coeffs[i + 3][l] * self.part_data[0].pressure_field[-3+l]
                
                for r in range(3,6):
                    right_sum += self.coeffs[i + 3][r] * self.part_data[1].pressure_field[-3+r]
                
                if t_s < self.sim_param.number_of_samples - 1:
                    if i < 0:
                        #right to left
                        Fi = right_sum
                        self.part_data[0].impulses[t_s][i] = Fi * self.sim_param.c**2 / (180. * self.part_data[0].h**2)
                    else:
                        #left to right
                        Fi = left_sum
                        self.part_data[1].impulses[t_s][i] = Fi * self.sim_param.c**2 / (180. * self.part_data[1].h**2)
                
            for i in range(len(self.part_data)):
                # Execute DCT for next sample
                self.part_data[i].forces = dct(self.part_data[i].impulses[t_s], n=self.part_data[i].space_divisions, type=1)
            

        #self.mic = np.zeros(shape=self.sim_param.number_of_samples, dtype=np.float)

        

        #self.mic = self.mic / np.max(self.mic)
        #write("impulse_response.wav", self.sim_param.Fs, self.mic.astype(np.float))
        
       # 1. For all interfaces: Interface handling 
       # to compute force f within each partition (Equation 12).
           
    @staticmethod
    def update_rule(M, omega_i, delta_t, Fn): # TODO Offload update rule here or scrap this function
        '''
        Relates to equation 8 of "An efficient GPU-based time domain solver for the acoustic wave 
        equation" paper.
        For reference, see https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/4.pdf.

        Parameters
        ----------
        M : ?
            ?
        omega_i : float
            Speed of sound (C) times pi.
        delta_t : float
            Time deviation [s].
        Fn : ?
            ?

        Returns
        -------
        float
            M_i ^ (n + 1)
        '''
        # TODO: Maybe use this?

    

    


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
