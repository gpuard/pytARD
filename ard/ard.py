import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import idct, dct
from scipy.io.wavfile import read, write


class ARDSimulator:
    # TODO Maybe create parameter class?
    def __init__(self, parameters):
        '''
        Create and run ARD simulator instance.

        Parameters
        ----------
        parameters : object
            Instance of ARDSimulator parameter class.
        '''
        # Parameter class instance
        self.param = parameters

        # Array, which stores air pressure at each given point in time in the voxelized grid
        self.pressure_field = None

        # Array for pressure field results (auralisation and visualisation)
        self.pressure_field_results = []

        # Forces (result of DCT computation) TODO: Elaborate more
        self.forces = None

        # Instantiate impulse array which keeps track of impulses in space over time.
        self.impulses = np.zeros(
            shape=[self.param.number_of_samples, self.param.space_divisions])

        # Narrowness and strenghth of dirac impulse. # TODO Create source signal container
        self.dirac_a = 0.1

        # Fill impulse array with impulses.  TODO: Switch between gaussian and dirac maybe? Also create source signal container
        print(self.impulses.shape)
        impulse_index = int((self.param.space_divisions - 1) * (self.param.src_pos[0]))
        print(impulse_index)
        #self.impulses[:, impulse_index] = [ARDSimulator.create_normalized_dirac_impulse(
        #        self.dirac_a, t) for t in np.arange(0, self.param.T, self.param.delta_t)]
        time_sample_indices = np.arange(0, self.param.number_of_samples, 1) # =x = [1 2 3 4 5] ------> sin(x_i * pi) ->>> sin(pi), sin(2pi) sin(3pi)
        A = 100
        #self.impulses[:, 0] = A*ARDSimulator.create_gaussian_impulse(time_sample_indices, 80*4, 80) - A*ARDSimulator.create_gaussian_impulse(time_sample_indices, 80*4*2, 80)
        #self.impulses[:, 0] = A * (np.sin(10 * ((1 / self.param.Fs) * time_sample_indices * np.pi))) + 10E-18

        (fs, wav) = read('track.wav')
        self.impulses[:, int(self.param.space_divisions / 2)] = 100 * wav[0:self.param.number_of_samples]

    def preprocessing(self):
        '''
        Preprocessing stage. Refers to Step 1 in the paper.

        Parameters
        ----------

        Returns
        -------
        '''
        # Step 1 a). Preparing pressure field. Equates to function p(x) on the paper.
        self.pressure_field = np.zeros(
            shape=[len(self.param.room_size), self.param.space_divisions])

        # Step 1 b). Rectangular decomposition. Skipped as of now.
        # TODO: Implement rectangular decomposition

        # Step 1 c). Precomputation for the DCTs to be performed. Transforming impulse to spatial
        # forces. Skipped partitions as of now. TODO: Implement partitions
        self.forces = dct(self.impulses[1],
                          n=self.param.space_divisions, type=1)

    def simulation(self):
        '''
        Simulation stage. Refers to Step 2 in the paper.

        Parameters
        ----------

        Returns
        -------
        '''

        # Relates to equation 5 and 8 of "An efficient GPU-based time domain solver for the
        # acoustic wave equation" paper.
        # For reference, see https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/4.pdf.
        omega_i = self.param.c * np.pi * \
            (np.arange(0, self.param.space_divisions, 1) / np.max(self.param.room_size))

        # Convert omega_i from row vector to column vector
        omega_i = omega_i.reshape([len(omega_i), 1])

        # Update time stepping. Relates to M^(n+1) and M^n in equation 8.
        # TODO Number of partitions missing atm
        M_previous = np.zeros(shape=[self.param.space_divisions, 1])
        M_current = np.zeros(shape=M_previous.shape)

        # Force field in spectral room. Temporary variable, right part after + sign of equation 8.
        force_field = np.zeros(shape=M_previous.shape) # TODO: force field isn't used as of yet!

        if self.param.verbose:
            print(f"Shape of omega_i: {omega_i.shape}")
            print(f"Shape of pressure field: {self.pressure_field.shape}")

        self.mic = np.zeros(shape=self.param.number_of_samples, dtype=np.float)

        # t_s: Time stepping.
        for t_s in range(2, self.param.number_of_samples):
            # Updating mode using the update rule in equation 8.
            # Relates to (2 * F^n) / (ω_i ^ 2) * (1 - cos(ω_i * Δ_t)) in equation 8.
            force_field = ((2 * self.forces.reshape([self.param.space_divisions, 1])) / (
                (omega_i + 0.00000001) ** 2)) * (1 - np.cos(omega_i * self.param.delta_t))
            # TODO Perhaps set zero element to zero in force field if something goes horribly wrong

            # Relates to M^(n+1) in equation 8.
            M_next = 2 * M_current * \
                np.cos(omega_i * self.param.delta_t) - M_previous + force_field

            # Convert modes to pressure values using inverse DCT.
            self.pressure_field = idct(M_next.reshape(
                self.param.space_divisions), n=self.param.space_divisions, type=1)

            self.pressure_field_results.append(self.pressure_field.copy())
            self.mic[t_s] = self.pressure_field[int(self.param.space_divisions * .75)]

            # Update time stepping to prepare for next time step / loop iteration.
            M_previous = M_current.copy()
            M_current = M_next.copy()

            # Execute DCT for next sample
            self.forces = dct(
                self.impulses[t_s], n=self.param.space_divisions, type=1)

        self.mic = self.mic / np.max(self.mic)
        write("impulse_response.wav", self.param.Fs, self.mic.astype(np.float))



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

    @staticmethod
    def create_dirac_impulse(a, x):
        '''
        Creates a dirac impulse, an infinitely thin and strong impulse.
        For reference, see https://en.wikipedia.org/wiki/Dirac_delta_function.
        TODO: Maybe create alternate method for gaussian white noise?

        Parameters
        ----------
        a : float
            Height / narrowness of impulse. The higher the value, the higher and narrower the
            impulse.
        x : float
            Location coordinate
        Returns
        -------
        float
            δ(x), the calculated dirac impulse.
        '''
        return (1 / (np.sqrt(np.pi) * a)) * (np.exp(-((x ** 2) / (a ** 2))))

    @staticmethod
    def create_normalized_dirac_impulse(a, x):
        '''
        Generate dirac impulse which would have a peak of y=1 at x=0
        Parameters
        ----------
        a : float
            Height / narrowness of impulse. The higher the value, the higher and narrower the
            impulse.
            Note that the impulse height is normalized to 1 at its peak.
        x : float
            Location coordinate
        Returns
        -------
        float
            δ(x), the calculated dirac impulse.
        '''
        imp = ARDSimulator.create_dirac_impulse(a, x)
        at_zero = ARDSimulator.create_dirac_impulse(a, 0)
        return (imp / at_zero)
    
    @staticmethod
    def create_gaussian_impulse(x, mu, sigma):
        '''
        Generate gaussian impulse
        Parameters
        ----------
        Returns
        -------
        '''
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

    


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
