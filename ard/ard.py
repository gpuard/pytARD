import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import idct, dct


class ARDSimulator:

    # TODO Maybe create parameter class?
    def __init__(self, room_size, src_pos, max_simulation_frequency, T, spatial_samples_per_wave_length=4, c=343, Fs=8000, verbose=False, visualize=False):
        '''
        Instantiates an ARD simulation session.

        Parameters
        ----------
        room_size : ndarray
            Size of the room in meters. Can be 1D, 2D or 3D.
        src_pos : ndarray
            Location of signal source inside the room. Can be 1D, 2D or 3D.
        max_simulation_frequency : float
            Uppermost frequency of simulation. Can be dialed in lower to enhance performance.
        T : float
            Simulation time [s].
        spatial_samples_per_wave_length : int
            Number of spatial samples per wave length. Usually 2 to 4. Lower values decrease 
            resolution but enhances performance.
        c : float
            Speed of sound [m/s]. Depends on air temperature, pressure and humidity. 
        Fs : int
            Sampling rate. The higher, the more fidelity but lower performance.
        verbose : boolean
            Prints information on the terminal for debugging and status purposes.
        visualize : boolean
            Visualizes wave propagation in a plot.
        '''
        assert(len(room_size) >= 1), "Room dimensions should be bigger than 1D."
        assert(len(room_size) <= 3), "Room dimensions should be lower than 3D."

        self.room_size = room_size
        self.src_pos = src_pos
        self.max_simulation_frequency = max_simulation_frequency

        # Array, which stores air pressure at each given point in time in the voxelized grid
        self.pressure_field = None

        # Array for pressure field results (auralisation and visualisation)
        self.pressure_field_results = []

        self.c = c
        self.Fs = Fs
        self.T = T

        # Calculating the number of samples the simulation takes.
        self.number_of_samples = T * Fs

        # Calculate time stepping (Δ_t)
        self.delta_t = T / self.number_of_samples

        # Voxel grid spacing. Changes according to frequency
        #self.H = self.calculate_voxelization_step(
       #     spatial_samples_per_wave_length)
        
        self.H = 0.1
        
        if verbose:
            print(f"H = {self.H}")

        # Longest room dimension length dividied by H (voxel grid spacing).
        self.space_divisions = int(np.max(room_size) / self.H)

        # Instantiate impulse array which keeps track of impulses in space over time.
        self.impulses = np.zeros(
            shape=[self.number_of_samples, self.space_divisions])
        self.impulse_location = 0  #  TODO Put into constructor/parameter class
        self.dirac_a = 0.1  #  TODO Put into constructor/parameter class

        # Fill impulse array with impulses.  TODO: Switch between gaussian and dirac maybe?
        self.impulses[:, int((self.space_divisions - 1) * (src_pos[0] / room_size[0]))] = [ARDSimulator.create_normalized_dirac_impulse(
            self.dirac_a, t) for t in np.arange(0, T, self.delta_t)]

        self.verbose = verbose
        self.visualize = visualize

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
            shape=[len(self.room_size), self.space_divisions])

        # Step 1 b). Rectangular decomposition. Skipped as of now.
        # TODO: Implement rectangular decomposition

        # Step 1 c). Precomputation for the DCTs to be performed. Transforming impulse to spatial
        # forces. Skipped partitions as of now. TODO: Implement partitions
        self.forces = dct(self.impulses[1], n=self.space_divisions, type=1)

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
        omega_i = self.c * np.pi * (np.linspace(0, np.max(self.room_size), self.space_divisions) / np.max(self.room_size))

        # Convert omega_i from row vector to column vector
        omega_i = omega_i.reshape([len(omega_i), 1])

        # Update time stepping. Relates to M^(n+1) and M^n in equation 8.
        # TODO Number of partitions missing atm
        M_previous = np.zeros(shape=[self.space_divisions, 1])
        M_current = np.zeros(shape=M_previous.shape)

        # Force field in spectral room. Temporary variable, right part after + sign of equation 8.
        force_field = np.zeros(shape=M_previous.shape)

        if self.verbose:
            print(f"Shape of omega_i: {omega_i.shape}")
            print(f"Shape of pressure field: {self.pressure_field.shape}")

        # t_s: Time stepping.
        for t_s in range(2, self.number_of_samples):
            # Updating mode using the update rule in equation 8.
            # Relates to (2 * F^n) / (ω_i ^ 2) * (1 - cos(ω_i * Δ_t)) in equation 8.
            force_field = ((2 * self.forces.reshape([self.space_divisions, 1])) / (
                (omega_i + 0.00000001) ** 2)) * (1 - np.cos(omega_i * self.delta_t))
            # TODO Perhaps set zero element to zero in force field if something goes horribly wrong

            # Relates to M^(n+1) in equation 8.
            M_next = 2 * M_current * \
                np.cos(omega_i * self.delta_t) - M_previous + force_field

            # Convert modes to pressure values using inverse DCT.
            self.pressure_field = idct(M_next.reshape(
                self.space_divisions), n=self.space_divisions, type=1)
            
            self.pressure_field_results.append(self.pressure_field.copy())

            # Update time stepping to prepare for next time step / loop iteration.
            M_previous = M_current.copy()
            M_current = M_next.copy()

            # Execute DCT for next sample
            self.forces = dct(self.impulses[t_s], n=self.space_divisions, type=1)

    def calculate_voxelization_step(self, spatial_samples_per_wave_length):
        '''
        Calculate voxelization step for the segmentation of the room (voxelizing the scene).
        The cell size is fixed by the chosen wave length and the number of spatial samples per 
        wave length.
        Calculation: ℎ <= 𝑐 / (2 * 𝑓_max)

        Parameters
        ----------
        spatial_samples_per_wave_length : int
            Number of spatial samples per wave length. Usually 2 to 4.

        Returns
        -------
        float
            ℎ, the voxelization step. In numerics and papers, it's usually referred to ℎ. 
        '''
        return self.c / (spatial_samples_per_wave_length * self.max_simulation_frequency)

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
    def update_rule(M, omega_i, delta_t, Fn):
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
