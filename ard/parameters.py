import numpy as np

class ARDParamters:
    '''
    Parameter container class for ARD simulator.
    '''
    def __init__(self, room_size, max_simulation_frequency, T, spatial_samples_per_wave_length=4, c=343, Fs=8000, verbose=False, visualize=False):
        '''
        Instantiates an ARD simulation session.

        Parameters
        ----------
        room_size : ndarray
            Size of the room in meters. Can be 1D, 2D or 3D.
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
        self.max_simulation_frequency = max_simulation_frequency

        # Array, which stores air pressure at each given point in time in the voxelized grid
        self.pressure_field = None

        # Array for pressure field results (auralisation and visualisation)
        self.pressure_field_results = []

        self.c = c
        self.Fs = Fs

        # Calculating the number of samples the simulation takes.
        self.number_of_samples = T * Fs

        # Calculate time stepping (Δ_t)
        self.delta_t = T / self.number_of_samples

        # Voxel grid spacing. Changes according to frequency
        self.H = self.calculate_voxelization_step(
            spatial_samples_per_wave_length)
        if verbose:
            print(f"H = {self.H}")

        # Longest room dimension length dividied by H (voxel grid spacing).
        self.space_divisions = int(np.max(room_size) / self.H)

        # Instantiate impulse array which keeps track of impulses in space over time.
        self.impulses = np.zeros(
            shape=[self.number_of_samples, self.space_divisions])
        self.impulse_location = 0  #  TODO Put into constructor/parameter class
        self.dirac_a = 0.1  #  TODO Put into constructor/parameter class

        self.verbose = verbose
        self.visualize = visualize
