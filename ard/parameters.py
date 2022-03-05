import numpy as np


class ARDParameters:
    def __init__(
        self,
        room_size,
        src_pos,
        max_simulation_frequency,
        T,
        spatial_samples_per_wave_length=4,
        c=343,
        Fs=8000,
        verbose=False,
        visualize=False
    ):
        '''
        Parameter container class for ARD simulator. Contains all relevant data to instantiate
        and run ARD simulator.

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

        self.room_size = [np.max(room_size) / spatial_samples_per_wave_length]
        self.src_pos = src_pos
        self.max_simulation_frequency = max_simulation_frequency
        self.c = c
        self.Fs = Fs
        self.T = T

        # Calculating the number of samples the simulation takes.
        self.number_of_samples = int(T * Fs)

        # Calculate time stepping (Δ_t)
        self.delta_t = T / self.number_of_samples

        # Voxel grid spacing. Changes according to frequency
        self.H = ARDParameters.calculate_voxelization_step(
            self.c, spatial_samples_per_wave_length, self.max_simulation_frequency)

        # Longest room dimension length dividied by H (voxel grid spacing).
        self.space_divisions = int(np.max(room_size) * spatial_samples_per_wave_length)

        self.impulse_location = 0

        self.verbose = verbose
        self.visualize = visualize

    @staticmethod
    def calculate_voxelization_step(c, spatial_samples_per_wave_length, max_simulation_frequency):
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
        return c / (spatial_samples_per_wave_length * max_simulation_frequency)
