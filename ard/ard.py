import numpy as np
from scipy.fftpack import idct, dct

class ARDSimulator:

    C = 343  #  Speed of sound [m/s]

    def __init__(self, room_size, max_simulation_frequency, spatial_samples_per_wave_length=4):
        '''
        Instantiates an ARD simulation session.

        Parameters
        ----------
        room_size : ndarray
            Size of the room in meters.
        max_simulation_frequency : float
            Uppermost frequency of simulation. Can be dialed in lower to enhance performance.
        spatial_samples_per_wave_length : int
            Number of spatial samples per wave length. Usually 2 to 4. Lower values decrease 
            resolution but enhances performance.
        '''
        assert(len(room_size) >= 1, "Room dimensions should be bigger than 1D.")
        assert(len(room_size) <= 3, "Room dimensions should be lower than 3D.")

        self.room_size = room_size
        self.max_simulation_frequency = max_simulation_frequency
        self.H = self.calculate_voxelization_step(
            spatial_samples_per_wave_length)
        self.pressure_field = None

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
            shape=[len(self.room_size), np.max(self.room_size) / self.H])

        # Step 1 b). Rectangular decomposition. Skipped as of now. TODO: Implement rectangular decomposition

        # Step 1 c). Precomputation for the DCTs to be performed. Skipped partitions as of now. TODO: Implement partitions

    def simulation(self):
        '''
        Simulation stage. Refers to Step 2 in the paper.

        Parameters
        ----------

        Returns
        -------
        '''
        pass

    @classmethod
    def pressure_field_calculation():
        pass

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
        return self.C / (spatial_samples_per_wave_length * self.max_simulation_frequency)
