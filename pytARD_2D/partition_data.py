import numpy as np
from scipy.io.wavfile import read
from scipy.fftpack import idct, dct


class PartitionData:
    def __init__(
        self,
        dimensions,
        sim_parameters,
        do_impulse=True,
        impulse_source=[0, 0]
    ):
        '''
        Parameter container class for ARD simulator. Contains all relevant data to instantiate
        and run ARD simulator.

        Parameters
        ----------
        dimensions : ndarray
            Size of the partition (room) in meters.
        sim_parameters : SimulationParameters
            Instance of simulation parameter class.
        do_impulse : bool
            Determines if the impulse is generated on this partition.
        '''
        self.dimensions = dimensions
        self.sim_param = sim_parameters

        # Longest room dimension length dividied by H (voxel grid spacing).
        # TODO Maybe do different X / Y space divisions?
        self.space_divisions_y = int(
            (dimensions[1]) * self.sim_param.spatial_samples_per_wave_length)
        self.space_divisions_x = int(
            (dimensions[0]) * self.sim_param.spatial_samples_per_wave_length)

        # Voxel grid spacing. Changes automatically according to frequency
        self.h_y = dimensions[1] / self.space_divisions_y
        self.h_x = dimensions[0] / self.space_divisions_y

        print(f"h_x = {self.h_x}")
        print(f"h_y = {self.h_y}")


        # Instantiate forces array, which corresponds to F in update rule (results of DCT computation). TODO: Elaborate more
        self.forces = None

        # Instantiate updated forces array. Combination of impulse and/or contribution of the interface.
        # DCT of new_forces will be written into forces. TODO: Is that correct?
        self.new_forces = None

        # Impulse array which keeps track of impulses in space over time.
        self.impulses = np.zeros(
            shape=[self.sim_param.number_of_samples, self.space_divisions_y, self.space_divisions_x])

        # Array, which stores air pressure at each given point in time in the voxelized grid
        self.pressure_field = None

        # Array for pressure field results (auralisation and visualisation)
        self.pressure_field_results = []

        # Fill impulse array with impulses.
        # TODO: Switch between different source signals via bool or enum? Also create source signal container
        if do_impulse:
            # Create indices for time samples. x = [1 2 3 4 5] -> sin(x_i * pi) -> sin(pi), sin(2pi) sin(3pi)
            time_sample_indices = np.arange(
                0, self.sim_param.number_of_samples, 1)

            # Amplitude of gaussian impulse
            # TODO: Position source via parameter
            A = 100000

            # TODO: Magic numbers! Bad!!!
            self.impulses[:, int(self.space_divisions_y / 2), int(self.space_divisions_x / 2)] = A * PartitionData.create_gaussian_impulse(time_sample_indices, 80 * 4, 80) - A * PartitionData.create_gaussian_impulse(time_sample_indices, 80 * 4 * 2, 80)

            if self.sim_param.visualize:
                import matplotlib.pyplot as plt
                plt.plot(self.impulses[:, int(self.space_divisions_y  / 2), int(self.space_divisions_x / 2)])
                plt.show()


        # Uncomment to inject wave file. TODO: Consolidize into source class
        '''
        if do_impulse:
            (fs, wav) = read('track.wav')
            self.impulses[:, int(self.space_divisions * 0.9)] = 100 * wav[0:self.sim_param.number_of_samples]
        '''

        if sim_parameters.verbose:
            print(f"Created partition with dimensions {self.dimensions} m\nℎ (y): {self.h_y}, ℎ (x): {self.h_x} | Space divisions: {self.space_divisions_y} ({self.dimensions/self.space_divisions_y} m each)")


    def preprocessing(self):
        '''
        Preprocessing stage. Refers to Step 1 in the paper.
        '''
        # Preparing pressure field. Equates to function p(x) on the paper.
        self.pressure_field = np.zeros(shape=[1, self.space_divisions_y, self.space_divisions_x])
        #print(f"presh field = {self.pressure_field}")

        # Precomputation for the DCTs to be performed. Transforming impulse to spatial forces. Skipped partitions as of now.
        self.new_forces = self.impulses[0].copy()

        # Relates to equation 5 and 8 of "An efficient GPU-based time domain solver for the
        # acoustic wave equation" paper.
        # For reference, see https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/4.pdf.

        self.omega_i = np.zeros(shape=[self.space_divisions_y, self.space_divisions_x, 1])
        for y in range(self.space_divisions_y):
            for x in range(self.space_divisions_x):
                self.omega_i[y, x, 0] = self.sim_param.c * ((np.pi ** 2) * (((x ** 2) / (self.dimensions[0] ** 2)) + ((y ** 2) / (self.dimensions[1] ** 2)))) ** 0.5

        # TODO Semi disgusting hack. Without it, the calculation of update rule (equation 9) would crash due to division by zero
        self.omega_i[0, 0] = 0.1

        # Update time stepping. Relates to M^(n+1) and M^n in equation 8.
        self.M_previous = np.zeros(shape=[self.space_divisions_y, self.space_divisions_x, 1])
        self.M_current = np.zeros(shape=self.M_previous.shape)
        self.M_next = None

        if self.sim_param.verbose:
            print(f"Shape of omega_i: {self.omega_i.shape}")
            print(f"Shape of pressure field: {self.pressure_field.shape}")

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
