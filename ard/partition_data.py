import numpy as np
from scipy.io.wavfile import read, write
from scipy.fftpack import idct, dct

class PartitionData:
    def __init__(
        self,
        dimensions,
        sim_param
    ):
        '''
        Parameter container class for ARD simulator. Contains all relevant data to instantiate
        and run ARD simulator.

        Parameters
        ----------
        dimensions : ndarray
            Size of the partition (room) in meters. Can be 1D, 2D or 3D.
        spatial_samples_per_wave_length : int
            Samples per wave length. Determines resolution and quality of wave sampling
        '''
        self.dimensions = dimensions
        '''
                assert(self.room_size.ndim >=
               1), "Room dimensions should be bigger than 1D."
        assert(self.room_size.ndim <=
               3), "Room dimensions should be lower than 3D."

        '''
        self.sim_param = sim_param

        # Longest room dimension length dividied by H (voxel grid spacing).
        self.space_divisions = int(np.max(dimensions) * self.sim_param.spatial_samples_per_wave_length)

        # Forces (result of DCT computation) TODO: Elaborate more
        self.forces = None

        # Instantiate impulse array which keeps track of impulses in space over time.
        self.impulses = np.zeros(
            shape=[self.sim_param.number_of_samples, self.space_divisions])

        # Array, which stores air pressure at each given point in time in the voxelized grid
        self.pressure_field = None

        # Array for pressure field results (auralisation and visualisation)
        self.pressure_field_results = []

        # Narrowness and strenghth of dirac impulse. # TODO Create source signal container
        self.dirac_a = 0.1

        # Fill impulse array with impulses.  TODO: Switch between gaussian and dirac maybe? Also create source signal container
        impulse_index = int((self.space_divisions - 1) * (self.sim_param.src_pos[0]))
        #self.impulses[:, impulse_index] = [ARDSimulator.create_normalized_dirac_impulse(
        #        self.dirac_a, t) for t in np.arange(0, self.param.T, self.param.delta_t)]
        time_sample_indices = np.arange(0, self.sim_param.number_of_samples, 1) # =x = [1 2 3 4 5] ------> sin(x_i * pi) ->>> sin(pi), sin(2pi) sin(3pi)
        A = 100
        self.impulses[:, 0] = A*PartitionData.create_gaussian_impulse(time_sample_indices, 80*4, 80) - A*PartitionData.create_gaussian_impulse(time_sample_indices, 80*4*2, 80)
        #self.impulses[:, 0] = A * (np.sin(10 * ((1 / self.param.Fs) * time_sample_indices * np.pi))) + 10E-18

        #(fs, wav) = read('track.wav')
        #self.impulses[:, int(self.sim_param.space_divisions / 2)] = 100 * wav[0:self.sim_param.number_of_samples]

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
            shape=[len(self.dimensions), self.space_divisions])

        # Step 1 b). Rectangular decomposition. Skipped as of now.
        # TODO: Implement rectangular decomposition

        # Step 1 c). Precomputation for the DCTs to be performed. Transforming impulse to spatial
        # forces. Skipped partitions as of now. TODO: Implement partitions
        self.forces = dct(self.impulses[1],
                          n=self.space_divisions, type=1)

        # Relates to equation 5 and 8 of "An efficient GPU-based time domain solver for the
        # acoustic wave equation" paper.
        # For reference, see https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/4.pdf.
        self.omega_i = self.sim_param.c * np.pi * \
            (np.arange(0, self.space_divisions, 1) / np.max(self.dimensions))

        # Convert omega_i from row vector to column vector
        self.omega_i = self.omega_i.reshape([len(self.omega_i), 1])

        # Update time stepping. Relates to M^(n+1) and M^n in equation 8.
        # TODO Number of partitions missing atm
        self.M_previous = np.zeros(shape=[self.space_divisions, 1])
        self.M_current = np.zeros(shape=self.M_previous.shape)
        self.M_next = None
        
        # Force field in spectral room. Temporary variable, right part after + sign of equation 8.
        self.force_field = np.zeros(shape=self.M_previous.shape) # TODO: force field isn't used as of yet!

        if self.sim_param.verbose:
            print(f"Shape of omega_i: {self.omega_i.shape}")
            print(f"Shape of pressure field: {self.pressure_field.shape}")
        
    
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

    # @staticmethod
    # def create_normalized_dirac_impulse(a, x):
    #     '''
    #     Generate dirac impulse which would have a peak of y=1 at x=0
    #     Parameters
    #     ----------
    #     a : float
    #         Height / narrowness of impulse. The higher the value, the higher and narrower the
    #         impulse.
    #         Note that the impulse height is normalized to 1 at its peak.
    #     x : float
    #         Location coordinate
    #     Returns
    #     -------
    #     float
    #         δ(x), the calculated dirac impulse.
    #     '''
    #     imp = ARDSimulator.create_dirac_impulse(a, x)
    #     at_zero = ARDSimulator.create_dirac_impulse(a, 0)
    #     return (imp / at_zero)
    
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


