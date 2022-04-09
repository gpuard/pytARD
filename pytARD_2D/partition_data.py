import numpy as np
from scipy.io.wavfile import read
from scipy.fftpack import idct, dct
from pytARD_2D.impulse import Impulse


class PartitionData:
    #air-partition
    def __init__(
        self,
        dimensions,
        sim_parameters,
        do_impulse = True,
        source_location = (0, 0),
        impulse=None
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
        impulse : Impulse
            Determines if the impulse is generated on this partition, and which kind of impulse. 
        '''
        self.dimensions = np.array(dimensions)
        self.sim_param = sim_parameters

        # Longest room dimension length dividied by H (voxel grid spacing).
        self.space_divisions_y = int(
            (dimensions[1]) * self.sim_param.samples_per_wave_length)
        self.space_divisions_x = int(
            (dimensions[0]) * self.sim_param.samples_per_wave_length)
                
        # Voxel grid spacing. Changes automatically according to frequency
        self.h_y = sim_parameters.dx
        self.h_x = sim_parameters.dy

        self.dx = sim_parameters.dx
        self.dy = sim_parameters.dy

        self.grid_x = np.arange(0,self.dimensions[0],self.dx)
        self.grid_y = np.arange(0,self.dimensions[1],self.dy)    
        
        self.grid_shape = (len(self.grid_x),len(self.grid_y))

        print(f"dx = {self.dx}")
        print(f"dy = {self.dy}")

        # Instantiate forces array, which corresponds to F in update rule (results of DCT computation). TODO: Elaborate more
        self.forces = None
        # f_spectral = None
        
        # Instantiate updated forces array. Combination of impulse and/or contribution of the interface.
        # DCT of new_forces will be written into forces. TODO: Is that correct?
        # self.new_forces = None
        self.new_forces = np.zeros(self.grid_shape)

        # Impulse array which keeps track of impulses in space over time.
        # self.impulses = np.zeros(
        #     shape=[self.sim_param.number_of_time_samples, self.space_divisions_y, self.space_divisions_x])

        self.impulses = np.zeros(shape=[self.sim_param.number_of_time_samples,self.grid_shape[0],self.grid_shape[1]])

        # Array, which stores air pressure at each given point in time in the voxelized grid
        self.pressure_field = None

        # Array for pressure field results (auralisation and visualisation)
        # self.pressure_field_results = []
        self.pressure_field_results = [np.zeros(self.grid_shape)] # we skipp 1 first steps

        # Fill impulse array with impulses.

        # TODO: Switch between different source signals via bool or enum? Also create source signal container
        if do_impulse:
            stype="1"
            if stype =="1":
                # Create indices for time samples. x = [1 2 3 4 5] -> sin(x_i * pi) -> sin(pi), sin(2pi) sin(3pi)
                time_sample_indices = np.arange(
                    0, int(self.sim_param.number_of_time_samples/4), 1)
    
                # Amplitude of gaussian impulse
                
                # Shape of the signal
                A = 400e20
                mu = time_sample_indices[20] # when is the peak
                sigma = 10 # temporal spread
    
                # Position of signal source on the grid.
                # src_pos_i = int(self.grid_shape[0] / 2)
                # src_pos_j = int(self.grid_shape[1] / 2)
                
                (src_pos_i, src_pos_j) = (int(source_location[0]/self.dx),int(source_location[1]/self.dy))         
                
                # Step at which the signal occures.
                t_start = 0
                
                # here the forcing terms for each voexel at each moment of time are precomputed
                self.impulses[t_start:t_start+len(time_sample_indices), src_pos_i, src_pos_j] = PartitionData.create_gaussian_impulse(time_sample_indices, A, mu, sigma)
                # self.impulses[t_start:t_start+len(time_sample_indices), src_pos_i, src_pos_j] = PartitionData.create_gaussian_impulse(time_sample_indices, A, mu, sigma) - PartitionData.create_gaussian_impulse(time_sample_indices, A, mu+10, sigma)
    
                # if self.sim_param.visualize:
                #     import matplotlib.pyplot as plt
                #     plt.plot(self.impulses[:, int(self.space_divisions_y  / 2), int(self.space_divisions_x / 2)])
                #     plt.show()
            if stype == "2":
                f0 = 4.9
                (src_pos_i, src_pos_j) = source_location 
                time = np.arange(0, self.sim_param.number_of_time_samples * self.sim_param.delta_t , self.sim_param.delta_t)
                self.impulses[:, src_pos_i, src_pos_j] = -np.pi**2 * 2 * f0 * np.exp( -np.pi**2 * (time - 1)**2)


        if impulse:
            # Emit impulse into room
            self.impulses[:, int(self.space_divisions_y * (impulse.location[1] / dimensions[1])), int(
                self.space_divisions_x * (impulse.location[0] / dimensions[0]))] = impulse.get()

            if self.sim_param.visualize:
                import matplotlib.pyplot as plt
                plt.plot(self.impulses[:, int(self.space_divisions_y * (impulse.location[1] / dimensions[1])), int(
                    self.space_divisions_x * (impulse.location[0] / dimensions[0]))])
                plt.show()

        if sim_parameters.verbose:
            print(
                f"Created partition with dimensions {self.dimensions} m\nℎ (y): {self.h_y}, ℎ (x): {self.h_x} | Space divisions: {self.space_divisions_y} ({self.dimensions/self.space_divisions_y} m each)")


    def preprocessing(self):
        '''
        Preprocessing stage. Refers to Step 1 in the paper.
        
        for the time step t = 0
        '''
        # Preparing pressure field. Equates to function p(x) on the paper.

        self.pressure_field = np.zeros(shape=self.grid_shape)

        #print(f"presh field = {self.pressure_field}")

        # Precomputation for the DCTs to be performed. Transforming impulse to spatial forces. Skipped partitions as of now.
        self.new_forces = self.impulses[0].copy()
        
        # Relates to equation 5 and 8 of "An efficient GPU-based time domain solver for the
        # acoustic wave equation" paper.
        # For reference, see https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/4.pdf.


        # self.omega_i = np.zeros(shape=self.grid_shape)
        # for y in range(self.space_divisions_y):
        #     for x in range(self.space_divisions_x):
        #         self.omega_i[y, x, 0] = self.sim_param.c * ((np.pi ** 2) * (((x ** 2) / (self.dimensions[0] ** 2)) + ((y ** 2) / (self.dimensions[1] ** 2)))) ** 0.5
        
        self.omega_i = np.zeros(shape=self.grid_shape)
        for y in range(self.grid_shape[1]):
            for x in range(self.grid_shape[0]):
                self.omega_i[x,y] = self.sim_param.c * ((np.pi ** 2) * (((x ** 2) / (self.dimensions[0] ** 2)) + ((y ** 2) / (self.dimensions[1] ** 2)))) ** 0.5


        # TODO Semi disgusting hack. Without it, the calculation of update rule (equation 9) would crash due to division by zero
        self.omega_i[0, 0] = 0.1

        # Update time stepping. Relates to M^(n+1) and M^n in equation 8.

        self.M_previous = np.zeros(shape=self.grid_shape)

        self.M_current = np.zeros(shape=self.M_previous.shape)
        self.M_next = None

        if self.sim_param.verbose:
            print(f"Shape of omega_i: {self.omega_i.shape}")
            print(f"Shape of pressure field: {self.pressure_field.shape}")


    @staticmethod
    def create_gaussian_impulse(t, A, mu, sigma):
        '''
        ----------
        Parameters
        ----------
        A : Type
            Affects the maximum value of the peak.
        t : TYPE
            Time step for which the calculation should be done.
        mu : float
            Time step in which the peak of the signal should occur. 
        sigma : TYPE
            The spread of signal over time(width). Shape of signal.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        return np.exp(-np.power(t - mu, 2.) / (2 * np.power(sigma, 2.)))

