import numpy as np
from scipy.io.wavfile import read
from scipy.fftpack import idct, dct

from common.parameters import SimulationParameters

class PartitionData:
    def __init__(
        self,
        dimensions,
        sim_parameters,
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
        self.dimensions = dimensions
        self.sim_param = sim_parameters

        # Voxel grid spacing. Changes automatically according to frequency
        self.h_y = SimulationParameters.calculate_voxelization_step(sim_parameters) #dimensions[1] / self.space_divisions_y
        self.h_x = SimulationParameters.calculate_voxelization_step(sim_parameters)  #dimensions[0] / self.space_divisions_x TODO: Remove h(y)?

        # Check stability of wave equation
        CFL = (sim_parameters.c * sim_parameters.delta_t) / self.h_x
        assert(CFL <= 1), f"Courant-Friedrichs-Lewy number (CFL = {CFL}) is greater than 1. Wave equation is unstable. Try using a higher sample rate or more spatial samples per wave length."
        
        # Longest room dimension length dividied by H (voxel grid spacing).
        self.space_divisions_y = int(dimensions[1] / self.h_y)
        self.space_divisions_x = int(dimensions[0] / self.h_x)

        # Instantiate force f to spectral domain array, which corresponds to 𝑓~. (results of DCT computation). TODO: Elaborate more
        self.forces = None

        # Instantiate updated force f to spectral domain array. Combination of impulse and/or contribution of the interface.
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
                f"Created partition with dimensions {self.dimensions} m\nℎ (y): {self.h_y}, ℎ (x): {self.h_x} | Space divisions (y): {self.space_divisions_y} (x): {self.space_divisions_x} | CFL = {CFL}\n")

    def preprocessing(self):
        '''
        Preprocessing stage. Refers to Step 1 in the paper.
        '''
        # Preparing pressure field. Equates to function p(x) on the paper.
        self.pressure_field = np.zeros(
            shape=[self.space_divisions_y, self.space_divisions_x])
        #print(f"presh field = {self.pressure_field}")

        # Precomputation for the DCTs to be performed. Transforming impulse to spatial forces.
        self.new_forces = self.impulses[0].copy()

        # Relates to equation 5 and 8 of "An efficient GPU-based time domain solver for the
        # acoustic wave equation" paper.
        # For reference, see https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/4.pdf.

        self.omega_i = np.zeros(
            shape=[self.space_divisions_y, self.space_divisions_x])
        for y in range(self.space_divisions_y):
            for x in range(self.space_divisions_x):
                self.omega_i[y, x] = self.sim_param.c * ((np.pi ** 2) * (((x ** 2) / (
                    self.dimensions[0] ** 2)) + ((y ** 2) / (self.dimensions[1] ** 2)))) ** 0.5

        # TODO Semi disgusting hack - Without it, the calculation of update rule (equation 9) would crash due to division by zero TODO: clean up.
        self.omega_i[0, 0] = 1e-16

        # Update time stepping. Relates to M^(n+1) and M^n in equation 8.
        self.M_previous = np.zeros(
            shape=[self.space_divisions_y, self.space_divisions_x])
        self.M_current = np.zeros(shape=self.M_previous.shape)
        self.M_next = None

        if self.sim_param.verbose:
            print(f"Preprocessing started.\nShape of omega_i: {self.omega_i.shape}\nShape of pressure field: {self.pressure_field.shape}\n")

    
