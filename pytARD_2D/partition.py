from common.parameters import SimulationParameters

import numpy as np
import enum
from scipy.fft import idctn, dctn
from scipy.io.wavfile import read
from scipy.fftpack import idct, dct

class Partition(): # TODO Implement
    def __init__(self, dimensions, sim_param):
        pass

    def preprocessing():
        pass

    def simulate():
        pass

    @staticmethod
    def check_CFL(sim_param, h_x, h_y):
        CFL = sim_param.c * sim_param.delta_t * ((1 / h_x) + (1 / h_y))
        CFL_target = np.sqrt(1/3)
        assert(CFL <= CFL_target), f"Courant-Friedrichs-Lewy number (CFL = {CFL}) is greater than {CFL_target}. Wave equation is unstable. Try using a higher sample rate or more spatial samples per wave length."
        if sim_param.verbose:
            print(f"CFL = {CFL}")

    @staticmethod
    def calculate_h_x_y(sim_param):
        # Voxel grid spacing. Changes automatically according to frequency
        h_y = SimulationParameters.calculate_voxelization_step(sim_param) 
        h_x = SimulationParameters.calculate_voxelization_step(sim_param)     
        return h_y, h_x 
'''
class PMLType(enum):
    LEFT = {
        # kx
        "Min": 0.2, "Max": 0.0
    }

    RIGHT = {
        # kx
        "Min": 0.0, "Max": 0.2
    }

    TOP = {
        # ky
        "Min": 0.2, "Max": 0.0
    }

    LEFT = {
        # ky
        "Min": 0.0, "Max": 0.2
    }
'''

class PMLPartition:
    def __init__(
        self,
        dimensions,
        sim_param,
    ):
        self.dimensions = dimensions
        self.sim_param = sim_param
        
        # Voxel grid spacing. Changes automatically according to frequency
        self.h_y, self.h_x = Partition.calculate_h_x_y(sim_param)

        # Check stability of wave equation
        Partition.check_CFL(self.sim_param, self.h_x, self.h_y)

        # Longest room dimension length dividied by H (voxel grid spacing).
        self.space_divisions_y = int(dimensions[1] / self.h_y)
        self.space_divisions_x = int(dimensions[0] / self.h_x)

        # Instantiate force f to spectral domain array, which corresponds to 𝑓~. (results of DCT computation). TODO: Elaborate more
        self.force = np.zeros(shape=[self.space_divisions_y, self.space_divisions_x])

        # Array, which stores air pressure at each given point in time in the voxelized grid
        self.p_old = np.zeros(shape=[self.space_divisions_y, self.space_divisions_x])

        # TODO: Who dis? -> Document
        self.p = np.zeros(shape=[self.space_divisions_y, self.space_divisions_x])

        # Array for pressure field results (auralisation and visualisation)
        self.p_new = np.zeros(shape=[self.space_divisions_y, self.space_divisions_x])

        # See paper TODO: Make better documentation
        self.phi_x = np.zeros(shape=[self.space_divisions_y, self.space_divisions_x])
        self.phi_x_new = np.zeros(shape=[self.space_divisions_y, self.space_divisions_x])
        self.phi_y = np.zeros(shape=[self.space_divisions_y, self.space_divisions_x])
        self.phi_y_new = np.zeros(shape=[self.space_divisions_y, self.space_divisions_x])

        self.include_self_terms = False
        self.render = False

    def preprocessing():
        pass

    def simulate():
        pass

class AirPartition:
    def __init__(
        self,
        dimensions,
        sim_param,
        impulse=None
    ):
        '''
        Parameter container class for ARD simulator. Contains all relevant data to instantiate
        and run ARD simulator.

        Parameters
        ----------
        dimensions : ndarray
            Size of the partition (room) in meters.
        sim_param : SimulationParameters
            Instance of simulation parameter class.
        impulse : Impulse
            Determines if the impulse is generated on this partition, and which kind of impulse. 
        '''
        self.dimensions = dimensions
        self.sim_param = sim_param

        # Voxel grid spacing. Changes automatically according to frequency
        self.h_y, self.h_x = Partition.calculate_h_x_y(sim_param)

        # Check stability of wave equation
        Partition.check_CFL(self.sim_param, self.h_x, self.h_y)

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

        if sim_param.verbose:
            print(
                f"Created partition with dimensions {self.dimensions[0]}x{self.dimensions[1]} m\nℎ (y): {self.h_y}, ℎ (x): {self.h_x} | Space divisions (y): {self.space_divisions_y} (x): {self.space_divisions_x}")

    def preprocessing(self):
        '''
        Preprocessing stage. Refers to Step 1 in the paper.
        '''
        # Preparing pressure field. Equates to function p(x) on the paper.
        self.pressure_field = np.zeros(
            shape=[self.space_divisions_y, self.space_divisions_x])

        # Precomputation for the DCTs to be performed. Transforming impulse to spatial forces.
        self.new_forces = self.impulses[0].copy()

        # Relates to equation 5 and 8 of "An efficient GPU-based time domain solver for the
        # acoustic wave equation" paper.
        # For reference, see https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/4.pdf.
        self.omega_i = np.zeros(shape=[self.space_divisions_y, self.space_divisions_x])

        # Initialize omega_i
        for y in range(self.space_divisions_y):
            for x in range(self.space_divisions_x):
                self.omega_i[y, x] = \
                    self.sim_param.c * (
                        (np.pi ** 2) * (
                            ((x ** 2) / (self.dimensions[0] ** 2)) + 
                            ((y ** 2) / (self.dimensions[1] ** 2))
                        )
                    ) ** 0.5

        # TODO Semi disgusting hack. Without it, the calculation of update rule (equation 9) would crash due to division by zero
        self.omega_i[0, 0] = 1e-8   

        # Update time stepping. Relates to M^(n+1) and M^n in equation 8.
        self.M_previous = np.zeros(shape=[self.space_divisions_y, self.space_divisions_x])
        self.M_current = np.zeros(shape=self.M_previous.shape)
        self.M_next = None

        if self.sim_param.verbose:
            print(f"Preprocessing started.\nShape of omega_i: {self.omega_i.shape}\nShape of pressure field: {self.pressure_field.shape}\n")

    def simulate(self, t_s, normalization_factor=1):
        # Execute DCT for next sample
        self.forces = dctn(self.new_forces, 
            type=2, 
            s=[ # TODO This parameter may be unnecessary
                self.space_divisions_y, 
                self.space_divisions_x
            ])

        # Updating mode for spectral coefficients p.
        # Relates to (2 * F^n) / (ω_i ^ 2) * (1 - cos(ω_i * Δ_t)) in equation 8.
        self.force_field = (
            (2 * self.forces) / ((self.omega_i) ** 2)) * (
                1 - np.cos(self.omega_i * self.sim_param.delta_t))

        # Edge case for first iteration according to Nikunj Raghuvanshi. p[n+1] = 2*p[n] – p[n-1] + (\delta t)^2 f[n], while f is impulse and p is pressure field.
        self.force_field[0, 0] = 2 * self.M_current[0, 0] - self.M_previous[0, 0] + \
            self.sim_param.delta_t ** 2 * \
                self.impulses[t_s][0, 0]

        # Relates to M^(n+1) in equation 8.
        self.M_next = (2 * self.M_current * np.cos(
            self.omega_i * self.sim_param.delta_t) - self.M_previous + self.force_field)

        # Convert modes to pressure values using inverse DCT.
        self.pressure_field = idctn(self.M_next.reshape(
            self.space_divisions_y, 
            self.space_divisions_x), 
            type=2, 
            s=[ # TODO This parameter may be unnecessary
                self.space_divisions_y, 
                self.space_divisions_x
            ])

        # Normalize pressure p by using normalization constant.
        self.pressure_field *= np.sqrt(normalization_factor)

        # Add results of IDCT to pressure field
        self.pressure_field_results.append(
            self.pressure_field.copy())

        # Update time stepping to prepare for next time step / loop iteration.
        self.M_previous = self.M_current.copy()
        self.M_current = self.M_next.copy()

        # Update impulses
        self.new_forces = self.impulses[t_s].copy()
