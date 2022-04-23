from common.parameters import SimulationParameters

import numpy as np
from enum import Enum
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


class PMLType(Enum):
    LEFT = { # for kx
        "Min": 0.2, "Max": 0.0
    }
    RIGHT = { # for kx
        "Min": 0.0, "Max": 0.2
    }
    TOP = { # for ky
        "Min": 0.2, "Max": 0.0
    }
    BOTTOM = { # for ky
        "Min": 0.0, "Max": 0.2
    }


class PMLPartition(Partition):
    def __init__(
        self,
        dimensions: np.ndarray,
        sim_param: SimulationParameters, 
        type: PMLType
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
    
        if type == PMLType.LEFT or type == PMLType.RIGHT:
            #self.space_divisions_x += 1
            print("horizontal")
        else:
            #self.space_divisions_y += 1
            print("vertical")

        # Instantiate force f to spectral domain array, which corresponds to 𝑓~. (results of DCT computation). TODO: Elaborate more
        self.new_forces = np.zeros(shape=[self.space_divisions_y + 1, self.space_divisions_x + 1])

        # Array, which stores air pressure at each given point in time in the voxelized grid
        self.p_old = np.zeros(shape=[self.space_divisions_y + 1, self.space_divisions_x + 1])

        # TODO: Who dis? -> Document
        self.pressure_field = np.zeros(shape=[self.space_divisions_y + 1, self.space_divisions_x + 1])

        # Array for pressure field results (auralisation and visualisation)
        self.p_new = np.zeros(shape=[self.space_divisions_y + 1, self.space_divisions_x + 1])

        # See paper TODO: Make better documentation
        self.phi_x = np.zeros(shape=[self.space_divisions_y + 1, self.space_divisions_x + 1])
        self.phi_x_new = np.zeros(shape=[self.space_divisions_y + 1, self.space_divisions_x + 1])
        self.phi_y = np.zeros(shape=[self.space_divisions_y + 1, self.space_divisions_x + 1])
        self.phi_y_new = np.zeros(shape=[self.space_divisions_y + 1, self.space_divisions_x + 1])

        self.include_self_terms = False
        self.render = False
        self.type = type

        self.FDTD_coeffs = [2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0]
        self.fourth_coeffs = [1.0, -8.0, 0.0, 8.0, -1.0]

        # Array for pressure field results (auralisation and visualisation)
        self.pressure_field_results = []

        if sim_param.verbose:
            print(
                f"Created PML partition with dimensions {self.dimensions[0]}x{self.dimensions[1]} m\nℎ (y): {self.h_y}, ℎ (x): {self.h_x} | Space divisions (y): {self.space_divisions_y} (x): {self.space_divisions_x}")


    def preprocessing(self):
        pass

    def get_safe(self, source, y, x):
        if x < 0 or x >= self.space_divisions_x or y < 0 or y >= self.space_divisions_y:
            return source[-1, -1]
        return source[y, x]    
        

    def simulate(self, t_s, normalization_factor=1):
        dx = 1.0 
        dy = 1.0

        for i in range(self.space_divisions_x):
            kx = 0.0
            ky = 0.0

            # TODO put both ifs together into one -> optimize
            if self.type == PMLType.LEFT:
                if i < 20:
                    kx = (20 - i) * self.type.value['Min'] / 10.0
                    ky = 0.05
                else:
                    kx = 0.0
                    ky = 0.0
            
            if self.type == PMLType.RIGHT:
                if i < 20:
                    kx = (i - 20) * self.type.value['Max'] / 10.0
                    ky = 0.05
                else:
                    kx = 0.0
                    ky = 0.0
            
            for j in range(self.space_divisions_y):
                if self.type == PMLType.TOP:
                    if j < 20:
                        ky = (20 - j) * self.type.value['Min'] / 10.0
                        kx = 0.05
                    else:
                        kx = 0.0
                        ky = 0.0
                
                if self.type == PMLType.BOTTOM:
                    if j < 20:
                        ky = (j - 20) * self.type.value['Max'] / 10.0
                        kx = 0.05
                    else:
                        kx = 0.0
                        ky = 0.0
                
                KPx = 0.0
                KPy = 0.0

                for k in range(len(self.FDTD_coeffs)):
                    KPx += self.FDTD_coeffs[k] * self.get_safe(self.pressure_field, j, i * k - 3)
                    KPy += self.FDTD_coeffs[k] * self.get_safe(self.pressure_field, j + k - 3, i)
                    
                KPx /= 180.0
                KPy /= 180.0

                term1 = 2 * self.pressure_field[j, i]
                term2 = -self.p_old[j, i]
                #if t_s < 10:
                #term3 = (self.sim_param.c ** 2) * (KPx + KPy + self.new_forces[j, i])
                term3 = (self.sim_param.c ** 2) * (KPx + KPy)
                #else:
                #    term3 = (self.sim_param.c ** 2) * (KPx + KPy)
                #print(f"{term3}", end="\t")
                term4 = -(kx + ky) * (self.pressure_field[j, i] - self.p_old[j, i]) / self.sim_param.delta_t
                term5 = -kx * ky * self.pressure_field[j, i]

                dphidx = 0.0
                dphidy = 0.0

                for k in range(len(self.fourth_coeffs)):
                    dphidx += self.fourth_coeffs[k] * self.get_safe(self.phi_x, j, i * k - 2)
                    dphidy += self.fourth_coeffs[k] * self.get_safe(self.phi_y, j + k - 2, i)

                dphidx /= 12.0
                dphidy /= 12.0

                term6 = dphidx + dphidy

                # Calculation of next wave
                #self.p_new[j, i] = term1 + term2 + ((self.sim_param.delta_t ** 2) * (term3 + term4 + term5 + term6))
                self.p_new[j, i] = term1 + term2 + ((self.sim_param.delta_t ** 2) * (term3 + term4 + term5 + term6)) + self.sim_param.delta_t**2 * self.new_forces[j, i] / (1 + ((KPx+KPy)/2) * self.sim_param.delta_t)

                dudx = 0.0
                dudy = 0.0

                for k in range(len(self.fourth_coeffs)):
                    dudx += self.fourth_coeffs[k] * self.get_safe(self.p_new, j, i + k - 2)
                    dudy += self.fourth_coeffs[k] * self.get_safe(self.p_new, j + k - 2, i)

                dudx /= 12.0
                dudy /= 12.0

                self.phi_x_new[j, i] = self.phi_x[j, i] - self.sim_param.delta_t * kx * self.phi_x[j, i] + self.sim_param.delta_t * (self.sim_param.c ** 2) * (ky - kx) * dudx
                self.phi_y_new[j, i] = self.phi_y[j, i] - self.sim_param.delta_t * ky * self.phi_y[j, i] + self.sim_param.delta_t * (self.sim_param.c ** 2) * (kx - ky) * dudy

        self.pressure_field_results.append(self.p_new.copy())

        # Swap old with new phis with the new switcheroo
        self.phi_x, self.phi_x_new = self.phi_x_new, self.phi_x
        self.phi_y, self.phi_y_new = self.phi_y_new, self.phi_y
        
        # Do the ol' switcheroo
        temp = self.p_old.copy()
        self.p_old = self.pressure_field.copy()
        self.pressure_field = self.p_new.copy()
        self.p_new = temp

        # Reset force
        self.new_forces = np.zeros(shape=self.new_forces.shape)

class AirPartition(Partition):
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
                f"Created air partition with dimensions {self.dimensions[0]}x{self.dimensions[1]} m\nℎ (y): {self.h_y}, ℎ (x): {self.h_x} | Space divisions (y): {self.space_divisions_y} (x): {self.space_divisions_x}")

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
