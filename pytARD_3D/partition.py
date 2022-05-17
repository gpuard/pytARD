from common.parameters import SimulationParameters
import common.finite_differences as FD

import numpy as np
from enum import Enum
from scipy.fft import idctn, dctn

class Partition3D():
    def __init__(self, dimensions: np.ndarray, sim_param: SimulationParameters):
        pass

    def preprocessing():
        pass

    def simulate():
        pass

    @staticmethod
    def check_CFL(sim_param, h_x, h_y, h_z):
        CFL = sim_param.c * sim_param.delta_t * ((1 / h_x) + (1 / h_y) + (1 / h_z))
        CFL_target = np.sqrt(1/3)
        assert(CFL <= CFL_target), f"Courant-Friedrichs-Lewy number (CFL = {CFL}) is greater than {CFL_target}. Wave equation is unstable. Try using a higher sample rate or more spatial samples per wave length."
        if sim_param.verbose:
            print(f"CFL = {CFL}")

    @staticmethod
    def calculate_h_x_y(sim_param):
        # Voxel grid spacing. Changes automatically according to frequency
        h_z = SimulationParameters.calculate_voxelization_step(sim_param)     
        h_y = SimulationParameters.calculate_voxelization_step(sim_param) 
        h_x = SimulationParameters.calculate_voxelization_step(sim_param)     
        return h_z, h_y, h_x 


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

class DampingProfile:
    '''
    Damping profile. Determines how intense the reflections of the PML partition are, or how much sound energy is absorbed.
    '''

    def __init__(self, room_length: float, c: float, reflection_coefficient: float):
        '''
        Instantiates a damping profile for a PML partition.

        Parameters
        ----------
        room_length : float
            Length of partition.
        c : float 
            Speed of sound.
        reflection_coefficient : float
            Reflection coefficient R. Determines how intense the reflections of the PML partition are.
        '''
        self.zetta_i = DampingProfile.calculate_zetta(room_length, c, reflection_coefficient)

    def damping_profile(self, x, width):
        return self.zetta_i * (x / width - np.sin(2 * np.pi * x / width) / (2 * np.pi))

    @staticmethod
    def calculate_zetta(L: float, c: float, R: float):
        '''
        Calculating zetta_i value from given reflection coefficient.

        Parameters
        ----------
        L : float
            Length of PML given by room dimensions
        c : float
            Speed of sound
        R : float
            Reflection coefficient, ranging from 0 to 1.

        Returns
        -------
        float
            Zetta_i value
        '''
        assert(R < 1), "Reflection coefficient should be smaller than 1."
        assert(R > 0), "Reflection coefficient should be bigger than 0."
        return (c / L) * np.log(1 / R)

class PMLPartition3D(Partition3D):
    '''
    PML partition. Absorps sound energy depending on the damping profile.
    '''
    def __init__(
        self,
        dimensions: np.ndarray,
        sim_param: SimulationParameters, 
        pml_type: PMLType,
        damping_profile: DampingProfile
    ):

        '''
        Instantiates a PML partition in the domain. Absorps sound energy depending on the damping profile.

        Parameters
        ----------
        dimensions : ndarray
            Size of the partition (room) in meters.
        sim_param : SimulationParameters
            Instance of simulation parameter class.
        pml_type : PMLType
            Type (direction) of PML partition.
        damping_profile : DampingProfile
            Damping profile of the PML partition, determines the intensity of wave absorption.
        '''

        self.dimensions = dimensions
        self.sim_param = sim_param
        
        # Voxel grid spacing. Changes automatically according to frequency
        self.h_z, self.h_y, self.h_x = Partition3D.calculate_h_x_y(sim_param)

        # Check stability of wave equation
        Partition3D.check_CFL(self.sim_param, self.h_x, self.h_y, self.h_z)

        # Longest room dimension length dividied by H (voxel grid spacing).
        # self.space_divisions_z = int(dimensions[2] / self.h_z)
        # self.space_divisions_y = int(dimensions[1] / self.h_y)
        # self.space_divisions_x = int(dimensions[0] / self.h_x)
        self.space_divisions_z = int(dimensions[2] / self.h_z)
        self.space_divisions_y = int(dimensions[1] / self.h_y)
        if pml_type == PMLType.RIGHT or pml_type == PMLType.LEFT:
            self.space_divisions_x = 7
        else:
            self.space_divisions_x = int(dimensions[0] / self.h_x)

        self.grid_shape = (self.space_divisions_x, self.space_divisions_y, self.space_divisions_z)
        
        shape_template = np.zeros(shape=[self.space_divisions_z, self.space_divisions_y, self.space_divisions_x])

        # Instantiate force f to spectral domain array, which corresponds to 𝑓~. TODO: Elaborate more
        self.new_forces = shape_template.copy()

        # Array, which stores air pressure at each given point in time in the voxelized grid
        self.p_old = shape_template.copy()

        # TODO: Who dis? -> Document
        self.pressure_field = shape_template.copy()

        # Array for pressure field results (auralisation and visualisation)
        self.p_new = shape_template.copy()

        # See paper TODO: Make better documentation
        self.phi_x = shape_template.copy()
        self.phi_x_new = shape_template.copy()
        self.phi_y = shape_template.copy()
        self.phi_y_new = shape_template.copy()
        self.phi_z = shape_template.copy()
        self.phi_z_new = shape_template.copy()
        
        #staggered in time
        self.psi = shape_template.copy()
        self.psi_new = shape_template.copy()

        self.include_self_terms = False
        self.render = False
        self.pml_type = pml_type

        self.FDTD_coeffs = [2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0]
        self.fourth_coeffs = [1.0, -8.0, 0.0, 8.0, -1.0]

        # Array for pressure field results (auralisation and visualisation)
        self.pressure_field_results = []

        self.damping_profile = damping_profile

        if sim_param.verbose:
            print(
                f"Created PML partition with dimensions {self.dimensions[0]}x{self.dimensions[1]}x{self.dimensions[2]} m\nℎ (z): {self.h_z} ℎ (y): {self.h_y}, ℎ (x): {self.h_x} | Space divisions (y): {self.space_divisions_y} (x): {self.space_divisions_x} | Zetta_i: {self.damping_profile.zetta_i}")
            print(
                f"PML grid shape: {self.grid_shape}")


    def preprocessing(self):
        pass

    def get_safe(self, source, z, y, x):
        if x < 0 or x >= self.space_divisions_x or y < 0 or y >= self.space_divisions_y or z < 0 or z >= self.space_divisions_z:
            return source[-1, -1, -1]
        return source[z, y, x]    
        
    # @profile
    def simulate(self, t_s, normalization_factor=1):
        dx = 1.0 
        dy = 1.0
        dz = 1.0

        for x in range(self.space_divisions_x):
            #kx = 0.0
            #ky = 0.0
            kx = self.damping_profile.damping_profile(x, self.space_divisions_x)
            
            for y in range(self.space_divisions_y):
                ky = self.damping_profile.damping_profile(y, self.space_divisions_y)

                for z in range(self.space_divisions_z):
                    kz = self.damping_profile.damping_profile(z, self.space_divisions_z)

                    KPx = 0.0
                    KPy = 0.0
                    KPz = 0.0

                    for k in range(len(self.FDTD_coeffs)):
                        KPx += self.FDTD_coeffs[k] * self.get_safe(self.pressure_field, z, y, x + k - 3)
                        KPy += self.FDTD_coeffs[k] * self.get_safe(self.pressure_field, z, y + k - 3, x)
                        KPz += self.FDTD_coeffs[k] * self.get_safe(self.pressure_field, z + k - 3, y, x)
                    
                    KPx /= 180.0
                    KPy /= 180.0
                    KPz /= 180.0
                    
                    # mirrors the wave
                    # kx = 1000
                    # breaks simulation
                    # kx = 1000
                    # ky = 10000
                    # kz = 10000
                    

                    term1 = 2 * self.pressure_field[z, y, x]
                    term2 = -self.p_old[z, y, x]
                    term3 = (self.sim_param.c ** 2) * (KPx + KPy + KPz)
                    term4 = -(kx + ky + kz) * (self.pressure_field[z, y, x] - self.p_old[z, y, x]) / self.sim_param.delta_t
                    term5 = -kx * ky * kz * self.pressure_field[z, y, x] #??
                    
                    
                    dphidx = 0.0
                    dphidy = 0.0
                    dphidz = 0.0

                    for k in range(len(self.fourth_coeffs)):
                        dphidx += self.fourth_coeffs[k] * self.get_safe(self.phi_x, z, y, x + k - 2)
                        dphidy += self.fourth_coeffs[k] * self.get_safe(self.phi_y, z, y + k - 2, x)
                        dphidz += self.fourth_coeffs[k] * self.get_safe(self.phi_y, z + k - 2, y, x)

                    dphidx /= 12.0
                    dphidy /= 12.0
                    dphidz /= 12.0

                    term6 = dphidx + dphidy + dphidz

                    # Calculation of next wave
                    self.p_new[z, y, x] = term1 + term2 + ((self.sim_param.delta_t ** 2) * (term3 + term4 + term5 + term6)) + self.sim_param.delta_t ** 2 * self.new_forces[z, y, x] / (1 + ((KPx + KPy + KPz) / 2) * self.sim_param.delta_t)

                    dudx = 0.0
                    dudy = 0.0
                    dudz = 0.0

                    for k in range(len(self.fourth_coeffs)):
                        dudx += self.fourth_coeffs[k] * self.get_safe(self.p_new, z, y, x + k - 2)
                        dudy += self.fourth_coeffs[k] * self.get_safe(self.p_new, z, y + k - 2, x)
                        dudz += self.fourth_coeffs[k] * self.get_safe(self.p_new, z + k - 2, y, x)

                    dudx /= 12.0
                    dudy /= 12.0
                    dudz /= 12.0
                    
                    dpsidx = 0.0
                    dpsidy = 0.0
                    dpsidz = 0.0
                    
                    for k in range(len(self.fourth_coeffs)):
                        dpsidx += self.fourth_coeffs[k] * self.get_safe(self.p_new, z, y, x + k - 2)
                        dpsidy += self.fourth_coeffs[k] * self.get_safe(self.p_new, z, y + k - 2, x)
                        dpsidz += self.fourth_coeffs[k] * self.get_safe(self.p_new, z + k - 2, y, x)

                    dpsidx /= 12.0
                    dpsidy /= 12.0
                    dpsidz /= 12.0
                    
                    self.phi_x_new[z, y, x] = self.phi_x[z, y, x] - self.sim_param.delta_t * kx * self.phi_x[z, y, x] + self.sim_param.delta_t * (self.sim_param.c ** 2) * (ky + kz - kx) * dudx + self.sim_param.delta_t * ky * kz * dpsidx
                    self.phi_y_new[z, y, x] = self.phi_y[z, y, x] - self.sim_param.delta_t * ky * self.phi_y[z, y, x] + self.sim_param.delta_t * (self.sim_param.c ** 2) * (kz + kx - ky) * dudy + self.sim_param.delta_t * kz * kx * dpsidy
                    self.phi_z_new[z, y, x] = self.phi_z[z, y, x] - self.sim_param.delta_t * kz * self.phi_z[z, y, x] + self.sim_param.delta_t * (self.sim_param.c ** 2) * (kx + ky - kz) * dudz + self.sim_param.delta_t * kx * ky * dpsidz
                    
                    self.psi_new[z,y,x] = self.sim_param.delta_t * self.pressure_field[z,y,x] + self.psi[z,y,x]

        self.pressure_field_results.append(self.p_new.copy())

        # Swap old with new phis with the new switcheroo
        self.phi_x = self.phi_x_new.copy()
        self.phi_y = self.phi_y_new.copy()
        self.phi_z = self.phi_z_new.copy()
        
        self.psi = self.psi_new.copy()

        # Do the ol' switcheroo
        temp = self.p_old.copy()
        self.p_old = self.pressure_field.copy()
        self.pressure_field = self.p_new.copy()
        self.p_new = temp

        # Reset force
        self.new_forces = np.zeros(shape=self.new_forces.shape)
    
    # def simulate(self, t_s, normalization_factor=1):
        # based on paper; using staggered grid
    # # a possibility to split calculation of phi    
    #     dx = 1.0 
    #     dy = 1.0
    #     dz = 1.0
    #     for x in range(self.space_divisions_x)[1:-1]:
    #         #kx = 0.0
    #         #ky = 0.0
    #         kx = self.damping_profile.damping_profile(x, self.space_divisions_x)
            
    #         for y in range(self.space_divisions_y)[1:-1]:
    #             ky = self.damping_profile.damping_profile(y, self.space_divisions_y)

    #             for z in range(self.space_divisions_z)[1:-1]:
    #                 kz = self.damping_profile.damping_profile(z, self.space_divisions_z)
    #                 # kx = 40
    #                 # ky = 40
    #                 # kz = 40
        
    #                 # UPDATE RULE(the extrapolation scheme): P @ time n+1
                    
    #                 term1 = (1 / self.sim_param.delta_t**2 + (kx + ky + kz) * 1 /self.sim_param.delta_t)
    #                 term2 = (kx + ky + kz) * self.p_old[z, y, x] / (2*self.sim_param.delta_t)
    #                 term3 = - (kx * ky + ky * kz + kz * ky) * self.pressure_field[z,y,x]

                                        
    #                 # TODO: You may want to get the once and store as self.fd_coefs                    
    #                 # think of x as an anchor
    #                 h_nbr_pts = int(len(FD.get_fd_coefficients(2,2))/2)
    #                 # TODO: it smells here of matrix
    #                 dpdx = np.dot(self.pressure_field[z,y,x-h_nbr_pts:x+h_nbr_pts+1],FD.get_fd_coefficients(2,2))
    #                 dpdy = np.dot(self.pressure_field[z,y-h_nbr_pts:y+h_nbr_pts+1,x],FD.get_fd_coefficients(2,2))
    #                 dpdz = np.dot(self.pressure_field[z-h_nbr_pts:z+h_nbr_pts+1,y,x],FD.get_fd_coefficients(2,2))

    #                 term4 = (dpdx + dpdy + dpdz) * self.sim_param.c**2
                                         
    #                 # (self.phi_x[z-1:z+2,y-1,x+1:x+2] - self.phi_x[z-1:z+2,y-1,x+1:x+2])[1,:]=0
    #                 # (self.phi_x[z-1:z+2,y-1,x+1:x+2] - self.phi_x[z-1:z+2,y-1,x+1:x+2])[:,1]=0
    #                 # np.sum()
    #                 # dphidx = 0.25 / self.self.h_x * (   self.phi_x[z-1,y-1,x+1] + self.phi_x[z+1,y-1,x+1] + self.phi_x[z-1,y+1,x+1] + self.phi_x[z+1,y+1,x+1]
    #                 #                                   - self.phi_x[z-1,y-1,x-1] - self.phi_x[z+1,y-1,x-1] - self.phi_x[z-1,y+1,x-1] - self.phi_x[z+1,y+1,x-1])
                    
                    
    #                 # dphidy = 0.25 / self.self.h_y * (   self.phi_y[z-1,y+1,x-1] + self.phi_y[z+1,y+1,x-1] + self.phi_y[z-1,y+1,x+1] + self.phi_y[z+1,y+1,x+1]
    #                 #                                   - self.phi_y[z-1,y-1,x-1] - self.phi_y[z+1,y-1,x-1] - self.phi_y[z-1,y-1,x+1] - self.phi_y[z+1,y-1,x+1])
                    
                    
    #                 # dphidz = 0.25 / self.self.h_z * (   self.phi_z[z+1,y-1,x-1] + self.phi_z[z+1,y+1,x-1] + self.phi_z[z+1,y-1,x+1] + self.phi_z[z+1,y+1,x+1]
    #                 #                                   - self.phi_z[z-1,y-1,x-1] - self.phi_z[z-1,y+1,x-1] - self.phi_z[z-1,y-1,x+1] - self.phi_z[z-1,y+1,x+1])
                    
    #                 dphidx = 0.25 / self.h_x * (   self.phi_x[z-1,y-1,x+1] + self.phi_x[z+1,y-1,x+1] + self.phi_x[z-1,y+1,x+1] + self.phi_x[z+1,y+1,x+1]
    #                                                   - self.phi_x[z-1,y-1,x-1] - self.phi_x[z+1,y-1,x-1] - self.phi_x[z-1,y+1,x-1] - self.phi_x[z+1,y+1,x-1])
                    
                    
    #                 dphidy = 0.25 / self.h_y * (   self.phi_y[z-1,y+1,x-1] + self.phi_y[z+1,y+1,x-1] + self.phi_y[z-1,y+1,x+1] + self.phi_y[z+1,y+1,x+1]
    #                                                   - self.phi_y[z-1,y-1,x-1] - self.phi_y[z+1,y-1,x-1] - self.phi_y[z-1,y-1,x+1] - self.phi_y[z+1,y-1,x+1])
                    
                    
    #                 dphidz = 0.25 / self.h_z * (   self.phi_z[z+1,y-1,x-1] + self.phi_z[z+1,y+1,x-1] + self.phi_z[z+1,y-1,x+1] + self.phi_z[z+1,y+1,x+1]
    #                                                   - self.phi_z[z-1,y-1,x-1] - self.phi_z[z-1,y+1,x-1] - self.phi_z[z-1,y-1,x+1] - self.phi_z[z-1,y+1,x+1])
                    
    #                 term5 = dphidx  + dphidy + dphidz
                    
    #                 term6 = -kx * ky * kz * self.pressure_field[z, y, x]
                    
    #                 self.p_new[z, y, x] = (term2 + term3 * term4 + term5 + term6 + self.new_forces[z,y,x]) / term1 # NOTE: adde forcing field
                    
    #                 # UPDATE RULE: phi
    #                 avg_p = lambda k,j,i: self.pressure_field[k,j,i] + self.pressure_field[k+1,j,i] + self.pressure_field[k,j+1,i] + self.pressure_field[k+1,j+1,i]
    #                 avg_pn = lambda k,j,i: self.p_new[k,j,i] + self.p_new[k+1,j,i] + self.p_new[k,j+1,i] + self.p_new[k+1,j+1,i]
    #                 avg_psi = lambda k,j,i: self.psi[k,j,i] + self.psi[k+1,j,i] + self.psi[k,j+1,i] + self.psi[k+1,j+1,i]
                    
    #                 term1 = (1/self.sim_param.delta_t + kx/2)
    #                 term2 = self.phi_x[z+1,y+1,x+1]/self.sim_param.delta_t - kx*self.phi_x[z+1,y+1,x+1]/2
    #                 term3 = self.sim_param.c**2 * (ky+kz-kx) * 0.5 / self.h_x * (avg_pn(z,y,x+1)-avg_pn(z,y,x)+avg_p(z,y,x+1)-avg_p(z,y,x))
    #                 term4 = self.sim_param.c**2 * ky*kz/self.h_x * 0.25 * (avg_psi(z,y,x+1) - avg_psi(z,y,x))
                    
    #                 self.phi_x_new[z, y, x]  = 1/term1 * (term2 + term3 + term4)
                    
    #                 avg_p = lambda k,j,i: self.pressure_field[k,j,i] + self.pressure_field[k+1,j,i] + self.pressure_field[k,j,i+1] + self.pressure_field[k+1,j,i+1]
    #                 avg_pn = lambda k,j,i: self.p_new[k,j,i] + self.p_new[k+1,j,i] + self.p_new[k,j,i+1] + self.p_new[k+1,j,i+1]
    #                 avg_psi = lambda k,j,i: self.psi[k,j,i] + self.psi[k+1,j,i] + self.psi[k,j,i+1] + self.psi[k+1,j,i+1]
    #                 term1 = (1/self.sim_param.delta_t + ky/2)
    #                 term2 = self.phi_x[z+1,y+1,x+1]/self.sim_param.delta_t - kx*self.phi_x[z+1,y+1,x+1]/2
    #                 term3 = self.sim_param.c**2 * (kz+kx-ky) * 0.5 / self.h_x * (avg_pn(z,y+1,x)-avg_pn(z,y,x)+avg_p(z,y+1,x)-avg_p(z,y,x))
    #                 term4 = self.sim_param.c**2 * kz*kx/self.h_x * 0.25 * (avg_psi(z,y+1,x) - avg_psi(z,y,x))
                    
    #                 self.phi_y_new[z, y, x]  = 1/term1 * (term2 + term3 + term4)

    #                 avg_p = lambda k,j,i: self.pressure_field[k,j,i] + self.pressure_field[k,j,i+1] + self.pressure_field[k,j+1,i] + self.pressure_field[k,j+1,i+1]
    #                 avg_pn = lambda k,j,i: self.p_new[k,j,i] + self.p_new[k,j+1,i] + self.p_new[k,j,i+1] + self.p_new[k,j+1,i+1]
    #                 avg_psi = lambda k,j,i: self.psi[k,j,i] + self.psi[k,j+1,i] + self.psi[k,j,i+1] + self.psi[k,j+1,i+1]
    #                 term1 = (1/self.sim_param.delta_t + kz/2)
    #                 term2 = self.phi_x[z+1,y+1,x+1]/self.sim_param.delta_t- kx*self.phi_x[z+1,y+1,x+1]/2
    #                 term3 = self.sim_param.c**2 * (kx+ky-kz) * 0.5 / self.h_x * (avg_pn(z+1,y,x)-avg_pn(z,y,x)+avg_p(z+1,y,x)-avg_p(z,y,x))
    #                 term4 = self.sim_param.c**2 * kx*ky/self.h_x * 0.25 * (avg_psi(z+1,y,x) - avg_psi(z,y,x))
                    
    #                 self.phi_z_new[z, y, x]  = 1/term1 * (term2 + term3 + term4)
                    
    #                 self.psi_new[z, y, x]  = self.sim_param.delta_t * self.p_new[z,y,x] - self.psi[z, y, x] 

                
    #     self.pressure_field_results.append(self.p_new.copy())

    #     self.phi_x = self.phi_x_new.copy()
    #     self.phi_y = self.phi_y_new.copy()
    #     self.phi_z = self.phi_z_new.copy()
    #     self.psi = self.psi_new.copy() 

    #     # Do the ol' switcheroo
    #     temp = self.p_old.copy()
    #     self.p_old = self.pressure_field.copy()
    #     self.pressure_field = self.p_new.copy()
    #     self.p_new = temp

    #     # Reset force
    #     self.new_forces = np.zeros(shape=self.new_forces.shape)         

class AirPartition3D:
    def __init__(
        self,
        dimensions: np.ndarray,
        sim_param: SimulationParameters,
        impulse: bool=None
    ):
        '''
        Air partition. Resembles an empty space in which sound can travel through.

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
        self.h_z = SimulationParameters.calculate_voxelization_step(sim_param)
        self.h_y = SimulationParameters.calculate_voxelization_step(sim_param)
        self.h_x = SimulationParameters.calculate_voxelization_step(sim_param)

        # Check stability of wave equation
        CFL = sim_param.c * sim_param.delta_t * ((1 / self.h_x) + (1 / self.h_y) + (1 / self.h_z))
        CFL_target = np.sqrt(1/3)
        assert(CFL <= CFL_target), f"Courant-Friedrichs-Lewy number (CFL = {CFL}) is greater than {CFL_target}. Wave equation is unstable. Try using a higher sample rate or more spatial samples per wave length."

        # Longest room dimension length dividied by H (voxel grid spacing).
        self.space_divisions_z = int(dimensions[2] / self.h_z)
        self.space_divisions_y = int(dimensions[1] / self.h_y)
        self.space_divisions_x = int(dimensions[0] / self.h_x)

        # Instantiate forces array, which corresponds to F in update rule (results of DCT computation). TODO: Elaborate more
        self.forces = None

        # Instantiate updated forces array. Combination of impulse and/or contribution of the interface.
        # DCT of new_forces will be written into forces. TODO: Is that correct?
        self.new_forces = None

        # Impulse array which keeps track of impulses in space over time.
        self.impulses = np.zeros(
            shape=[self.sim_param.number_of_samples, self.space_divisions_z, self.space_divisions_y, self.space_divisions_x])

        # Array, which stores air pressure at each given point in time in the voxelized grid
        self.pressure_field = None

        # Array for pressure field results (auralisation and visualisation)
        self.pressure_field_results = []

        # Fill impulse array with impulses.
        if impulse:
            # Emit impulse into room
            self.src_grid_loc =  (  int(self.space_divisions_z * (impulse.location[2] / dimensions[2])),
                                    int(self.space_divisions_y * (impulse.location[1] / dimensions[1])),
                                    int(self.space_divisions_x * (impulse.location[0] / dimensions[0])))
            self.impulses[:, 
                int(self.space_divisions_z * (impulse.location[2] / dimensions[2])),
                int(self.space_divisions_y * (impulse.location[1] / dimensions[1])), 
                int(self.space_divisions_x * (impulse.location[0] / dimensions[0]))] = impulse.get()
        if sim_param.verbose:
            print(
                f"Created partition with dimensions {self.dimensions[0]}x{self.dimensions[1]}x{self.dimensions[2]} m\nℎ (z): {self.h_z}, ℎ (y): {self.h_y}, ℎ (x): {self.h_x} | Space divisions: {self.space_divisions_y} | CFL = {CFL}")

    def preprocessing(self):
        '''
        Preprocessing stage. Refers to Step 1 in the paper.
        '''
        # Preparing pressure field. Equates to function p(x) on the paper.
        self.pressure_field = np.zeros(
            shape=[
                self.space_divisions_z, 
                self.space_divisions_y, 
                self.space_divisions_x
            ]
        )

        # Precomputation for the DCTs to be performed. Transforming impulse to spatial forces. Skipped partitions as of now.
        self.new_forces = self.impulses[0].copy()

        # Relates to equation 5 and 8 of "An efficient GPU-based time domain solver for the
        # acoustic wave equation" paper.
        # For reference, see https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/4.pdf.

        self.omega_i = np.zeros(
            shape=[
                self.space_divisions_z, 
                self.space_divisions_y, 
                self.space_divisions_x]
            )

        for z in range(self.space_divisions_z):
            for y in range(self.space_divisions_y):
                for x in range(self.space_divisions_x):
                    self.omega_i[z, y, x] = \
                        self.sim_param.c * (
                            (np.pi ** 2) *
                            (
                                ((x ** 2) / (self.dimensions[0] ** 2)) +
                                ((y ** 2) / (self.dimensions[1] ** 2)) +
                                ((z ** 2) / (self.dimensions[2] ** 2))
                            )
                        ) ** 0.5

        # TODO Semi disgusting hack. Without it, the calculation of update rule (equation 9) would crash due to division by zero
        self.omega_i[0, 0, 0] = 1e-8

        # Update time stepping. Relates to M^(n+1) and M^n in equation 8.
        self.M_previous = np.zeros(shape=[self.space_divisions_z, self.space_divisions_y, self.space_divisions_x])
        self.M_current = np.zeros(shape=self.M_previous.shape)
        self.M_next = None

        if self.sim_param.verbose:
            print(f"Preprocessing started.\nShape of omega_i: {self.omega_i.shape}\nShape of pressure field: {self.pressure_field.shape}\n")

    def simulate(self, t_s: int, normalization_factor: float=1):
        '''
        Executes the simulation for the partition.

        Parameters
        ----------
        t_s : int
            Current time step.
        normalization_factor : float
            Normalization multiplier to harmonize amplitudes between partitions.
        '''
        # Execute DCT for next sample
        self.forces = dctn(self.new_forces, 
        type=2, 
        s=[ # TODO This parameter may be unnecessary
            self.space_divisions_z, 
            self.space_divisions_y, 
            self.space_divisions_x
        ])

        # Updating mode using the update rule in equation 8.
        # Relates to (2 * F^n) / (ω_i ^ 2) * (1 - cos(ω_i * Δ_t)) in equation 8.
        self.force_field = (
            (2 * self.forces) / ((self.omega_i) ** 2)) * (
                1 - np.cos(self.omega_i * self.sim_param.delta_t))

        # Edge case for first iteration according to Nikunj Raghuvanshi. p[n+1] = 2*p[n] – p[n-1] + (\delta t)^2 f[n], while f is impulse and p is pressure field.
        self.force_field[0, 0, 0] = 2 * self.M_current[0, 0, 0] - self.M_previous[0, 0, 0] + \
            self.sim_param.delta_t ** 2 * \
                self.impulses[t_s][0, 0, 0]

        # Relates to M^(n+1) in equation 8.
        self.M_next = 2 * self.M_current * \
        np.cos(self.omega_i * self.sim_param.delta_t) - self.M_previous + self.force_field
        
        # Convert modes to pressure values using inverse DCT.
        self.pressure_field = idctn(
            self.M_next.reshape(
                self.space_divisions_z, 
                self.space_divisions_y, 
                self.space_divisions_x
            ), type=2,
        s=[ # TODO This parameter may be unnecessary
            self.space_divisions_z, 
            self.space_divisions_y, 
            self.space_divisions_x
        ])
        
        self.pressure_field_results.append(self.pressure_field.copy())
        
        # Update time stepping to prepare for next time step / loop iteration.
        self.M_previous = self.M_current.copy()
        self.M_current = self.M_next.copy()

        # Update impulses
        self.new_forces = self.impulses[t_s].copy()
