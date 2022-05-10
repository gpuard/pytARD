from common.parameters import SimulationParameters
from common.finite_differences import get_laplacian_matrix

import numpy as np
import enum

class Direction3D(enum.Enum):
    '''
    Direction in which sound waves traverse the interface.
    '''
    X = 'HORIZONTAL'
    Y = 'VERTICAL'
    Z = 'HEIGHT'
    
class InterfaceData3D():
    def __init__(self, part1_index: int, part2_index: int, direction: Direction3D,looped=False):
        self.part1_index: int = part1_index
        self.part2_index: int = part2_index
        self.direction: Direction3D = direction
        self.looped = False

class Interface3D():

    def __init__(self, sim_params: SimulationParameters, part_data, fdtd_order: int=2, fdtd_acc: int=6):
        '''
        TODO: Doc
        '''

        self.part_data = part_data

        # 2D FDTD coefficents array. Normalize FDTD coefficents with space divisions and speed of sound. 
        fdtd_coeffs_not_normalized = get_laplacian_matrix(fdtd_order, fdtd_acc)
        
        # TODO: Unify h of partition data, atm it's hard coded to first partition
        # Important: For each direction the sound passes through an interface, the according FDTD coeffs should be used.
        self.FDTD_COEFFS_X = fdtd_coeffs_not_normalized * ((sim_params.c / part_data[0].h_x) ** 2)
        self.FDTD_COEFFS_Y = fdtd_coeffs_not_normalized * ((sim_params.c / part_data[0].h_y) ** 2)
        self.FDTD_COEFFS_Z = fdtd_coeffs_not_normalized * ((sim_params.c / part_data[0].h_z) ** 2)

        # FDTD kernel size.
        self.INTERFACE_SIZE = int((len(fdtd_coeffs_not_normalized[0])) / 2) 

    def handle_interface(self, interface_data):
        '''
        TODO: Doc
        '''
        if interface_data.direction == Direction3D.X:
            p_x0 = self.part_data[interface_data.part1_index].pressure_field[:, :, -self.INTERFACE_SIZE:]
            p_x1 = self.part_data[interface_data.part2_index].pressure_field[:, :, :self.INTERFACE_SIZE]

            # Calculate new forces transmitted into room
            p_along_yz = np.concatenate((p_x0, p_x1),axis=2)
            new_forces_from_interface_y = np.matmul(p_along_yz, self.FDTD_COEFFS_Y)
            # (60,60,6)
            #new_forces_from_interface_y = np.einsum('kk,ijk->ijk', self.FDTD_COEFFS_Y,p_along_yz) # WORKS too!!

            # Add everything together
            self.part_data[interface_data.part1_index].new_forces[:, :, -self.INTERFACE_SIZE:] += new_forces_from_interface_y[:, :, :self.INTERFACE_SIZE]
            self.part_data[interface_data.part2_index].new_forces[:, :, :self.INTERFACE_SIZE] += new_forces_from_interface_y[:, :, -self.INTERFACE_SIZE :]

        elif interface_data.direction == Direction3D.Y:
            p_y0 = self.part_data[interface_data.part1_index].pressure_field[:, -self.INTERFACE_SIZE :, :]
            p_y1 = self.part_data[interface_data.part2_index].pressure_field[:, :self.INTERFACE_SIZE, :]

            # Calculate new forces transmitted into room
            p_along_zx = np.concatenate((p_y0, p_y1),axis=1)
            new_forces_from_interface_x = np.matmul(self.FDTD_COEFFS_X, p_along_zx)

            # Add everything together
            self.part_data[interface_data.part1_index].new_forces[:, -self.INTERFACE_SIZE :, :] += new_forces_from_interface_x[:, :self.INTERFACE_SIZE, :]
            self.part_data[interface_data.part2_index].new_forces[:, :self.INTERFACE_SIZE, :] += new_forces_from_interface_x[:, -self.INTERFACE_SIZE :, :]

        elif interface_data.direction == Direction3D.Z:
            p_z0 = self.part_data[interface_data.part1_index].pressure_field[-self.INTERFACE_SIZE:, :, :]
            p_z1 = self.part_data[interface_data.part2_index].pressure_field[:self.INTERFACE_SIZE, :, :]

            # Calculate new forces transmitted into room
            p_along_xy = np.concatenate((p_z0, p_z1),axis=0)
            # new_forces_from_interface_z = np.matmul(p_along_xy,self.FDTD_COEFFS_Z)
            # (6, 60, 60) x (6, 6) -> (6, 60, 60)
            # new_forces_from_interface_z = np.einsum('ijk,ii->ijk', p_along_xy, self.FDTD_COEFFS_Z)
            new_forces_from_interface_z = np.einsum('ii,ijk->ijk', self.FDTD_COEFFS_Y,p_along_xy)
            # new_forces_from_interface_z = np.tensordot(p_along_xy,self.FDTD_COEFFS_Z, axes=([1,0],[0,1]))

            # Add everything together
            self.part_data[interface_data.part1_index].new_forces[-self.INTERFACE_SIZE:, :, :] += new_forces_from_interface_z[:self.INTERFACE_SIZE, :, :]
            self.part_data[interface_data.part2_index].new_forces[:self.INTERFACE_SIZE, :, :] += new_forces_from_interface_z[-self.INTERFACE_SIZE:, :, :]

# ───────▄██████████████████▄───────
# ────▄███████████████████████▄─────
# ───███████████████████████████────
# ──█████████████████████████████───
# ─████████████▀─────────▀████████──
# ██████████▀───────────────▀██████─
# ███████▀────────────────────█████▌
# ██████───▄▀▀▀▀▄──────▄▀▀▀▀▄──█████
# █████▀──────────────────▄▄▄───████
# ████────▄█████▄───────▄█▀▀▀█▄──██▀
# ████──▄█▀────▀██─────█▀────────█▀─
# ─▀██───────────▀────────▄███▄──██─
# ──██───▄▄██▀█▄──▀▄▄▄▀─▄██▄▀────███
# ▄███────▀▀▀▀▀──────────────▄▄──██▐
# █▄▀█──▀▀▀▄▄▄▀▀───────▀▀▄▄▄▀────█▌▐
# █▐─█────────────▄───▄──────────█▌▐
# █▐─▀───────▐──▄▀─────▀▄──▌─────██▐
# █─▀────────▌──▀▄─────▄▀──▐─────██▀
# ▀█─█──────▐─────▀▀▄▀▀─────▌────█──
# ─▀█▀───────▄────────────▄──────█──
# ───█─────▄▀──▄█████████▄─▀▄───▄█──
# ───█────█──▄██▀░░░░░░░▀██▄─█──█───
# ───█▄───▀▄──▀██▄█████▄██▀─▄▀─▄█───
# ────█▄────▀───▀▀▀▀──▀▀▀──▀──▄█────
# ─────█▄────────▄▀▀▀▀▀▄─────▄█─────
# ──────███▄──────────────▄▄██──────
# ─────▄█─▀█████▄▄────▄▄████▀█▄─────
# ────▄█───────▀▀██████▀▀─────█▄────
# ───▄█─────▄▀───────────▀▄────█▄───
# ──▄█─────▀───────────────▀────█▄──
# ──────────────────────────────────
# ▐▌▐█▄█▌▐▀▀█▐▀▀▌─█▀─█▀─▐▌▐▀█▐▀█─█─█
# ▐▌▐─▀─▌▐▀▀▀▐──▌─▀█─▀█─▐▌▐▀▄▐▀▄─█─█
# ▐▌▐───▌▐───▐▄▄▌─▄█─▄█─▐▌▐▄█▐─█─█▄█


class Interface3DLooped():

    def __init__(self, sim_params, part_data, fdtd_order=2, fdtd_acc=6):
        '''
        TODO: Doc
        '''

        self.part_data = part_data

        # 2D FDTD coefficents array. Normalize FDTD coefficents with space divisions and speed of sound. 
        fdtd_coeffs_not_normalized = get_laplacian_matrix(fdtd_order, fdtd_acc)
        
        # TODO: Unify h of partition data, atm it's hard coded to first partition
        # Important: For each direction the sound passes through an interface, the according FDTD coeffs should be used.
        self.FDTD_COEFFS_X = fdtd_coeffs_not_normalized * ((sim_params.c / part_data[0].h_x) ** 2)
        self.FDTD_COEFFS_Y = fdtd_coeffs_not_normalized * ((sim_params.c / part_data[0].h_y) ** 2)
        self.FDTD_COEFFS_Z = fdtd_coeffs_not_normalized * ((sim_params.c / part_data[0].h_z) ** 2)

        # FDTD kernel size.
        self.FDTD_KERNEL_SIZE = int((len(fdtd_coeffs_not_normalized[0])) / 2) 

    def handle_interface(self, interface_data):
        '''
        TODO: Doc
        '''
        if interface_data.direction == Direction3D.X:
            for z in range(self.part_data[interface_data.part1_index].space_divisions_z):
                for y in range(self.part_data[interface_data.part1_index].space_divisions_y):
                    pressure_field_around_interface_y = np.zeros(shape=[2 * self.FDTD_KERNEL_SIZE])
                    pressure_field_around_interface_y[0 : self.FDTD_KERNEL_SIZE] = self.part_data[interface_data.part1_index].pressure_field[z, y, -self.FDTD_KERNEL_SIZE : ].copy()#.reshape([self.FDTD_KERNEL_SIZE, 1])
                    pressure_field_around_interface_y[self.FDTD_KERNEL_SIZE : 2 * self.FDTD_KERNEL_SIZE] = self.part_data[interface_data.part2_index].pressure_field[z, y, 0 : self.FDTD_KERNEL_SIZE].copy()#.reshape(self.FDTD_KERNEL_SIZE, 1)

                    # Calculate new forces transmitted into room
                    new_forces_from_interface_y = self.FDTD_COEFFS_Y.dot(pressure_field_around_interface_y)

                    # Add everything together
                    self.part_data[interface_data.part1_index].new_forces[z, y, -3] += new_forces_from_interface_y[0]
                    self.part_data[interface_data.part1_index].new_forces[z, y, -2] += new_forces_from_interface_y[1]
                    self.part_data[interface_data.part1_index].new_forces[z, y, -1] += new_forces_from_interface_y[2]
                    self.part_data[interface_data.part2_index].new_forces[z, y, 0] += new_forces_from_interface_y[3]
                    self.part_data[interface_data.part2_index].new_forces[z, y, 1] += new_forces_from_interface_y[4]
                    self.part_data[interface_data.part2_index].new_forces[z, y, 2] += new_forces_from_interface_y[5]
    
        elif interface_data.direction == Direction3D.Y:
            for z in range(self.part_data[interface_data.part1_index].space_divisions_z):
                for x in range(self.part_data[interface_data.part1_index].space_divisions_x):
                    pressure_field_around_interface_x = np.zeros(shape=[2 * self.FDTD_KERNEL_SIZE])
                    pressure_field_around_interface_x[0 : self.FDTD_KERNEL_SIZE] = self.part_data[1].pressure_field[z, -self.FDTD_KERNEL_SIZE : , x].copy()#.reshape([self.FDTD_KERNEL_SIZE, 1])
                    pressure_field_around_interface_x[self.FDTD_KERNEL_SIZE : 2 * self.FDTD_KERNEL_SIZE] = self.part_data[2].pressure_field[z, 0 : self.FDTD_KERNEL_SIZE, x].copy()#.reshape(self.FDTD_KERNEL_SIZE, 1)

                    # Calculate new forces transmitted into  room
                    new_forces_from_interface_x = self.FDTD_COEFFS_X.dot(pressure_field_around_interface_x)

                    # Add everything together
                    self.part_data[interface_data.part1_index].new_forces[z, -3, x] += new_forces_from_interface_x[0]
                    self.part_data[interface_data.part1_index].new_forces[z, -2, x] += new_forces_from_interface_x[1]
                    self.part_data[interface_data.part1_index].new_forces[z, -1, x] += new_forces_from_interface_x[2]
                    self.part_data[interface_data.part2_index].new_forces[z, 0, x] += new_forces_from_interface_x[3]
                    self.part_data[interface_data.part2_index].new_forces[z, 1, x] += new_forces_from_interface_x[4]
                    self.part_data[interface_data.part2_index].new_forces[z, 2, x] += new_forces_from_interface_x[5]
                    