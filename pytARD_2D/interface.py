import numpy as np
import enum

from common.parameters import SimulationParameters
from common.finite_differences import get_laplacian_matrix
from pytARD_2D.partition import Partition

class Direction2D(enum.Enum):
    '''
    Direction in which sound waves traverse the interface.
    '''
    X = 'HORIZONTAL'
    Y = 'VERTICAL'
    
class InterfaceData2D():
    def __init__(self, part1_index, part2_index, direction):
        self.part1_index: int = part1_index
        self.part2_index: int = part2_index
        self.direction: Direction2D = direction

class Interface2D():

    def __init__(self, sim_params: SimulationParameters, part_data: Partition, fdtd_order=2, fdtd_acc=6):
        '''
        TODO: Doc
        '''

        self.part_data = part_data

        # 2D FDTD coefficents array. Normalize FDTD coefficents with space divisions and speed of sound. 
        fdtd_coeffs_not_normalized = get_laplacian_matrix(fdtd_order, fdtd_acc)

        # TODO: Unify h of partition data, atm it's hard coded to first partition
        self.FDTD_COEFFS_X = fdtd_coeffs_not_normalized * ((sim_params.c / part_data[0].h_x) ** 2)
        self.FDTD_COEFFS_Y = fdtd_coeffs_not_normalized * ((sim_params.c / part_data[0].h_y) ** 2)

        # FDTD kernel size.
        self.INTERFACE_SIZE = int((len(fdtd_coeffs_not_normalized[0])) / 2) 

    def handle_interface(self, interface_data):
        '''
        TODO: Doc
        '''
        if interface_data.direction == Direction2D.X:
            p_left = self.part_data[interface_data.part1_index].pressure_field[:, -self.INTERFACE_SIZE:]
            p_right = self.part_data[interface_data.part2_index].pressure_field[:, :self.INTERFACE_SIZE]

            # Calculate new forces transmitted into room
            pressures_along_y = np.hstack((p_left, p_right))
            new_forces_from_interface_y = np.matmul(pressures_along_y, self.FDTD_COEFFS_Y)

            # Add everything together
            self.part_data[interface_data.part1_index].new_forces[:, -self.INTERFACE_SIZE:] += new_forces_from_interface_y[:, :self.INTERFACE_SIZE]
            self.part_data[interface_data.part2_index].new_forces[:, :self.INTERFACE_SIZE] += new_forces_from_interface_y[:, -self.INTERFACE_SIZE:]

        elif interface_data.direction == Direction2D.Y:
            p_top = self.part_data[interface_data.part1_index].pressure_field[-self.INTERFACE_SIZE:, :]
            p_bot = self.part_data[interface_data.part2_index].pressure_field[:self.INTERFACE_SIZE, :]

            # Calculate new forces transmitted into room
            pressures_along_x = np.vstack((p_top, p_bot))
            new_forces_from_interface_x = np.matmul(self.FDTD_COEFFS_X, pressures_along_x)

            # Add everything together
            self.part_data[interface_data.part1_index].new_forces[-self.INTERFACE_SIZE:, :] += new_forces_from_interface_x[:self.INTERFACE_SIZE, :]
            self.part_data[interface_data.part2_index].new_forces[:self.INTERFACE_SIZE, :] += new_forces_from_interface_x[-self.INTERFACE_SIZE:, :]


class Interface2DLooped():
    '''
    Version of interfaces with for loops instead of matrix mult
    '''

    def __init__(self, sim_params: SimulationParameters, part_data: Partition, fdtd_order=2, fdtd_acc=6):
        '''
        TODO: Doc
        '''

        self.part_data = part_data

        # 2D FDTD coefficents array. Normalize FDTD coefficents with space divisions and speed of sound. 
        fdtd_coeffs_not_normalized = get_laplacian_matrix(fdtd_order, fdtd_acc)

        # TODO: Unify h of partition data, atm it's hard coded to first partition
        self.FDTD_COEFFS_X = fdtd_coeffs_not_normalized * ((sim_params.c / part_data[0].h_x) ** 2)
        self.FDTD_COEFFS_Y = fdtd_coeffs_not_normalized * ((sim_params.c / part_data[0].h_y) ** 2)

        # FDTD kernel size.
        self.INTERFACE_SIZE = int((len(fdtd_coeffs_not_normalized[0])) / 2) 

    def handle_interface(self, interface_data):
        '''
        TODO: Doc
        '''
        if interface_data.direction == Direction2D.X:
            for y in range(self.part_data[interface_data.part1_index].space_divisions_y):
                pressure_field_around_interface_y = np.zeros(shape=[2 * self.INTERFACE_SIZE])
                pressure_field_around_interface_y[0 : self.INTERFACE_SIZE] = self.part_data[interface_data.part1_index].pressure_field[y, -self.INTERFACE_SIZE : ].copy()
                pressure_field_around_interface_y[self.INTERFACE_SIZE : 2 * self.INTERFACE_SIZE] = self.part_data[interface_data.part2_index].pressure_field[y, 0 : self.INTERFACE_SIZE].copy()

                # Calculate new forces transmitted into room
                new_forces_from_interface_y = np.matmul(pressure_field_around_interface_y, self.FDTD_COEFFS_Y)

                # Add everything together
                self.part_data[interface_data.part1_index].new_forces[y, -self.INTERFACE_SIZE:] += new_forces_from_interface_y[0:self.INTERFACE_SIZE]
                self.part_data[interface_data.part2_index].new_forces[y, :self.INTERFACE_SIZE] += new_forces_from_interface_y[self.INTERFACE_SIZE : self.INTERFACE_SIZE * 2]
    

        elif interface_data.direction == Direction2D.Y:
            for x in range(self.part_data[interface_data.part1_index].space_divisions_x):
                pressure_field_around_interface_x = np.zeros(shape=[2 * self.INTERFACE_SIZE])
                pressure_field_around_interface_x[0 : self.INTERFACE_SIZE] = self.part_data[interface_data.part1_index].pressure_field[-self.INTERFACE_SIZE : , x].copy()
                pressure_field_around_interface_x[self.INTERFACE_SIZE : 2 * self.INTERFACE_SIZE] = self.part_data[interface_data.part2_index].pressure_field[0 : self.INTERFACE_SIZE, x].copy()

                # Calculate new forces transmitted into room
                new_forces_from_interface_x = np.matmul(pressure_field_around_interface_x, self.FDTD_COEFFS_X)

                # Add everything together
                self.part_data[interface_data.part1_index].new_forces[-self.INTERFACE_SIZE:, x] += new_forces_from_interface_x[0:self.INTERFACE_SIZE]
                self.part_data[interface_data.part2_index].new_forces[:self.INTERFACE_SIZE, x] += new_forces_from_interface_x[self.INTERFACE_SIZE : self.INTERFACE_SIZE * 2]
