import numpy as np
from common.finite_differences import get_laplacian_matrix
    
class InterfaceData1D():
    def __init__(self, part1_index, part2_index, fdtd_acc=6):
        self.part1_index = part1_index
        self.part2_index = part2_index
        self.fdtd_acc = fdtd_acc

class Interface1D():

    def __init__(self, sim_params, part_data, fdtd_order=2, fdtd_acc=6):
        '''
        TODO: Doc
        '''

        self.part_data = part_data

        # 1D FDTD coefficents array. Normalize FDTD coefficents with space divisions and speed of sound. 
        fdtd_coeffs_not_normalized = get_laplacian_matrix(fdtd_order, fdtd_acc)

        # TODO: Unify h of partition data, atm it's hard coded to first partition
        # TODO: In the papers it's supposed to be multiplied by *(c/h)**2
        self.FDTD_COEFFS = fdtd_coeffs_not_normalized * (sim_params.c / part_data[0].h) 

        # Interface size derived from FDTD kernel size.
        # If FDTD Accuracy is for example 6, then the interface is 3 on each
        # side of the interface (3 voxels left, 3 voxels right)
        self.INTERFACE_SIZE = int((len(fdtd_coeffs_not_normalized[0])) / 2) 

    def handle_interface(self, interface_data):
        '''
        TODO: Doc
        '''
        pressure_field_around_interface = np.zeros(shape=[2 * self.INTERFACE_SIZE])

        # Left rod
        pressure_field_around_interface[0 : self.INTERFACE_SIZE] = self.part_data[interface_data.part1_index].pressure_field[ -self.INTERFACE_SIZE : ].copy()

        # Right rod
        pressure_field_around_interface[self.INTERFACE_SIZE : 2 * self.INTERFACE_SIZE] = self.part_data[interface_data.part2_index].pressure_field[0 : self.INTERFACE_SIZE].copy()

        new_forces_from_interface = np.matmul(pressure_field_around_interface, self.FDTD_COEFFS)

        # Add everything together
        self.part_data[interface_data.part1_index].new_forces[-self.INTERFACE_SIZE:] += new_forces_from_interface[0:self.INTERFACE_SIZE]
        self.part_data[interface_data.part2_index].new_forces[:self.INTERFACE_SIZE] += new_forces_from_interface[self.INTERFACE_SIZE : self.INTERFACE_SIZE * 2]
