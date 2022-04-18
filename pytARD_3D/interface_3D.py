import numpy as np
import enum

class Direction3D(enum.Enum):
    '''
    Direction in which sound waves traverse the interface.
    '''
    Z = ''
    Y = ''
    X = ''
    
class InterfaceData3D():
    def __init__(self, part1_index, part2_index, direction):
        self.part1_index = part1_index
        self.part2_index = part2_index
        self.direction = direction

class Interface3D():

    def __init__(self, sim_params, part_data):
        '''
        TODO: Doc
        '''

        self.part_data = part_data

        # 3D FDTD coefficents array. Normalize FDTD coefficents with space divisions and speed of sound. 
        fdtd_coeffs_not_normalized = np.array(
            [
                [-0.,         -0.,         -0.01111111,  0.01111111,  0.,          0.        ],
                [-0.,         -0.01111111,  0.15,       -0.15,        0.01111111,  0.        ],
                [-0.01111111,  0.15,       -1.5,         1.5,        -0.15,        0.01111111],
                [ 0.01111111, -0.15,        1.5,        -1.5,         0.15,       -0.01111111],
                [ 0.,          0.01111111, -0.15,        0.15,       -0.01111111, -0.        ],
                [ 0.,          0.,          0.01111111, -0.01111111, -0.,         -0.        ]
            ]
        )

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
        if interface_data.direction == Direction3D.Y:
            for z in range(self.part_data[interface_data.part1_index].space_divisions_z):
                for y in range(self.part_data[interface_data.part1_index].space_divisions_y):
                    pressure_field_around_interface_y = np.zeros(shape=[2 * self.FDTD_KERNEL_SIZE])
                    pressure_field_around_interface_y[0 : self.FDTD_KERNEL_SIZE] = self.part_data[interface_data.part1_index].pressure_field[z, y, -self.FDTD_KERNEL_SIZE : ].copy().reshape([self.FDTD_KERNEL_SIZE, 1])
                    pressure_field_around_interface_y[self.FDTD_KERNEL_SIZE : 2 * self.FDTD_KERNEL_SIZE] = self.part_data[interface_data.part2_index].pressure_field[z, y, 0 : self.FDTD_KERNEL_SIZE].copy().reshape(self.FDTD_KERNEL_SIZE, 1)

                    # Calculate new forces transmitted into room
                    new_forces_from_interface_y = self.FDTD_COEFFS_Y.dot(pressure_field_around_interface_y)

                    # Add everything together
                    self.part_data[interface_data.part1_index].new_forces[z, y, -3] += new_forces_from_interface_y[0]
                    self.part_data[interface_data.part1_index].new_forces[z, y, -2] += new_forces_from_interface_y[1]
                    self.part_data[interface_data.part1_index].new_forces[z, y, -1] += new_forces_from_interface_y[2]
                    self.part_data[interface_data.part2_index].new_forces[z, y, 0] += new_forces_from_interface_y[3]
                    self.part_data[interface_data.part2_index].new_forces[z, y, 1] += new_forces_from_interface_y[4]
                    self.part_data[interface_data.part2_index].new_forces[z, y, 2] += new_forces_from_interface_y[5]
    
        elif interface_data.direction == Direction3D.X:
            for z in range(self.part_data[interface_data.part1_index].space_divisions_z):
                for x in range(self.part_data[interface_data.part1_index].space_divisions_x):
                    pressure_field_around_interface_x = np.zeros(shape=[2 * self.FDTD_KERNEL_SIZE])
                    pressure_field_around_interface_x[0 : self.FDTD_KERNEL_SIZE] = self.part_data[1].pressure_field[z, -self.FDTD_KERNEL_SIZE : , x].copy().reshape([self.FDTD_KERNEL_SIZE, 1])
                    pressure_field_around_interface_x[self.FDTD_KERNEL_SIZE : 2 * self.FDTD_KERNEL_SIZE] = self.part_data[2].pressure_field[z, 0 : self.FDTD_KERNEL_SIZE, x].copy().reshape(self.FDTD_KERNEL_SIZE, 1)

                    # Calculate new forces transmitted into  room
                    new_forces_from_interface_x = self.FDTD_COEFFS_X.dot(pressure_field_around_interface_x)

                    # Add everything together
                    self.part_data[interface_data.part1_index].new_forces[z, -3, x] += new_forces_from_interface_x[0]
                    self.part_data[interface_data.part1_index].new_forces[z, -2, x] += new_forces_from_interface_x[1]
                    self.part_data[interface_data.part1_index].new_forces[z, -1, x] += new_forces_from_interface_x[2]
                    self.part_data[interface_data.part2_index].new_forces[z, 0, x] += new_forces_from_interface_x[3]
                    self.part_data[interface_data.part2_index].new_forces[z, 1, x] += new_forces_from_interface_x[4]
                    self.part_data[interface_data.part2_index].new_forces[z, 2, x] += new_forces_from_interface_x[5]