import numpy as np
    
class InterfaceData1D():
    def __init__(self, part1_index, part2_index):
        self.part1_index = part1_index
        self.part2_index = part2_index

class Interface1D():

    def __init__(self, sim_params, part_data):
        '''
        TODO: Doc
        '''

        self.part_data = part_data

        # 1D FDTD coefficents array. Normalize FDTD coefficents with space divisions and speed of sound. 
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
        self.FDTD_COEFFS = fdtd_coeffs_not_normalized * (sim_params.c / part_data[0].h) 

        # FDTD kernel size.
        self.FDTD_KERNEL_SIZE = int((len(fdtd_coeffs_not_normalized[0])) / 2) 

    def handle_interface(self, interface_data):
        '''
        TODO: Doc
        '''
        pressure_field_around_interface = np.zeros(shape=[2 * self.FDTD_KERNEL_SIZE])

        # Left rod
        pressure_field_around_interface[0 : self.FDTD_KERNEL_SIZE] = self.part_data[interface_data.part1_index].pressure_field[ -self.FDTD_KERNEL_SIZE : ].copy()

        # Right rod
        pressure_field_around_interface[self.FDTD_KERNEL_SIZE : 2 * self.FDTD_KERNEL_SIZE] = self.part_data[interface_data.part2_index].pressure_field[0 : self.FDTD_KERNEL_SIZE].copy()

        new_forces_from_interface = self.FDTD_COEFFS.dot(pressure_field_around_interface)

        # Add everything together
        self.part_data[interface_data.part1_index].new_forces[-3] += new_forces_from_interface[0]
        self.part_data[interface_data.part1_index].new_forces[-2] += new_forces_from_interface[1]
        self.part_data[interface_data.part1_index].new_forces[-1] += new_forces_from_interface[2]
        self.part_data[interface_data.part2_index].new_forces[0] += new_forces_from_interface[3]
        self.part_data[interface_data.part2_index].new_forces[1] += new_forces_from_interface[4]
        self.part_data[interface_data.part2_index].new_forces[2] += new_forces_from_interface[5]
