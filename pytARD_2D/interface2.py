import numpy as np
import enum

class inf_type(enum.Enum):
    '''
    inf_type in which sound waves traverse the interface.
    '''
    HORIZONTAL = 0
    VERTICAL = 1
    
class Interface2D():
      
    def __init__(self, width, partL, partR, inf_type, sim_params):
        self.partL = partL
        self.partR = partR
        self.inf_type = inf_type
        # todo extend
        if self.inf_type == inf_type.HORIZONTAL:
            self.dy = width
            self.dx = min(partL.dx, partR.dx)

        # 2D FDTD coefficents array. Normalize FDTD coefficents with space divisions and speed of sound. 
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
        self.FDTD_COEFFS_X = fdtd_coeffs_not_normalized * ((sim_params.c / sim_params.dx) ** 2)
        self.FDTD_COEFFS_Y = fdtd_coeffs_not_normalized * ((sim_params.c / sim_params.dy) ** 2)

        # FDTD kernel size.
        self.FDTD_KERNEL_SIZE = int((len(fdtd_coeffs_not_normalized[0])) / 2) 

    def handle_interface(self):
        '''
        TODO: Doc
        '''
        if self.inf_type == inf_type.VERTICAL:
            for y in range(self.partL.space_divisions_y):
                pressure_field_around_interface_y = np.zeros(shape=[2 * self.FDTD_KERNEL_SIZE, 1])

                # Left room
                pressure_field_around_interface_y[0 : self.FDTD_KERNEL_SIZE] = self.partL.pressure_field[y, -self.FDTD_KERNEL_SIZE : ].copy().reshape([self.FDTD_KERNEL_SIZE, 1])

                # Right room
                pressure_field_around_interface_y[self.FDTD_KERNEL_SIZE : 2 * self.FDTD_KERNEL_SIZE] = self.partR.pressure_field[y, 0 : self.FDTD_KERNEL_SIZE].copy().reshape(self.FDTD_KERNEL_SIZE, 1)

                # Calculate new forces transmitted into room
                new_forces_from_interface_y = self.FDTD_COEFFS_Y.dot(pressure_field_around_interface_y)

                # Add everything together
                self.partL.new_forces[y, -3] += new_forces_from_interface_y[0]
                self.partL.new_forces[y, -2] += new_forces_from_interface_y[1]
                self.partL.new_forces[y, -1] += new_forces_from_interface_y[2]
                self.partR.new_forces[y, 0] += new_forces_from_interface_y[3]
                self.partR.new_forces[y, 1] += new_forces_from_interface_y[4]
                self.partR.new_forces[y, 2] += new_forces_from_interface_y[5]
    
        elif self.inf_type == inf_type.HORIZONTAL:
            for x in range(self.partL.space_divisions_x):
                pressure_field_around_interface_x = np.zeros(shape=[2 * self.FDTD_KERNEL_SIZE, 1])

                # Right top room
                pressure_field_around_interface_x[0 : self.FDTD_KERNEL_SIZE] = self.partL.pressure_field[-self.FDTD_KERNEL_SIZE : , x].copy().reshape([self.FDTD_KERNEL_SIZE, 1])

                # Right bottom room
                pressure_field_around_interface_x[self.FDTD_KERNEL_SIZE : 2 * self.FDTD_KERNEL_SIZE] = self.partR.pressure_field[0 : self.FDTD_KERNEL_SIZE, x].copy().reshape(self.FDTD_KERNEL_SIZE, 1)

                # Calculate new forces transmitted into room
                new_forces_from_interface_x = self.FDTD_COEFFS_X.dot(pressure_field_around_interface_x)

                # Add everything together
                self.partL.new_forces[-3, x] += new_forces_from_interface_x[0]
                self.partL.new_forces[-2, x] += new_forces_from_interface_x[1]
                self.partL.new_forces[-1, x] += new_forces_from_interface_x[2]
                self.partR.new_forces[0, x] += new_forces_from_interface_x[3]
                self.partR.new_forces[1, x] += new_forces_from_interface_x[4]
                self.partR.new_forces[2, x] += new_forces_from_interface_x[5]