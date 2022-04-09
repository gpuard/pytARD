from common.serializer import Serializer
import matplotlib.pyplot as plt
import numpy as np

class Plotter():

    def __init__(self, verbose=False):
        self.sim_params = None
        self.partitions = None
        self.verbose=verbose

    def set_data_from_file(self, file_name):
        serializer = Serializer(compressed=True)
        (sim_params, partitions) = serializer.read(file_name)
        self.sim_params = sim_params
        self.partitions = partitions
        
    def set_data_from_simulation(self, sim_params, partitions):
        self.sim_params = sim_params
        self.partitions = partitions


    def plot_1D(self):
        # TODO Implement
        pass


    def plot_2D(self):
        
        partition_1 = self.partitions[0]
        partition_2 = self.partitions[1]
        partition_3 = self.partitions[2]
        

        room_dims = np.linspace(0., partition_1.dimensions[0], len(partition_1.pressure_field_results[0]))
        ytop = np.max(partition_1.pressure_field_results)
        ybtm = np.min(partition_1.pressure_field_results)

        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax_1 = fig.add_subplot(2, 2, 1)
        ax_2 = fig.add_subplot(2, 2, 2)
        ax_3 = fig.add_subplot(2, 2, 4)

        # TODO Make plots dynamic (just if we can make time at some point)

        temp_X_1 = np.linspace(0, partition_1.space_divisions_x, partition_1.space_divisions_x)
        temp_Y_1 = np.linspace(0, partition_1.space_divisions_y, partition_1.space_divisions_y)
        X_1, Y_1 = np.meshgrid(temp_X_1, temp_Y_1)

        temp_X_2 = np.linspace(0, partition_2.space_divisions_x, partition_2.space_divisions_x)
        temp_Y_2 = np.linspace(0, partition_2.space_divisions_y, partition_2.space_divisions_y)
        X_2, Y_2 = np.meshgrid(temp_X_2, temp_Y_2)

        temp_X_3 = np.linspace(0, partition_3.space_divisions_x, partition_3.space_divisions_x)
        temp_Y_3 = np.linspace(0, partition_3.space_divisions_y, partition_3.space_divisions_y)
        X_3, Y_3 = np.meshgrid(temp_X_3, temp_Y_3)

        plot_limit_min = np.min(partition_2.pressure_field_results[:])
        plot_limit_max = np.max(partition_2.pressure_field_results[:])

        for i in range(0, len(partition_1.pressure_field_results), 50):
            Z_1 = partition_1.pressure_field_results[i]
            Z_2 = partition_2.pressure_field_results[i]
            Z_3 = partition_3.pressure_field_results[i]

            ax_1.cla()
            ax_2.cla()
            ax_3.cla()

            plt.title(f"t = {(self.sim_params.T * (i / self.sim_params.number_of_samples)):.4f}s")

            ax_1.imshow(Z_1)
            ax_2.imshow(Z_2)
            ax_3.imshow(Z_3)

            plt.pause(0.005)

        plot_step = 100

    def plot_3D(self):
        # TODO Implement
        pass


    def plot(self, mic_posi):
        # TODO Check which dimension it is and call corresponding plot function
        dimension = len(self.partitions[0].dimensions)
        if self.verbose: 
            print(f"Data is {dimension}-D.")
        if dimension == 1: 
            self.plot_1D()
        if dimension == 2: 
            self.plot_2D()
        if dimension == 3: 
            self.plot_3D()