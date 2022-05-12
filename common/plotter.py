import string
from common.serializer import Serializer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class Plotter():

    def __init__(self, verbose: bool=False):
        self.sim_param = None
        self.partitions = None
        self.verbose=verbose

    def set_data_from_file(self, file_name: string):
        serializer = Serializer(compress=True)
        (sim_params, partitions) = serializer.read(file_name)
        self.sim_param = sim_params
        self.partitions = partitions
        
    def set_data_from_simulation(self, sim_params, partitions):
        self.sim_param = sim_params
        self.partitions = partitions


    def plot_1D(self):
        # TODO Implement
        pass


    def plot_2D(self):
        partition_1 = self.partitions[0]
        partition_2 = self.partitions[1]
        #partition_3 = self.partitions[2]
        

        room_dims = np.linspace(0., partition_1.dimensions[0], len(partition_1.pressure_field_results[0]))
        ytop = np.max(partition_1.pressure_field_results)
        ybtm = np.min(partition_1.pressure_field_results)

        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax_1 = fig.add_subplot(2, 2, 1)
        ax_2 = fig.add_subplot(2, 2, 2)
        #ax_3 = fig.add_subplot(2, 2, 4)

        # TODO Make plots dynamic (just if we can make time at some point)

        temp_X_1 = np.linspace(0, partition_1.space_divisions_x, partition_1.space_divisions_x)
        temp_Y_1 = np.linspace(0, partition_1.space_divisions_y, partition_1.space_divisions_y)
        X_1, Y_1 = np.meshgrid(temp_X_1, temp_Y_1)

        temp_X_2 = np.linspace(0, partition_2.space_divisions_x, partition_2.space_divisions_x)
        temp_Y_2 = np.linspace(0, partition_2.space_divisions_y, partition_2.space_divisions_y)
        X_2, Y_2 = np.meshgrid(temp_X_2, temp_Y_2)

        # temp_X_3 = np.linspace(0, partition_3.space_divisions_x, partition_3.space_divisions_x)
        #temp_Y_3 = np.linspace(0, partition_3.space_divisions_y, partition_3.space_divisions_y)
        #X_3, Y_3 = np.meshgrid(temp_X_3, temp_Y_3)

        plot_limit_min = np.min(partition_2.pressure_field_results[:])
        plot_limit_max = np.max(partition_2.pressure_field_results[:])

        for i in range(0, len(partition_1.pressure_field_results), 50):
            Z_1 = partition_1.pressure_field_results[i]
            Z_2 = partition_2.pressure_field_results[i]
            #   Z_3 = partition_3.pressure_field_results[i]

            ax_1.cla()
            ax_2.cla()
            #  ax_3.cla()

            plt.title(f"t = {(self.sim_param.T * (i / self.sim_param.number_of_samples)):.4f}s")

            image1 = ax_1.imshow(Z_1)
            ax_2.imshow(Z_2)
            # ax_3.imshow(Z_3)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
            bar = fig.colorbar(image1, cax=cbar_ax)
            plt.pause(0.1)
            bar.remove()

        plot_step = 1

    def plot_3D(self):
        partition_1 = self.partitions[0]
        partition_2 = self.partitions[1]
        # partition_3 = self.partitions[2]

        room_dims = np.linspace(0., partition_1.dimensions[0], len(partition_1.pressure_field_results[0]))
        ytop = np.max(partition_1.pressure_field_results)
        ybtm = np.min(partition_1.pressure_field_results)

        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax_1 = fig.add_subplot(2, 2, 1)
        ax_2 = fig.add_subplot(2, 2, 2)
        # ax_3 = fig.add_subplot(2, 2, 4)

        temp_X_1 = np.linspace(0, partition_1.space_divisions_x, partition_1.space_divisions_x)
        temp_Y_1 = np.linspace(0, partition_1.space_divisions_y, partition_1.space_divisions_y)
        X_1, Y_1 = np.meshgrid(temp_X_1, temp_Y_1)
        
        temp_X_2 = np.linspace(0, partition_2.space_divisions_x, partition_2.space_divisions_x)
        temp_Y_2 = np.linspace(0, partition_2.space_divisions_y, partition_2.space_divisions_y)
        X_2, Y_2 = np.meshgrid(temp_X_2, temp_Y_2)
        
        # temp_X_3 = np.linspace(0, partition_3.space_divisions_x, partition_3.space_divisions_x)
        # temp_Y_3 = np.linspace(0, partition_3.space_divisions_y, partition_3.space_divisions_y)
        # X_3, Y_3 = np.meshgrid(temp_X_3, temp_Y_3)
        
        plot_limit_min = np.min(partition_1.pressure_field_results[:])
        plot_limit_max = np.max(partition_1.pressure_field_results[:])

        vertical_slice = int(partition_1.space_divisions_z / 2) # middle of Z axis

        for i in range(0, len(partition_1.pressure_field_results), 50):
            Z_1 = partition_1.pressure_field_results[i][vertical_slice]
            Z_2 = partition_2.pressure_field_results[i][vertical_slice]
            # Z_3 = partition_3.pressure_field_results[i][vertical_slice]

            ax_1.cla()
            ax_2.cla()
            # ax_3.cla()

            plt.title(f"t = {(self.sim_param.T * (i / self.sim_param.number_of_samples)):.4f}s")

            ax_1.imshow(Z_1)
            ax_2.imshow(Z_2)
            # ax_3.imshow(Z_3)

            plt.pause(0.005)

        plot_step = 100


    def plot(self):
        dimension = len(self.partitions[0].dimensions)
        if self.verbose: 
            print(f"Data is {dimension}-D.")
        if dimension == 1: 
            self.plot_1D()
        if dimension == 2: 
            self.plot_2D()
        if dimension == 3: 
            self.plot_3D()

class AnimationPlotter():
    
    @staticmethod
    def plot_3D(p_field_t, simulation_parameters,title='', interval=0, video_output=False, file_name='', zyx=None):
        plt.close() # close any existing plots from runs before
        # xyz is e.g, source location
        if zyx is not None:
            (z,y,x) = zyx
            fig, (X,Y,Z) = plt.subplots(nrows=3, ncols=1, figsize=(10,10))
            
            # for each time step we slice existing p field
            # X-slice=> YZ-Plane
            pX = [p_field_t[i][:,:,x] for i in range(len(p_field_t))]
            # Y-slice=> ZX-Plane
            pY = [p_field_t[i][:,y,:] for i in range(len(p_field_t))]
            pZ = [p_field_t[i][z,:,:] for i in range(len(p_field_t))]
            
            fig.suptitle(title, fontsize=14, fontweight='bold')
            # mi = np.min([p_field_t])
            # ma = np.max([p_field_t])
            
            # text = fig.text(0.95, 0.01, '',
            #         verticalalignment='top', horizontalalignment='right',
            #         transform=fig.transAxes,
            #         color='green', fontsize=15)
            text = fig.text(0.1, 0.9, '', # X, Y; 1-top or right
                    verticalalignment='center', horizontalalignment='center',
                    color='green', fontsize=15)
            
            k = np.max([np.min(np.abs([p_field_t])),np.max(np.abs([p_field_t]))])
            k= 0.5*k
            ma = k
            mi = -k
        
            colormap = ['Greys','seismic','coolwarm','twilight'][1]
            imX = X.imshow(np.zeros_like(pX[0]), vmin=mi, vmax=ma,aspect='equal',cmap=colormap)
            imY = Y.imshow(np.zeros_like(pY[0]), vmin=mi, vmax=ma,aspect='equal',cmap=colormap)
            imZ = Z.imshow(np.zeros_like(pZ[0]), vmin=mi, vmax=ma,aspect='equal',cmap=colormap)

            # Color Bar
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
            fig.colorbar(imX, cax=cbar_ax)
                        
            def init_func():
                X.set_title('YZ-Plane')
                Y.set_title('ZX-Plane')
                Z.set_title('XY-Plane')
                
            def update_plot(time_step):
                time = simulation_parameters.delta_t * time_step       
                # fig.suptitle("Time: %.2f sec" % time)
                text.set_text("Time: %.2f sec" % time)
                imX.set_data(pX[time_step])
                imY.set_data(pY[time_step])
                imZ.set_data(pZ[time_step])
                return [imX,imY,imZ]
            
            # keep the reference
            anim = FuncAnimation(   fig,
                                    update_plot,
                                    frames=range(simulation_parameters.number_of_samples),
                                    init_func=init_func,
                                    interval=interval, # Delay between frames in milliseconds
                                    blit=False)
            if video_output:
                AnimationPlotter.write_video(anim, file_name)
            return anim
        else:
            # surface plotter
            pass


    @staticmethod
    def plot_2D(p_field_t, simulation_parameters, interval=0, video_output=False, file_name=''):      

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        
        fig.suptitle("Time: %.2f sec" % 0)
    
        mi = np.min(-np.abs([p_field_t]))
        ma = np.max(np.abs([p_field_t]))
        
    
        im = ax.imshow(np.zeros_like(p_field_t[0]),vmin=mi, vmax=ma)
        
        # Color Bar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        
        def init_func():
            pass
        
        def update_plot(time_step):
            time = simulation_parameters.dt * time_step       
            fig.suptitle("Time: %.2f sec" % time)
            im.set_data(p_field_t[time_step])
            return [im]
        
        # keep the reference
        anim = FuncAnimation(   fig,
                                update_plot,
                                frames=simulation_parameters.time_steps,
                                init_func=init_func,
                                interval=interval, # Delay between frames in milliseconds
                                blit=False)
        if video_output:
            AnimationPlotter.write_video(anim, file_name)
        return anim
       
    def plot_1D(p_field_t, simulation_parameters, frames, interval=0, video_output=False, file_name=''):
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        
        fig.suptitle("Time: %.2f sec" % 0)

        mi = np.min(-np.abs([p_field_t]))
        ma = np.max(np.abs([p_field_t]))

        ln, = ax.plot(0,0)
        def init_func():
            ax.set_ylim(mi,ma)
            ax.set_xlim(0,len(p_field_t[0]))
            ln.set_xdata(np.arange(len(p_field_t[0])))
            
        def update_plot(time_step):
            time = simulation_parameters.dt * time_step       
            fig.suptitle("Time: %.2f sec" % time)
            # ln.set_xdata(np.arange(len(p_field_t[time_step])))
            ln.set_ydata(p_field_t[time_step])
            return [ln]
        
        # keep the reference
        anim = FuncAnimation(   fig,
                                update_plot,
                                frames=simulation_parameters.time_steps,
                                init_func=init_func,
                                interval=interval, # Delay between frames in milliseconds
                                blit=False)
        if video_output:
            AnimationPlotter.write_video(anim, file_name)
        return anim
        
    @staticmethod
    def write_video(anim, file_name):
        
        from matplotlib.animation import FFMpegWriter
        from datetime import datetime

        writervideo = FFMpegWriter(fps=60)
        fileloc = "videos/"
        filename  = file_name + '_'+ datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".mp4"
        anim.save(fileloc+filename,
                  dpi=300,
                  # fps=60,
                  writer=writervideo) 
        
    def plot(self):
        dimension = 3
        if self.verbose: 
            print(f"Data is {dimension}-D.")
        if dimension == 1: 
            self.plot_1D()
        if dimension == 2: 
            self.plot_2D()
        if dimension == 3: 
            self.plot_3D()