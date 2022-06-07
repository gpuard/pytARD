from common.parameters import SimulationParameters

import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt
from datetime import datetime


class AnimationPlotter():
    '''
    Plotting class with FFMPEG video support. 
    Caution: The plotter is static and can be a bit of a chore to display everything correctly.
    You have to edit the code for each setup you want to display.
    '''

    @staticmethod
    def plot_3D(pressure_field: np.ndarray, sim_param: SimulationParameters, title: str = '', interval=0, video_output: bool = False, file_name: str = '', source_zyx: tuple = None, direction: np.character = [None, 'x', 'y', 'z'][1]):
        '''
        Plots 3D domain in real-time with video output.

        Parameters
        ----------
        pressure_field : ndarray
            Pressure field data.
        sim_param : SimulationParameters
            Instance of simulation parameter class.
        title : str
            Title of the plot.
        interval : int
            Delay between frames in milliseconds.
        video_output: bool
            Displays the video on screen.
        file_name : str
            File name of video to write on disk.
        zyx : tuple
            Z, Y, X of source location.
        direction : char
            Direction. Either None, x, y or z.
        
        Returns
        -------
        tuple
            Animation and FuncAnimation instances.
        '''

        plt.close()  # close any existing plots

        if source_zyx is not None:
            (z, y, x) = source_zyx
            fig, (X, Y, Z) = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))

            animation = None
            if direction is not None:
                p_t = list()
                if direction == 'x':
                    for frame in pressure_field:
                        p_t.append(frame[z, y, :])
                if direction == 'y':
                    for frame in pressure_field:
                        p_t.append(frame[z, :, x])
                if direction == 'z':
                    for frame in pressure_field:
                        p_t.append(frame[:, y, x])

                animation = AnimationPlotter.plot_1D(
                    p_t, sim_param, interval=0, video_output=False, file_name='')

            p_x = [pressure_field[i][:, :, x] for i in range(len(pressure_field))]
            p_y = [pressure_field[i][:, y, :] for i in range(len(pressure_field))]
            p_z = [pressure_field[i][z, :, :] for i in range(len(pressure_field))]

            fig.suptitle(title, fontsize=14, fontweight='bold')
            text = fig.text(0.1, 0.9, '',  # X, Y; 1-top or right
                            verticalalignment='center', horizontalalignment='center',
                            color='green', fontsize=15)

            k = np.max([np.min(np.abs([pressure_field])),
                       np.max(np.abs([pressure_field]))])
            k = 0.5*k
            ma = k
            mi = -k

            colormap = ['Greys', 'seismic', 'coolwarm', 'twilight'][1]
            im_x = X.imshow(np.zeros_like(
                p_x[0]), vmin=mi, vmax=ma, aspect='equal', cmap=colormap)
            im_y = Y.imshow(np.zeros_like(
                p_y[0]), vmin=mi, vmax=ma, aspect='equal', cmap=colormap)
            im_z = Z.imshow(np.zeros_like(
                p_z[0]), vmin=mi, vmax=ma, aspect='equal', cmap=colormap)

            # Color Bar
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
            fig.colorbar(im_x, cax=cbar_ax)

            def init_func():
                X.set_title('YZ-Plane')
                Y.set_title('ZX-Plane')
                Z.set_title('XY-Plane')

            def update_plot(time_step):
                time = sim_param.delta_t * time_step
                text.set_text("Time: %.2f sec" % time)
                im_x.set_data(p_x[time_step])
                im_y.set_data(p_y[time_step])
                im_z.set_data(p_z[time_step])
                return [im_x, im_y, im_z]

            # keep the reference
            func_animation = FuncAnimation(
                fig,
                update_plot,
                frames=range(sim_param.number_of_samples),
                init_func=init_func,
                interval=interval,  # Delay between frames in milliseconds
                blit=False)
            if video_output:
                AnimationPlotter.write_video(func_animation, file_name)
            return [animation, func_animation]
        else:
            pass

    @staticmethod
    def plot_2D(pressure_field: np.ndarray, sim_param: SimulationParameters, interval: int = 0, video_output: bool = False, file_name: str = ''):
        '''
        Plots 3D domain in real-time with video output.

        Parameters
        ----------
        pressure_field : ndarray
            Pressure field data.
        sim_param : SimulationParameters
            Instance of simulation parameter class.
        interval : int
            Delay between frames in milliseconds.
        video_output: bool
            Displays the video on screen.
        file_name : str
            File name of video to write on disk.
        
        Returns
        -------
        FuncAnimation
            FuncAnimation instance.
        '''

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

        fig.suptitle("Time: %.2f sec" % 0)

        mi = np.min(-np.abs([pressure_field]))
        ma = np.max(np.abs([pressure_field]))

        im = ax.imshow(np.zeros_like(pressure_field[0]), vmin=mi, vmax=ma)

        # Color Bar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        def init_func():
            '''
            No implementation.
            '''
            pass

        def update_plot(time_step):
            time = sim_param.dt * time_step
            fig.suptitle("Time: %.2f sec" % time)
            im.set_data(pressure_field[time_step])
            return [im]

        # keep the reference
        animation = FuncAnimation(fig,
                                  update_plot,
                                  frames=sim_param.time_steps,
                                  init_func=init_func,
                                  interval=interval,  # Delay between frames in milliseconds
                                  blit=False)
        if video_output:
            AnimationPlotter.write_video(animation, file_name)
        return animation

    @staticmethod
    def plot_1D(p_field_t: np.ndarray, sim_param: SimulationParameters, interval: int = 0, video_output: bool = False, file_name: str = ''):
        '''
        Plots 3D domain in real-time with video output.

        Parameters
        ----------
        pressure_field : ndarray
            Pressure field data.
        sim_param : SimulationParameters
            Instance of simulation parameter class.
        interval : int
            Delay between frames in milliseconds.
        video_output: bool
            Displays the video on screen.
        file_name : str
            File name of video to write on disk.
        
        Returns
        -------
        FuncAnimation
            FuncAnimation instance.
        '''
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

        fig.suptitle("Time: %.2f sec" % 0)

        k = np.max([np.min(np.abs([p_field_t])), np.max(np.abs([p_field_t]))])
        k = 0.5*k
        ma = k
        mi = -k

        ln, = ax.plot(0, 0)

        def init_func():
            ax.set_ylim(mi, ma)
            ax.set_xlim(0, len(p_field_t[0]))
            ln.set_xdata(np.arange(len(p_field_t[0])))

        def update_plot(time_step):
            time = sim_param.delta_t * time_step
            fig.suptitle("Time: %.2f sec" % time)
            ln.set_ydata(p_field_t[time_step])
            return [ln]

        animation = FuncAnimation(fig,
                                  update_plot,
                                  frames=range(
                                      sim_param.number_of_samples),
                                  init_func=init_func,
                                  interval=interval,  # Delay between frames in milliseconds
                                  blit=False)
        if video_output:
            AnimationPlotter.write_video(animation, file_name)
        return animation

    @staticmethod
    def write_video(animation: FuncAnimation, file_name: str):
        '''
        Writes video to disk.

        Parameters
        ----------
        animation : FuncAnimation
            FuncAnimation instance, contains the animation.
        file_name : str
            Name of the file to be written on disk.
        '''

        writervideo = FFMpegWriter(fps=60)
        fileloc = "videos/"
        filename = file_name + '_' + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".mp4"
        animation.save(fileloc+filename,
                       dpi=300,
                       writer=writervideo)