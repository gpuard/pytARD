# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Plotter:
    
    @staticmethod
    def plot2d(p_field_t, simulation_parameters, frames, interval=0, video_output=False, file_name=''):      

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
                                frames=frames,
                                init_func=init_func,
                                interval=interval, # Delay between frames in milliseconds
                                blit=False)
        if video_output:
            Plotter.write_video(anim, file_name)
        return anim
       
    def plot1d(p_field_t, simulation_parameters, frames, interval=0, video_output=False, file_name=''):
        
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
            Plotter.write_video(anim, file_name)
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
