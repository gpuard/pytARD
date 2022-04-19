# -*- coding: utf-8 -*-
from ard_simulator import ARDSimulator
from simulation_parameters import SimulationParameters
from signals import GaussianFirstDerivative
from pml_partition_stand_alone import PMLPartition
from air_partition import AirPartition
from interface import X_Interface

import numpy as np

#######
# PML # 
#######
# Procedure parameters
verbose = True
animation = True
write_to_file = True
video_output = False

sim_params = SimulationParameters(  wave_speed = 20, # in meter per second
                                    max_simulation_frequency = 30, # in herz
                                    samples_per_wave_length = 20, # samples per meter
                                    simulation_time = 1, # in seconds
                                    time_sampling_rate = 4000, # in samples per second
                                    verbose = True, 
                                    visualize = False)

sim_params.dt = 0.5
sim_params.num_time_samples = 250
sim_params.time_steps = range(sim_params.num_time_samples)
# SOURCES
signal = GaussianFirstDerivative(   sim_params, 
                                    signal_location = (10, 10), 
                                    dominant_frequency = 28,
                                    time_offset = 0)

# PML-Paritions
pml_paritition1 = PMLPartition((5, 5),sim_params,signal)
pml_parititions = [pml_paritition1]


sim = ARDSimulator(sim_params, [], [], pml_parititions)
sim.preprocess()
sim.simulate()


if animation:
    
    p_field_t = pml_parititions[0].pressure_fields
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    
    fig.suptitle("Time-step: %d" % 0)

    # mi = np.min(-np.abs([p_field_t]))
    # ma = np.max(np.abs([p_field_t]))
    mi = np.min([p_field_t])
    ma = np.max([p_field_t])
    

    # im = ax.imshow(np.zeros_like(p_field_t[0]),vmin=mi, vmax=ma,cmap='jet')
    im = ax.imshow(np.zeros_like(p_field_t[0]),vmin=mi, vmax=ma)
    # im = ax.imshow(np.zeros_like(p_field_t[0]))
    # # attach color bar
    # fig.subplots_adjust(right=0.85)
    # cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    # fig.colorbar(im, cax=cbar_ax)
    
    def init_func():
        im.set_data(np.zeros_like(p_field_t[0]))
        
    def update_plot(time_step):
        time = time_step       
        fig.suptitle("Time-step: %d" % time)
        im.set_data(p_field_t[time_step])
        return [im]
    
    # keep the reference
    anim = FuncAnimation(   fig,
                            update_plot,
                            frames=len(p_field_t),
                            init_func=init_func,
                            interval=0, # Delay between frames in milliseconds
                            blit=False)       
    if video_output:
        
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        from datetime import datetime

        writervideo = FFMpegWriter(fps=60)
        fileloc = "videos/"
        filename  = "give_your_name_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".mp4"
        anim.save(fileloc+filename,
                  dpi=300,
                  # fps=60,
                  writer=writervideo) 
