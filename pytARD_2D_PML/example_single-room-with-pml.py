# -*- coding: utf-8 -*-
from ard_simulator import ARDSimulator
from simulation_parameters import SimulationParameters
from signals import GaussianFirstDerivative
# from pml_partition import PMLPartition, PMLType
from air_partition import AirPartition
# from interface import Interface, InfType

import numpy as np

# Procedure parameters
verbose = True
animation = True
write_to_file = True
video_output = False

sim_params = SimulationParameters(  wave_speed = 20, # in meter per second
                                    max_simulation_frequency = 30, # in herz
                                    samples_per_wave_length = 7, # samples per meter
                                    simulation_time = 3, # in seconds
                                    time_sampling_rate = 200, # in samples per second
                                    verbose = True, 
                                    visualize = True)

room_x = 8 
room_y = 5
pml_thickness = 5

# SOURCES
source_loc = (room_y/2, room_x/2)
signal = GaussianFirstDerivative(   sim_params, 
                                    signal_location = (room_y/2,room_x/2), 
                                    dominant_frequency = 25,
                                    time_offset = 1)

# AIR-Partitions
air_partition_1 = AirPartition((room_y, room_x), sim_params, signal)
air_partitions = [air_partition_1]

# # PML-Paritions
# pml_paritition1 = PMLPartition((room_y, pml_thickness),sim_params, pml_type = PMLType.RIGHT)
# pml_parititions = [pml_paritition1]

# # # INTERFACES
# interface1 = Interface(5,air_partition_1, pml_paritition1, InfType.VERTICAL, sim_params)
# interfaces = [interface1]

# sim = ARDSimulator(sim_params, air_partitions, interfaces, pml_parititions)
sim = ARDSimulator(sim_params, air_partitions,[],[])
sim.preprocessing()
sim.simulate()

if animation:
    
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), sharex=True, sharey=True, gridspec_kw = {'wspace':0, 'hspace':0})
    p = np.zeros_like(air_partitions[0].pressure_fields[0])
    
    fig.suptitle("Time: %.2f sec" % 0)

    
    # result = np.zeros()
    mi = np.min([air_partitions[0].pressure_fields])
    ma = np.max([air_partitions[0].pressure_fields])
    v = np.max(np.abs([mi,ma]))
    
    im_air = ax.imshow(np.zeros(air_partition_1.grid_shape),vmin=mi, vmax=ma)

    # attach color bar
    # fig.subplots_adjust(right=0.85)
    # cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.6])
    # fig.colorbar(ax, cax=cbar_ax)
    
    def init_func():
        im_air.set_data(np.zeros(air_partitions[0].grid_shape))

        
    def update_plot(time_step):
        # fig.subplots_adjust(hspace=0.1)
        # display current time step
        time = sim_params.dt * time_step       
        fig.suptitle("Time: %.2f sec" % time)
        
        # for i in range(len(im_air)):
        #     im_air[i].set_data(air_partitions[i].pressure_fields[time_step])
        #     # im_air[i].autoscale() # need
        im_air.set_data(air_partitions[0].pressure_fields[time_step])
        return [im_air]
    
    # keep the reference
    anim = FuncAnimation(   fig,
                            update_plot,
                            frames=sim_params.time_steps,
                            init_func=init_func,
                            interval=1, # Delay between frames in milliseconds
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
