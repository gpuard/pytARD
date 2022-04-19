# -*- coding: utf-8 -*-
from ard_simulator import ARDSimulator
from simulation_parameters import SimulationParameters
from signals import GaussianFirstDerivative
from pml_partition import PMLPartition, PMLType
from air_partition import AirPartition
from interface import X_Interface

import numpy as np
      #     #
###################
# AIR # PML # AIR #
###################
      #     #
# Procedure parameters
verbose = True
animation = True
write_to_file = True
video_output = False

sim_params = SimulationParameters(  wave_speed = 20, # in meter per second
                                    max_simulation_frequency = 30, # in herz
                                    samples_per_wave_length = 20, # samples per meter
                                    simulation_time = 0.09, # in seconds
                                    time_sampling_rate = 4000, # in samples per second
                                    verbose = True, 
                                    visualize = False)

room_x = 4
room_y = 5
pml_thickness = 5

# SOURCES
signal = GaussianFirstDerivative(   sim_params, 
                                    signal_location = (room_y/2, room_x*0.9), 
                                    dominant_frequency = 28,
                                    time_offset = 0)

# AIR-Partitions
air_partition_1 = AirPartition((room_y, room_x), sim_params, signal)
air_partition_2 = AirPartition((room_y, room_x), sim_params, signal=None)
air_partitions = [air_partition_1,air_partition_2]

# PML-Paritions
pml_paritition1 = PMLPartition((room_y, pml_thickness),sim_params,air_partition_1, pml_type = PMLType.RIGHT)
pml_parititions = [pml_paritition1]

# # INTERFACES
# INTERFACES
interface1 = X_Interface(air_partition_1,pml_paritition1, sim_params)
interface2 = X_Interface(pml_paritition1,air_partition_2, sim_params)
interfaces = [interface1, interface2]

sim = ARDSimulator(sim_params, air_partitions, interfaces, pml_parititions)
sim.preprocess()
sim.simulate()


if animation:
    
    p_field_t = list()
    [p_field_t.append(np.hstack([air_partitions[0].pressure_fields[i],air_partitions[1].pressure_fields[i]])) for i in sim_params.time_steps]
    # [p_field_t.append(np.hstack([np.zeros_like(air_partitions[1].pressure_fields[i]),air_partitions[1].pressure_fields[i]])) for i in sim_params.time_steps]
    

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), sharex=True, sharey=True, gridspec_kw = {'wspace':0, 'hspace':0})
    p = np.zeros_like(p_field_t[0])
    
    fig.suptitle("Time: %.2f sec" % 0)

    mi = np.min(-np.abs([p_field_t]))
    ma = np.max(np.abs([p_field_t]))
    

    im = ax.imshow(np.zeros_like(p_field_t[0]),vmin=mi, vmax=ma)
    # attach color bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    def init_func():
        pass
        
    def update_plot(time_step):
        time = sim_params.dt * time_step       
        fig.suptitle("Time: %.2f sec" % time)
        im.set_data(p_field_t[time_step])
        return [im]
    
    # keep the reference
    anim = FuncAnimation(   fig,
                            update_plot,
                            frames=sim_params.time_steps,
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
