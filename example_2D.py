from common.parameters import SimulationParameters as SIMP

from pytARD_2D.ard import ARDSimulator as ARDS
from pytARD_2D.partition_data import PartitionData as PARTD

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation
visualize = True
 
# Compilation of room parameters into parameter class
sim_params = SIMP(
    max_wave_frequency                  = 50, # in Hz 
    simulation_time                     = 7, # in seconds
    c                                   = 10, # speed of sound in meter/sec 
    Fs                                  = 500, # sampling rate in samples/sec
    spatial_samples_per_wave_length     = 3, 
    enable_multicore                    = False, 
    verbose                             = True,
)

partition_1 = PARTD((30,30), sim_params)
partition_2 = PARTD((30,30), sim_params,do_impulse=False)

part_data = [partition_1, partition_2]
# part_data = [partition_1]

# Instantiating and executing simulation
sim = ARDS(sim_params, part_data)
sim.preprocessing()
sim.simulation()

if visualize:
    fig, ax = plt.subplots(nrows=2, ncols=2)
    p = np.zeros_like(part_data[0].pressure_field_results[0])

    # mi = np.absolute(np.min(part_data[0].pressure_field_results))
    # ma = np.absolute(np.max(part_data[0].pressure_field_results))
    # v = np.max([mi,ma])
    # v = int(v)
    # im = ax[0,0].imshow(p, interpolation='nearest', animated=True, vmin=-v, vmax=+v, cmap=plt.cm.RdBu)
    
    im = [ax[0,0].imshow(p),ax[0,1].imshow(p)]
    
    # fig.tight_layout()
    # for i in range(len(im)):
    #     fig.colorbar(im[i])

    def init_func():
        for i in range(len(im)):
            im[i].set_data(np.zeros(part_data[0].pressure_field_results[0].shape))
        # for a in range(len(ax)):
        #     ax[0,i].grid(True)
        return im,
        
    def update_plot(t):
        for i in range(len(im)):
            im[i].set_data(part_data[i].pressure_field_results[t])
            im[i].autoscale() # need
        # display current time step
        t = sim_params.delta_t*i       
        fig.suptitle("Time: %.2f sec" % t)
        return im,
    
    # keep the reference
    anim = FuncAnimation(   fig,
                            update_plot,
                            frames=np.arange(2, sim_params.number_of_time_samples,50),
                            init_func=init_func,
                            interval=1)       
        

