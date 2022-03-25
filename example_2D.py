from common.parameters import SimulationParameters as SIMP

from pytARD_2D.ard import ARDSimulator as ARDS
from pytARD_2D.partition_data import PartitionData as PARTD
from pytARD_2D.pml_partition import PMLPartition as PMLP

import numpy as np

visualize = True
 
# Compilation of room parameters into parameter class
sim_params = SIMP(
    max_wave_frequency                  = 5, # maximal signal frequency in Hz 
    simulation_time                     = 10, # in seconds
    c                                   = 5, # speed of sound in meter/sec 
    samples_per_second                  = 40, # sampling rate in samples/sec -> how often are the grid points checked
    samples_per_wave_length             = 7, # number of samples per wavelength -> grid resolution
    enable_multicore                    = False, 
    verbose                             = True,
)

air_partition_1 = PARTD((20,20), sim_params)
pml_partition_1 = PMLP((20,5), sim_params) #first dimesion is "y"

air_partitions = [air_partition_1]
pml_partitions = [pml_partition_1]
# air_partitions = [partition_1]

# Instantiating and executing simulation
sim = ARDS(sim_params, air_partitions, pml_partitions)
sim.preprocessing()
sim.simulation()

if visualize:
    
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    p = np.zeros_like(air_partitions[0].pressure_field_results[0])

    mi = np.min(air_partitions[0].pressure_field_results)
    ma = np.max(air_partitions[0].pressure_field_results)
    v = np.max(np.abs([mi,ma]))
    
    v = np.max(np.abs([np.min(air_partition_1.pressure_field_results),np.max(air_partition_1.pressure_field_results)]))
    
    im_air = [ax[0,0].imshow(np.zeros(air_partition_1.grid_shape), interpolation='nearest', animated=False, vmin=-v, vmax=+v)]
    im_pml = [ax[0,1].imshow(np.zeros(pml_partition_1.grid_shape))]
    ax[1,0].imshow(p)
    ax[1,1].imshow(p)
    
    fig.tight_layout()
    # for i in range(len(im)):
    #     fig.colorbar(im[i])

    def init_func():
        for i in range(len(im_air)):
            im_air[i].set_data(np.zeros(air_partitions[i].grid_shape))
            
        for i in range(len(im_pml)):
            im_pml[i].set_data(np.zeros(pml_partitions[i].grid_shape))
            
        # for a in range(len(ax)):
        #     ax[0,i].grid(True)
        return [im_air,im_pml]
        
    def update_plot(time_step):
        # display current time step
        time = sim_params.delta_t * time_step       
        fig.suptitle("Time: %.2f sec" % time)
        
        for i in range(len(im_air)):
            im_air[i].set_data(air_partitions[i].pressure_field_results[time_step])
            # im_air[i].autoscale() # need
            
        # for i in range(len(im_pml)):
        #     im_pml[i].set_data(pml_partitions[i].pressure_field_results[time_step])
        #     im_pml[i].autoscale() # need
            
        return [im_air,im_pml]
    
    # keep the reference
    anim = FuncAnimation(   fig,
                            update_plot,
                            frames=np.arange(2, sim_params.number_of_time_samples,1),
                            init_func=init_func,
                            interval=1)       
        

