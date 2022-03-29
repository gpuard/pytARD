from common.parameters import SimulationParameters as SIMP

from pytARD_2D.ard import ARDSimulator as ARDS
from pytARD_2D.partition_data import PartitionData as PARTD
from pytARD_2D.pml_partition import PMLPartition as PMLP

import numpy as np
visualize = True
 
# Compilation of room parameters into parameter class
sim_params = SIMP(
    max_wave_frequency                  = 5, # maximal signal frequency in Hz 
    simulation_time                     = 20, # in seconds
    c                                   = 1, # speed of sound in meter/sec 
    samples_per_second                  = 40, # sampling rate in samples/sec -> how often are the grid points checked
    samples_per_wave_length             = 2, # number of samples per wavelength -> grid resolution
    enable_multicore                    = False, 
    verbose                             = True,
)


air_partition_1 = PARTD((15,15), sim_params)
air_partition_2 = PARTD((15,15), sim_params)
air_partition_3 = PARTD((15,15), sim_params, do_impulse=False)
air_partitions = [air_partition_1,air_partition_2,air_partition_3]


pml_partition_1 = PMLP((15,15), sim_params) #first dimesion is "y"
pml_partitions = [pml_partition_1]


# Instantiating and executing simulation
sim = ARDS(sim_params, air_partitions, pml_partitions)
sim.preprocessing()
sim.simulation()
# ######
# room_dim_x = 20
# room_dim_y = 20
# room = np.array((room_dim_x,room_dim_y))
# results = list()
# for i in range(sim_params.number_of_time_samples):
#     results.append(room)
# #####



if visualize:
    
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10), sharex=True, sharey=True, gridspec_kw = {'wspace':0, 'hspace':0})
    p = np.zeros_like(air_partitions[0].pressure_field_results[0])
    
    fig.suptitle("Time: %.2f sec" % 0)

    for i, a in enumerate(fig.axes):
    # for  a in ax:
        # a.grid('on', linestyle='--')
        a.set_xticklabels([])
        a.set_yticklabels([])

    #     a.set_aspect('equal')
    
    # result = np.zeros()
    mi = np.min([air_partitions[0].pressure_field_results,air_partitions[1].pressure_field_results,air_partitions[2].pressure_field_results])
    ma = np.max([air_partitions[0].pressure_field_results,air_partitions[1].pressure_field_results,air_partitions[2].pressure_field_results])
    v = np.max(np.abs([mi,ma]))
    
    v = np.max(np.abs([np.min(air_partition_1.pressure_field_results),np.max(air_partition_1.pressure_field_results)]))
    
    # im_air = [ax[0,0].imshow(np.zeros(air_partition_1.grid_shape), interpolation='nearest', animated=False, vmin=-v, vmax=+v)]
    # im_air = [ax[0,0].imshow(np.zeros(air_partition_1.grid_shape), interpolation='nearest', animated=False, cmap='jet', vmin=-v, vmax=+v),
    #           ax[1,0].imshow(np.zeros(air_partition_1.grid_shape), interpolation='nearest', animated=False, cmap='jet', vmin=-v, vmax=+v),]
    
    im_air = [ax[0,0].imshow(np.zeros(air_partition_1.grid_shape),vmin=mi, vmax=ma),
              ax[1,0].imshow(np.zeros(air_partition_2.grid_shape),vmin=mi, vmax=ma),
              ax[1,1].imshow(np.zeros(air_partition_3.grid_shape),vmin=mi, vmax=ma)]

    im_pml = [ax[0,1].imshow(np.zeros(pml_partition_1.grid_shape),vmin=mi, vmax=ma)]
        
    # add space for colour bar
    # fig.subplots_adjust(right=0.85)
    # cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    # fig.colorbar(im_air[0], cax=cbar_ax)
    
    def init_func():
        for i in range(len(im_air)):
            im_air[i].set_data(np.zeros(air_partitions[i].grid_shape))
            
        for i in range(len(im_pml)):
            im_pml[i].set_data(np.zeros(pml_partitions[i].grid_shape))
            
        # for a in range(len(ax)):
        #     ax[0,i].grid(True)
        return [im_air,im_pml]
        
    def update_plot(time_step):
        fig.subplots_adjust(hspace=0.1)
        # display current time step
        time = sim_params.delta_t * time_step       
        fig.suptitle("Time: %.2f sec" % time)
        
        for i in range(len(im_air)):
            im_air[i].set_data(air_partitions[i].pressure_field_results[time_step])
            # im_air[i].autoscale() # need
            
        for i in range(len(im_pml)):
            im_pml[i].set_data(pml_partitions[i].pressure_field_results[time_step])
            # im_pml[i].autoscale() # need
        
        return [im_air,im_pml]
    
    # keep the reference
    anim = FuncAnimation(   fig,
                            update_plot,
                            frames=np.arange(0, sim_params.number_of_time_samples,1),
                            init_func=init_func,
                            interval=1)       
        

