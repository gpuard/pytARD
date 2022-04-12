from pytARD_2D.ard import ARDSimulator as ARDS
from pytARD_2D.partition_data import PartitionData as PARTD
from pytARD_2D.interface import InterfaceData2D, Direction

from common.parameters import SimulationParameters as SIMP
from common.impulse import Gaussian, Unit, WaveFile
from common.microphone import Microphone as Mic

import numpy as np

# Room parameters
simulation_time = 3  #  seconds
time_sampling_rate = 1500 # samples / sec
spatial_samples_per_wave_length = 5

# Procedure parameters
verbose = True
visualize = True
write_to_file = True
video_output = False

# Compilation of room parameters into parameter class
sim_params = SIMP(
    max_simulation_frequency=100, # Hz
    T=simulation_time,
    c= 20, # meter / sec
    Fs=time_sampling_rate,
    spatial_samples_per_wave_length=spatial_samples_per_wave_length,
    verbose=verbose,
    visualize=visualize
)

room_x = 20 
room_y = 20

# SOURCES
impulse_location = np.array([[int(room_y / 2)], [int(room_x / 2)]])
source = Gaussian(sim_params, impulse_location, 10000)
# impulse = Unit(sim_params, impulse_location, 1)
# impulse = WaveFile(sim_params, impulse_location, 'clap_8000.wav', 100)

# AIR-Partitions
air_partition_1 = PARTD(np.array([[room_y], [room_x]]), sim_params, source)
air_partitions = [air_partition_1]

# # INTERFACES
# interface1 = InterfaceData2D(0, 1, Direction.Horizontal)
# interfaces = [interface1]

# # Microphones (are optional)
# mic1 = Mic(
#     0, # Parition number
#     # Position
#     [int(part_data[0].dimensions[0] / 2), 
#     int(part_data[0].dimensions[1] / 2)], 
#     sim_params, 
#     "left" # Name of resulting wave file
# )


# Compilation of all microphones into one mics object. Add or remove mics here. TODO change to obj.append()
# mics = [mic1]

# Instantiation serializer for reading and writing simulation state data
# serializer = Serializer(compressed=True)

# SIMULATION
# sim = ARDS(sim_params, part_data, interfaces, mics)
# sim = ARDS(sim_params, part_data, interfaces)
sim = ARDS(sim_params, air_partitions,[])
sim.preprocessing()
sim.simulation()


if visualize:
    
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), sharex=True, sharey=True, gridspec_kw = {'wspace':0, 'hspace':0})
    p = np.zeros_like(air_partitions[0].pressure_field_results[0])
    
    fig.suptitle("Time: %.2f sec" % 0)

    
    # result = np.zeros()
    mi = np.min([air_partitions[0].pressure_field_results])
    ma = np.max([air_partitions[0].pressure_field_results])
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
        time = sim_params.delta_t * time_step       
        fig.suptitle("Time: %.2f sec" % time)
        
        # for i in range(len(im_air)):
        #     im_air[i].set_data(air_partitions[i].pressure_field_results[time_step])
        #     # im_air[i].autoscale() # need
        im_air.set_data(air_partitions[0].pressure_field_results[time_step])
        return [im_air]
    
    # keep the reference
    anim = FuncAnimation(   fig,
                            update_plot,
                            frames=range(sim_params.number_of_time_samples),
                            init_func=init_func,
                            interval=100,
                            blit=False)       
    if video_output:
        
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        from datetime import datetime

        writervideo = FFMpegWriter(fps=60)
        fileloc = "videos/"
        filename  = "2d-ard_pml_demo_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".mp4"
        anim.save(fileloc+filename,
                  dpi=300,
                  # fps=60,
                  writer=writervideo) 
