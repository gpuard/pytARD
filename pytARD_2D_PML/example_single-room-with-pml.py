# -*- coding: utf-8 -*-
from ard_simulator import ARDSimulator
from simulation_parameters import SimulationParameters
from signals import GaussianFirstDerivative
from pml_partition import PMLPartition, PMLType
from air_partition import AirPartition
from interface import X_Interface

import numpy as np
      #
#############
# AIR # PML # 
#############
      #
# Procedure parameters
verbose = True
animation = True
write_to_file = True
video_output = False
# plot2d = False
plot2d = True
sim_params = SimulationParameters(  wave_speed = 20, # in meter per second
                                    max_simulation_frequency = 30, # in herz
                                    samples_per_wave_length = 20, # samples per meter
                                    simulation_time = 1, # in seconds
                                    time_sampling_rate = 4000, # in samples per second
                                    verbose = True, 
                                    visualize = False)

room_x = 4
# room_y = 5
room_y = 10
pml_thickness = 5

# SOURCES
signal = GaussianFirstDerivative(   sim_params, 
                                    signal_location = (room_y*0.5, room_x*0.5), 
                                    dominant_frequency = 28,
                                    time_offset = 0)

# AIR-Partitions
air_partition_1 = AirPartition((room_y, room_x), sim_params, signal)
air_partitions = [air_partition_1]

# PML-Paritions
pml_paritition1 = PMLPartition((room_y, pml_thickness),sim_params,air_partition_1, pml_type = PMLType.RIGHT)
pml_parititions = [pml_paritition1]

# # INTERFACES
# INTERFACES
interface1 = X_Interface(air_partition_1,pml_paritition1, sim_params)
interfaces = [interface1]

sim = ARDSimulator(sim_params, air_partitions, interfaces, pml_parititions)
sim.preprocess()
sim.simulate()


if animation:
    from plotter import Plotter
    p_field_t = air_partitions[0].pressure_fields
    
    if plot2d:
        anim = Plotter.plot2d(p_field_t, sim_params, frames = sim_params.time_steps, interval=0, video_output=False, file_name='')
    else:
        from extractor import Extractor
        p_field_t = Extractor.extract_x(p_field_t,signal.grid_loc)
        
        anim = Plotter.plot1d(p_field_t, sim_params, frames = sim_params.time_steps, interval=0, video_output=False, file_name='')