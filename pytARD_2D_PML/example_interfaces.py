# -*- coding: utf-8 -*-
from ard_simulator import ARDSimulator
from simulation_parameters import SimulationParameters
from signals import GaussianFirstDerivative
# from pml_partition import PMLPartition, PMLType
from air_partition import AirPartition
from interface import X_Interface

import numpy as np
      #
#############
# AIR # AIR #
#############
      #
# Procedure parameters
verbose = True
animation = True
write_to_file = True
video_output = False
plot2d = True

sim_params = SimulationParameters(  wave_speed = 20, # in meter per second
                                    max_simulation_frequency = 30, # in herz
                                    samples_per_wave_length = 20, # samples per meter
                                    simulation_time = 1, # in seconds
                                    time_sampling_rate = 4000, # in samples per second
                                    verbose = True, 
                                    visualize = False)

room_x = 5 
room_y = 5
pml_thickness = 5

# SOURCES
signal = GaussianFirstDerivative(   sim_params, 
                                    signal_location = (room_y/2,room_x/2), 
                                    dominant_frequency = 20,
                                    time_offset = 0.1)

# AIR-Partitions
air_partition_1 = AirPartition((room_y, room_x), sim_params, signal=None)
air_partition_2 = AirPartition((room_y, room_x), sim_params, signal)
air_partitions = [air_partition_1,air_partition_2]

# PML-Paritions
# pml_paritition1 = PMLPartition((room_y, pml_thickness),sim_params, pml_type = PMLType.RIGHT)
# pml_parititions = [pml_paritition1]

# INTERFACES
interface1 = X_Interface(air_partition_1, air_partition_2, sim_params)
interfaces = [interface1]

# sim = ARDSimulator(sim_params, air_partitions, interfaces, pml_parititions)
sim = ARDSimulator(sim_params, air_partitions, interfaces,[])
sim.preprocess()
sim.simulate()

if animation:
    from plotter import Plotter
    p_field_t = list()
    [p_field_t.append(np.hstack([air_partitions[0].pressure_fields[i],air_partitions[1].pressure_fields[i]])) for i in sim_params.time_steps]
    # [p_field_t.append(np.hstack([np.zeros_like(air_partitions[1].pressure_fields[i]),air_partitions[1].pressure_fields[i]])) for i in sim_params.time_steps]
    if plot2d:
        anim = Plotter.plot2d(p_field_t, sim_params, frames = sim_params.time_steps, interval=0, video_output=False, file_name='')
    else:
        from extractor import Extractor
        p_field_t = Extractor.extract_x(p_field_t,signal.grid_loc)
        anim = Plotter.plot1d(p_field_t, sim_params, frames = sim_params.time_steps, interval=0, video_output=False, file_name='')