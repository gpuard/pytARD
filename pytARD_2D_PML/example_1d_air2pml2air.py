# -*- coding: utf-8 -*-
from ard_simulator import ARDSimulator
from simulation_parameters import SimulationParameters
from signals import GaussianFirstDerivative
from pml_partition import PMLPartition_1D, PMLType
from air_partition import AirPartition
from interface import X_Interface_1D

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
CASE_AIR_PML_AIR = True
sim_params = SimulationParameters(  wave_speed = 20, # in meter per second
                                    max_simulation_frequency = 30, # in herz
                                    samples_per_wave_length = 20, # samples per meter
                                    simulation_time = 1, # in seconds
                                    time_sampling_rate = 4000, # in samples per second
                                    verbose = True, 
                                    visualize = False)
room_x = 5

pml_thickness = 5

# SOURCES
signal = GaussianFirstDerivative(   sim_params, 
                                    signal_location = (room_x/2,), 
                                    dominant_frequency = 20,
                                    time_offset = 0.1)

# AIR-Partitions
air_partition_1 = AirPartition((room_x,), sim_params, signal)
air_partition_2 = AirPartition((room_x,), sim_params, signal=None)
air_partitions = [air_partition_1,air_partition_2]

# PML-Paritions
pml_paritition1 = PMLPartition_1D((pml_thickness,),sim_params,air_partition_1, pml_type = PMLType.RIGHT)
pml_parititions = [pml_paritition1]

# INTERFACES
if CASE_AIR_PML_AIR:
    interface1 = X_Interface_1D(air_partition_1,pml_paritition1, sim_params)
    interface2 = X_Interface_1D(pml_paritition1,air_partition_2, sim_params)
else:
    # AIR-AIR-PML
    interface1 = X_Interface_1D(air_partition_1,air_partition_2, sim_params)
    interface2 = X_Interface_1D(air_partition_2,pml_paritition1, sim_params)

interfaces = [interface1, interface2]
simulation = ARDSimulator(sim_params, air_partitions, interfaces, pml_parititions)
# simulation = ARDSimulator(sim_params, air_partitions, interfaces, [])
simulation.preprocess()
simulation.simulate()

if animation:
    from plotter import Plotter
    p_field_t = list()
    if CASE_AIR_PML_AIR:
        [p_field_t.append(np.hstack([air_partitions[0].pressure_fields[i],pml_parititions[0].pressure_fields[i],air_partitions[1].pressure_fields[i]])) for i in sim_params.time_steps]
    else:
        [p_field_t.append(np.hstack([air_partitions[0].pressure_fields[i],air_partitions[1].pressure_fields[i],pml_parititions[0].pressure_fields[i]])) for i in sim_params.time_steps]
    anim = Plotter.plot1d(p_field_t, sim_params, frames = sim_params.time_steps, interval=0, video_output=False, file_name='')