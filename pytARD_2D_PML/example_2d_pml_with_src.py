# -*- coding: utf-8 -*-
from ard_simulator import ARDSimulator
from simulation_parameters import SimulationParameters
from signals import GaussianFirstDerivative
from pml_partition_stand_alone import PMLPartition
from air_partition import AirPartition
from interface import X_Interface

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
    from plotter import Plotter
    p_field_t = pml_parititions[0].pressure_fields
    anim = Plotter.plot2d(p_field_t, sim_params, frames = sim_params.time_steps, interval=0, video_output=False, file_name='')
