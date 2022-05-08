from matplotlib import ft2font
from common.impulse import ExperimentalUnit, Gaussian
from common.parameters import SimulationParameters as SIMP
from common.microphone import Microphone as Mic

from pytARD_1D.ard import ARDSimulator as ARDS
from pytARD_1D.partition import PartitionData as PARTD
from pytARD_1D.interface import InterfaceData1D

import numpy as np
import time
from matplotlib import pyplot as plt

# Room parameters
src_pos = [0] # m
duration = 2 # seconds
Fs = 8000 # sample rate
upper_frequency_limit = 300 # Hz
c = 342 # m/s
spatial_samples_per_wave_length = 6

# Procedure parameters
auralize = False
verbose = False
visualize = True
filename = "verify_1D_interface_reflection"

# Compilation of room parameters into parameter class
sim_param = SIMP(
    upper_frequency_limit, 
    duration, 
    c=c, 
    Fs=Fs,
    spatial_samples_per_wave_length=spatial_samples_per_wave_length, 
    verbose=verbose,
    visualize=False
)

impulse = Gaussian(sim_param, [0], 10000)

result_time_4 = []
result_time_6 = []
result_time_10 = []

for accuracy in [4, 6, 10]:
    for i in range(10):
        # Define test  rooms
        test_partition_1 = PARTD(np.array([c]), sim_param, impulse)
        test_partition_2 = PARTD(np.array([c]), sim_param)
        test_room = [test_partition_1, test_partition_2]

        # Define Interfaces
        interfaces = []
        interfaces.append(InterfaceData1D(0, 1, fdtd_acc=accuracy))

        # Define and position mics

        # Initialize & position mics. 
        mic = Mic(0, int(c / 2), sim_param, filename + "_" +"mic")
        test_mics = [mic]

        # Instantiating and executing simulation
        test_sim = ARDS(sim_param, test_room, 1, interface_data=interfaces, mics=test_mics)
        test_sim.preprocessing()
        start = time.time()
        test_sim.simulation()
        end = time.time()
        if accuracy == 4:
            result_time_4.append(end - start)
        elif accuracy == 6:
            result_time_6.append(end - start)
        elif accuracy == 10:
            result_time_10.append(end-start)

print(f"Average time of accuracy = 4: {np.sum(result_time_4) / len(result_time_4)}")
print(result_time_4)    
print(f"Average time of accuracy = 6: {np.sum(result_time_6) / len(result_time_6)}")
print(result_time_6)    
print(f"Average time of accuracy = 10: {np.sum(result_time_10) / len(result_time_10)}")
print(result_time_10)    

