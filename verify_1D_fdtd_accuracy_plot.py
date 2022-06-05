from matplotlib import ft2font
from common.impulse import ExperimentalUnit, Gaussian
from common.parameters import SimulationParameters as SIMP
from common.microphone import Microphone as Mic

from pytARD_1D.ard import ARDSimulator1D as ARDS
from pytARD_1D.partition import AirPartition1D as PARTD
from pytARD_1D.interface import InterfaceData1D

from matplotlib import pyplot as plt

'''
Companion script for verify_1D_fdtd_accuracy.py.
Creates a bar plot highlighting differences between each accuracy.
'''

plt.bar(8.97198486328125, label="Genauigkeit = 4")
plt.bar(8.896625137329101, label="Genauigkeit = 6")
plt.bar(8.693045711517334, label="Genauigkeit = 10")

plt.ylabel("Zeit [s]")
plt.xlabel("Iterationen")

plt.legend()
plt.show()