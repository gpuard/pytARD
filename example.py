from ard.ard import ARDSimulator as ARDS
from ard.parameters import ARDParamters as ARDP
import matplotlib.pyplot as plt


# param = ARDP([1], 1000, 1)
sim = ARDS([1], 1000, 1, verbose=True, visualize=True)

sim.preprocessing()
sim.simulation()


plt.figure()
for i in range(0, len(sim.pressure_field_results), 100):
    plt.clf()
    plt.plot(sim.pressure_field_results[i])
    plt.axis([0, sim.space_divisions, sim.pressure_field_results[i].min(), sim.pressure_field_results[i].max()])
    plt.pause(0.001)