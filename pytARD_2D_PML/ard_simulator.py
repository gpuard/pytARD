# -*- coding: utf-8 -*-
from interface import Interface, X_Interface

class ARDSimulator:
    '''
    ARD Simulation class. Creates and runs ARD simulator instance.
    '''

    def __init__(self, 
                 simulation_parameters,
                 air_partitions, interfaces, 
                 pml_parititions, mics=None):
        
        self.time_steps = simulation_parameters.time_steps
        self.air_partitions = air_partitions
        self.interfaces = interfaces
        self.pml_parititions = pml_parititions
        # self.mics = mics

    def preprocess(self):
        for p in self.air_partitions:
            p.preprocess()
            
        for p in self.interfaces:
            p.preprocess()
            
        # for p in self.pml_parititions:
        #     p.preprocess()

    def simulate(self):
        
        for time_step in self.time_steps[1:]:
            # print(time_step)
            # HANDLE INTERFACES
            for infs in self.interfaces:
                infs.simulate()
                
            # HANDLE AIR-PARTITIONs
            for p in self.air_partitions:
                p.simulate(time_step)
                
                
            # HANDLE PML-PARTITIONS
            for pml in self.pml_parititions:
                pml.simulate(time_step)
