# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 22:31:55 2022

@author: smailnik@students.zhaw.ch
"""

import numpy as np
import cupy as cp
from dct import dct_gpu
from dct import dct_cpu
import time

SAMPLES = [10**k for k in range(2, 8)]
NUMBER_RUNS = 15

def generate_samples(number_samples):
    return np.sin(range(number_samples))

samples_cases = []
gpu = []
cpu = []
# for s in SAMPLES:
for s in SAMPLES:
    samples_cases.append(s)
    x_cpu = generate_samples(s) # signal
    x_gpu = cp.asarray(x_cpu)
    
    gpu_times = []
    cpu_times = []
        
    times = []
    for t in range(NUMBER_RUNS):
        tick = time.time()
        gpu_res = cp.asnumpy(dct_gpu(x_gpu))
        tack = time.time()
        times.append(tack-tick)
    gpu.append(np.average(times))
    
    times = []
    for t in range(NUMBER_RUNS):
        tick = time.time()
        cpu_res = dct_cpu(x_cpu)
        tack = time.time()
        times.append(tack-tick)
    cpu.append(np.average(times))

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(samples_cases,gpu,label='gpu')
ax.plot(samples_cases,cpu,label='cpu')
ax.set_xlabel('number of samples')
ax.set_ylabel('run time')
ax.set_title(f'Average of {NUMBER_RUNS} runs')
ax.legend()
