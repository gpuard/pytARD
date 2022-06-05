# -*- coding: utf-8 -*-

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
size = 15
plt.rc('font', size=size) #controls default text size
plt.rc('axes', titlesize=size) #fontsize of the title
plt.rc('axes', labelsize=size) #fontsize of the x and y labels
plt.rc('xtick', labelsize=size) #fontsize of the x tick labels
plt.rc('ytick', labelsize=size) #fontsize of the y tick labels
plt.rc('legend', fontsize=size) #fontsize of the legend

fig, ax = plt.subplots()
ax.grid()

ax.plot(np.array(samples_cases)/1e6,gpu,label='GPU')
ax.plot(np.array(samples_cases)/1e6,cpu,label='CPU')
# ax.set_xlim(0.0)
ax.set_xlabel(r'Anzahl Samples [$10^6$]')
ax.set_ylabel('Laufzeit [s]')
# ax.set_title(f'Average of {NUMBER_RUNS} runs')
ax.legend()
fig.tight_layout()
plt.savefig('cupy.png', dpi=300)