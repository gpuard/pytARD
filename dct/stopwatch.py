# -*- coding: utf-8 -*-

import numpy as np
import cupy as cp
from dct import dct_gpu
from dct import dct_cpu
import time
from scipy.fft import dct as scipy_dct
from cupy.fft.config import get_plan_cache
import cupy
mempool = cupy.get_default_memory_pool()

# SAMPLES = np.linspace(1, 10**7, num=25, dtype=np.dtype(np.int32), endpoint=True)
# SAMPLES += 1
# SAMPLES = np.arange(1,10**7,10**5)
SAMPLES = [10**k for k in range(2, 8)]
# NUMBER_RUNS = 15
NUMBER_RUNS = 15

def generate_samples(number_samples):
    # return np.sin(range(number_samples))
    return np.random.rand(number_samples) # signal

samples_cases = []
gpu = []
cpu = []
s_dct = []
# for s in SAMPLES:
for s in SAMPLES:
    print(s)
    samples_cases.append(s)
    x_cpu = generate_samples(s) # signal
    x_gpu = cp.asarray(x_cpu)
    
    gpu_times = []
    cpu_times = []
        
    times = []
    for t in range(NUMBER_RUNS):
        tick = time.time()
        # gpu_res = cp.asnumpy(dct_gpu(x_gpu))
        dct_gpu(x_gpu)
        tack = time.time()
        times.append(tack-tick)
        
        fft_cache = get_plan_cache()
        fft_cache.set_size(0)

        mempool.free_all_blocks()
        
        # cupy reenable fft caching
        fft_cache.set_size(16)
        fft_cache.set_memsize(-1)
    gpu.append(np.average(times))
    
    times = []
    for t in range(NUMBER_RUNS):
        tick = time.time()
        # cpu_res = dct_cpu(x_cpu)
        dct_cpu(x_cpu)
        tack = time.time()
        times.append(tack-tick)
    cpu.append(np.average(times))

    times = []
    for t in range(NUMBER_RUNS):
        tick = time.time()
        # scipy_dct_res = scipy_dct(x_cpu)
        scipy_dct(x_cpu)
        tack = time.time()
        times.append(tack-tick)
    s_dct.append(np.average(times))


# gpu[-1]/s_dct[-1]
# Out[28]: 0.4483270253752866

# cpu[-1]/s_dct[-1]
# Out[29]: 4.015763097304218

# gpu[-1]/cpu[-1]
# Out[30]: 0.11164180119993845

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
# ax.set_xscale('log')

# ax.plot(np.array(samples_cases),gpu,':')
# ax.scatter(np.array(samples_cases),gpu,marker='x',label='GPU FCT')
# ax.plot(np.array(samples_cases),cpu,':')
# ax.scatter(np.array(samples_cases),cpu,marker='x',label='CPU FCT')
# ax.plot(np.array(samples_cases),s_dct,':')
# ax.scatter(np.array(samples_cases),s_dct,marker='x',label='SciPy DCT')
ax.plot(np.array(samples_cases)/1e6,gpu,':')
ax.scatter(np.array(samples_cases)/1e6,gpu,marker='x',label='GPU FCT')
ax.plot(np.array(samples_cases)/1e6,cpu,':')
ax.scatter(np.array(samples_cases)/1e6,cpu,marker='x',label='CPU FCT')
ax.plot(np.array(samples_cases)/1e6,s_dct,':')
ax.scatter(np.array(samples_cases)/1e6,s_dct,marker='x',label='SciPy DCT')

ax.set_xlabel(r'Anzahl Samples [$10^6$]')
ax.set_ylabel('Laufzeit [s]')
# ax.set_title(f'Average of {NUMBER_RUNS} runs')
ax.legend()
fig.tight_layout()
plt.savefig('cupy_25.png', dpi=300)