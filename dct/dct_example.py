import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct, idct
import random
import math


size = 40
i = np.linspace(0, 5*2*np.pi, size)
#x = np.sin(i)
x = np.zeros(size)
#x += np.array([random.randint(1, 256) for i in range(size)]) 
#x += np.sin(.5*i) * 255
x[5] = 255
x[6] = 255
x[7] = 255


X = dct(x)

print(x)
print(X)
print(x.shape)
print(X.shape)

pos = 0
l = 20

# Extract a block from image
plt.figure()
plt.imshow(x[pos:pos+l].reshape((l, 1)),cmap='inferno')
plt.title(f"An {l}x Sample block")

# Display the dct of that block
plt.figure()
plt.imshow(X[pos:pos+l].reshape((l, 1)),cmap='inferno', vmax=np.max(X), vmin=np.min(X))
plt.title(f"An {l}x DCT block")

#xs = idct(X)
#plt.plot(i, x, label='x', linewidth=8)
#plt.plot(i, xs, label='idct(x)', linewidth=3)
#plt.legend()
plt.show()

def alpha(u, N):
    if u == 0:
        return np.sqrt(1 / N)
    else:
        return np.sqrt(2 / N)

p = 0
wave_sum = np.zeros(len(x))
for a in X:
    wave = np.zeros(len(X))
    for w_i in range(len(wave)):
        #if w_i == 0:
        #    wave[w_i] = (a / np.sqrt(2)) * np.cos((np.pi*(2*w_i + 1) * p) / (2 * len(X)))
        #else:
        #    wave[w_i] = a * np.cos((np.pi*(2*w_i + 1) * p) / (2 * len(X)))
        wave[w_i] = alpha(p, len(X)) * a * np.cos((p * np.pi *(2*w_i + 1.)) / (2. * (len(X))))

    wave_sum += wave
    plt.plot(i, wave)
    p += 1


plt.show()

plt.plot(i, wave_sum, label='reconstructed from cosines')
plt.plot(i, x, label='original')
plt.legend()
plt.show()
