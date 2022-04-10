from scipy.io.wavfile import read, write
import numpy as np
import matplotlib.pyplot as plt

fsl, left = read('control_after.wav')
fsr, right = read('interface_after.wav')

left = np.array(left, dtype=np.float)
right = np.array(right, dtype=np.float)

diff = []

for i in range(3, len(left) - 1):
    diff.append(left[i] - right[i - 3])

diff = np.array(diff)

write('diff.wav', fsl, diff.astype(np.float))

plt.plot(diff)
plt.show()
