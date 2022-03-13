from scipy.io.wavfile import read, write
import numpy as np
import matplotlib.pyplot as plt

fsl, left = read('left0.wav')
fsr, right = read('right0.wav')

left = np.array(left, dtype=np.float)
right = np.array(right, dtype=np.float)
diff = left - right

write('diff.wav', fsl, diff.astype(np.float))

plt.plot(diff)
plt.show()
