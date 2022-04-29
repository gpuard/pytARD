import numpy as np
from wavdiff import wav_diff, visualize_multiple_waveforms 

wav_diff("roomA_mic1.wav", "roomB_mic1.wav", "mic1_diff.wav")
wav_diff("roomA_mic2.wav", "roomB_mic2.wav", "mic2_diff.wav")
visualize_multiple_waveforms(["roomA_mic1.wav", "roomB_mic1.wav", "mic1_diff.wav", "roomA_mic2.wav", "roomB_mic2.wav", "mic2_diff.wav"], dB=False)
