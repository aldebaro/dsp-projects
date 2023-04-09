'''
Signal corresponds to random noise with amplitudes
from a mixture of Gaussians (using three clusters).
It is similar to signal type 5 but here we add a
trend (add a ramp signal) to the original signal
and use ASCII (text) files, not binary ones.
'''

from compress_decompress import processing_pipeline
import numpy as np
import matplotlib.pyplot as plt

show_plot = True  # to show plots

# output file
output_filename = "original_6.txt"

# the file below must be generated first
filename = "original_5.double"
x_original = np.fromfile(filename, dtype=np.float64, count=-1)
Fs = 8000.0   # define sampling frequency (Hz)
original_num_bits = 64  # original number of bits per sample in original file

signal_type = 6  # identifier for this signal

num_unique_values = len(np.unique(x_original))
num_bits = np.ceil(np.log2(num_unique_values))  # Number of bits for the quantizer

print("x_original without trend =", x_original)
print("num_bits =", num_bits)

# add trend (a ramp signal)
N = len(x_original)
x_original_trended = np.zeros(x_original.shape)
for i in range(N):
    x_original_trended[i] = x_original[i] + (i+2)

print("x_original with trend =", x_original_trended)    

np.savetxt(output_filename, x_original_trended, fmt='%.18f')
print("Wrote file", output_filename)

# create de-trended signal, to be then compressed as binary file
x_uncompressed_detrended = np.loadtxt(output_filename)
print("x_uncompressed", x_uncompressed_detrended)
for i in range(N):
    x_uncompressed_detrended[i] -= i+2 # detrend signal

num_unique_values = len(np.unique(x_uncompressed_detrended))
num_bits = np.ceil(np.log2(num_unique_values))  # Number of bits for the quantizer
print("num_bits =", num_bits)

# execute all stages in pipeline
processing_pipeline(x_uncompressed_detrended, Fs, num_bits, signal_type, original_num_bits, show_plot=show_plot)

#assuming the MSE in stages before are 0, we can compare with the original
# add trend (a ramp signal)
N = len(x_original)
x_reconstructed_trended = np.zeros(x_original.shape)
for i in range(N):
    x_reconstructed_trended[i] = x_uncompressed_detrended[i] + (i+2)

print("x_uncompressed_detrended=", x_uncompressed_detrended)
print("x_reconstructed_trended=", x_reconstructed_trended)
error_signal = x_original_trended - x_reconstructed_trended
MSE = np.mean(error_signal*error_signal)
print("Final MSE comparing signal with trend and its reconstructed version =", MSE)

if show_plot:
    plt.plot(x_original_trended)
    plt.title("Original (with trend) signal")
    plt.show()
