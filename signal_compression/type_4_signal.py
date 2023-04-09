'''
Signal corresponds to upsampled random Gaussian noise.
It is then downsampled, which leads to a non-zero MSE
(lossy compression).
'''

from numpy.random import normal
import numpy as np
from lasse.dsp.scalar_quantization import UniformQuantizer
from compress_decompress import processing_pipeline
import scipy
import matplotlib.pyplot as plt
from compress_decompress import processing_pipeline_with_downsampling

show_plot = True  # to show plots

# define number of bits
num_bits_to_create_signal = 5
num_bits = 6  # Number of bits for the quantizer
# each double number is represented by 8 bytes, that is 8*8 = 64 bits
original_num_bits = 64  # original number of bits per sample in original file

signal_type = 4  # identifier for this signal

original_Fs = 800.0   # define initial sampling frequency (Hz)

# resample factors
upsampling_factor = 60  # to create original signal
downsampling_factor = 40  # adopted when compressing the signal
window_length = 31
Fs = original_Fs / downsampling_factor  # updated sampling frequency (after resample)

print("Maximum frequency (Hz) is approximately =", (original_Fs/2) / upsampling_factor)

# Gaussian parameters
average = 0  # Gaussian mean
standard_dev = 50  # Gaussian standard deviation
xmin = average - 3 * standard_dev
xmax = average + 3 * standard_dev

N = 500  # number of random samples before upsampling

# define the unquantized signal (it is not the original one, yet):
x_unquantized = normal(loc=average, scale=standard_dev, size=N)

# upsample signal to create x_original afterwards
down = 1
x_unquantized_upsampled = scipy.signal.resample_poly(x_unquantized, upsampling_factor, down, window=('kaiser', window_length), padtype='line')

# design quantizer
quantizer = UniformQuantizer(num_bits_to_create_signal, xmin, xmax, forceZeroLevel=False)

# quantize (obtain quantized values) to generate original signal
x_original, temp_i = quantizer.quantize_numpy_array(x_unquantized_upsampled)

# run pipeline that includes downsampling
processing_pipeline_with_downsampling(x_original, Fs, num_bits, signal_type, original_num_bits, original_Fs, show_plot=True)