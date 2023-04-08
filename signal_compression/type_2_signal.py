'''
Signal corresponds to random Gaussian noise.
'''

from numpy.random import normal
import numpy as np
from lasse.dsp.scalar_quantization import UniformQuantizer
from compress_decompress import processing_pipeline

show_plot = True  # to show plots

# define number of bits
# each double number is represented by 8 bytes, that is 8*8 = 64 bits
num_bits = 5  # Number of bits for the quantizer
original_num_bits = 64  # original number of bits per sample in original file

signal_type = 1  # identifier for this signal

Fs = 800   # define sampling frequency (Hz)
Ts = 1.0/Fs  # sampling interval (s)
duration = 4  # seconds
average = 5400  # Gaussian mean
standard_dev = 20  # Gaussian standard deviation
xmin = average - 3 * standard_dev
xmax = average + 3 * standard_dev

N = int(duration / Ts)  # number of random samples

# define the unquantized signal (it is not the original one, yet):
x_unquantized = normal(loc=average, scale=standard_dev, size=N)

# design quantizer
quantizer = UniformQuantizer(num_bits, xmin, xmax, forceZeroLevel=False)

# quantize (obtain quantized values) to generate original signal
x_original, temp_i = quantizer.quantize_numpy_array(x_unquantized)

# execute all stages in pipeline
processing_pipeline(x_original, Fs, num_bits, signal_type, original_num_bits, show_plot=show_plot)