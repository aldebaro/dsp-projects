'''
Signal corresponds to upsampled random Gaussian noise, such 
that it does not occupy the whole frequency band. MSE is zero.
'''

from numpy.random import normal
import numpy as np
from lasse.dsp.scalar_quantization import UniformQuantizer
from compress_decompress import processing_pipeline
import scipy

show_plot = True  # to show plots

# define number of bits
# each double number is represented by 8 bytes, that is 8*8 = 64 bits
num_bits = 6  # Number of bits for the quantizer
original_num_bits = 64  # original number of bits per sample in original file

signal_type = 3  # identifier for this signal

initial_Fs = 800   # define initial sampling frequency (Hz)

#resample factors
up = 4
down = 1
window_length = 31
Fs = (up/down)*initial_Fs  # updated sampling frequency (after resample)

Ts = 1.0/Fs  # sampling interval (s)
duration = 4  # seconds
average = 0  # Gaussian mean
standard_dev = 50  # Gaussian standard deviation
xmin = average - 3 * standard_dev
xmax = average + 3 * standard_dev

N = int(duration / Ts)  # number of random samples

# define the unquantized signal (it is not the original one, yet):
x_unquantized = normal(loc=average, scale=standard_dev, size=N)

x_unquantized = scipy.signal.resample_poly(x_unquantized, up, down, window=('kaiser', window_length), padtype='line')

# design quantizer
quantizer = UniformQuantizer(num_bits, xmin, xmax, forceZeroLevel=False)

# quantize (obtain quantized values) to generate original signal
x_original, temp_i = quantizer.quantize_numpy_array(x_unquantized)

# execute all stages in pipeline
processing_pipeline(x_original, Fs, num_bits, signal_type, original_num_bits, show_plot=show_plot)