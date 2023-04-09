'''
Signal 1 is composed by the sum of two sinusoids.
'''

import numpy as np
from lasse.dsp.scalar_quantization import UniformQuantizer
from compress_decompress import processing_pipeline

show_plot = True  # to show plots

# define number of bits
# each double number is represented by 8 bytes, that is 8*8 = 64 bits
num_bits = 4  # Number of bits for the quantizer
original_num_bits = 64  # original number of bits per sample in original file

signal_type = 1  # identifier for this signal

# sinusoid frequencies in Hz
f1 = 2300
f2 = 5000
phase1 = np.pi/4.0
phase2 = np.pi/3.0
Fs = 6.0*f2   # define sampling frequency (Hz)
Ts = 1.0/Fs  # sampling interval (s)
duration = 4  # seconds

t = np.arange(0, duration, Ts)  # discrete-time
# define the unquantized signal (it is not the original one, yet):
x_unquantized = 5.0*np.sin(2*np.pi*f1*t + phase1) + 3.0*np.cos(2*np.pi*f2*t + phase2)  

# dynamic range
xmin = np.min(x_unquantized)
xmax = np.max(x_unquantized)

# design quantizer
quantizer = UniformQuantizer(num_bits, xmin, xmax, forceZeroLevel=False)

# quantize (obtain quantized values) to generate original signal
x_original, temp_i = quantizer.quantize_numpy_array(x_unquantized)

# execute all stages in pipeline
processing_pipeline(x_original, Fs, num_bits, signal_type, original_num_bits, show_plot=show_plot)