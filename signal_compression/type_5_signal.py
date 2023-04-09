'''
Signal corresponds to random noise with amplitudes
from a mixture of Gaussians (using three clusters).
'''

from numpy.random import normal
from lasse.dsp.scalar_quantization import UniformQuantizer
from compress_decompress import processing_pipeline
import numpy as np

show_plot = True  # to show plots

#quantizer parameters
b = 3  # Number of bits when quantizing individual clusters
original_num_bits = 64  # original number of bits per sample in original file

signal_type = 5  # identifier for this signal

N = 5000 # number of random samples for each centroid value
Fs = 8000.0   # define sampling frequency (Hz)

# define number of bits
# each double number is represented by 8 bytes, that is 8*8 = 64 bits
#center of different clusters
cluster_centroids = np.array([-100, 30, 400])


standard_dev = 10 # Gaussian standard deviation

num_clusters = len(cluster_centroids)
x_q = np.zeros ( (num_clusters, N) , dtype = np.float32) # pre-allocate
for c in range( num_clusters ): # go over centroids
    # define dynamic range around the cluster centroid 
    average = cluster_centroids[c]
    xmin = average - 3 * standard_dev
    xmax = average + 3 * standard_dev

    # quantize
    quantizer = UniformQuantizer(b, xmin, xmax, forceZeroLevel=False)
    temp = normal(loc=average, scale=standard_dev, size=(1, N))
    temp_q, temp_i = quantizer.quantize_numpy_array(temp)
    
    # store quantized values for this centroid
    x_q[c] = temp_q

# shuffle the samples
x_q = x_q.ravel()
my_generator = np.random.default_rng()
x_q = my_generator.permutation(x_q)

x_original = x_q

num_unique_values = len(np.unique(x_original))
num_bits = np.ceil(np.log2(num_unique_values))  # Number of bits for the quantizer

print("x_original =", x_original)
print("num_bits =", num_bits)

# execute all stages in pipeline
processing_pipeline(x_original, Fs, num_bits, signal_type, original_num_bits, show_plot=show_plot)