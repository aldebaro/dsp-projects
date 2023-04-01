import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from numpy.random import normal
from lasse.dsp.scalar_quantization import UniformQuantizer

#center of different clusters
cluster_centroids = np.array([-100, 30, 400])

#quantizer parameters
delta = 0.5  # quantization step
b = 3  # Number of bits

standard_dev = 1 # Gaussian standard deviation
N = 1000 # number of random samples for each centroid value

num_clusters = len(cluster_centroids)
x_q = np.zeros ( (num_clusters, N) , dtype = np.float32) # pre-allocate
for c in range( num_clusters ): # go over centroids
    # define dynamic range around the cluster centroid 
    average = cluster_centroids[c]
    xmin = average - 3 
    xmax = average + 3

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
print(x_q)

# check unique values (that will indicate minimum number of bits)
print(np.unique(x_q))

# save to binary file with little-endian samples
filename = "test_float_little_endian.bin"
x_q.tofile(filename)

# read from file and check if consistent
#x_q2 = np.fromfile(filename, dtype='>f', count=-1)
x_q2 = np.fromfile(filename, dtype=np.float32, count=-1)
print(x_q2)

plt.plot(x_q - x_q2)
#plt.hist(x_q)
plt.show()
