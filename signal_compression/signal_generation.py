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

standard_dev = 10 # Gaussian standard deviation
N = 1000 # number of random samples for each centroid value

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
print(x_q)

# check unique values (that will indicate minimum number of bits)
unique_values = np.unique(x_q)
num_unique = len(unique_values)
print("# of unique values", num_unique)
print("unique_values", unique_values)

amplitude_encoding = {} #dictionary()
for i in range(num_unique):
    amplitude_encoding[unique_values[i]]=i
print(amplitude_encoding)

#encode all samples
x_encoded = np.zeros ( x_q.shape, dtype = np.int)
for i in range(num_clusters*N):
    x_encoded[i] = amplitude_encoding[x_q[i]]

print(x_encoded)

# save to binary file with little-endian samples
filename = "test_float_little_endian.bin"
#x_q.tofile(filename)

#assumption: languages support writing bytes (8 bits)
#we need to combine some samples in a multiple of a byte
#mmc(5,8) = 40
#8 samples of five bits and pack into 40 bits, and then save as 5 bytes.
#"we need to pack and unpack"

# read from file and check if consistent
#x_q2 = np.fromfile(filename, dtype='>f', count=-1)
x_q2 = np.fromfile(filename, dtype=np.float32, count=-1)
print(x_q2)

#plt.plot(x_q - x_q2)
#plt.plot(x_q)
plt.hist(x_encoded, bins=100)
plt.show()
