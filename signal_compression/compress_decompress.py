'''
The pipeline is:

1) Generate the compressed signal via the arrays:
x_unquantized ==> x_original ==> x_encoded ==> compressed_bitstream

2) Save x_original to file called original_file_name
and
save compressed_bitstream to file called compressed_file_name

3) Read array compressed_bitstream from compressed_bitstream file
and decompress it creating x_encoded_from_file

4) Generate the uncompressed signal using arrays:
x_encoded_from_file ==> x_reconstructed
'''

import numpy as np
from lasse.dsp.scalar_quantization import UniformQuantizer
from lasse.dsp.spectral_analysis import power_spectral_density
from lasse.dsp.spectral_analysis import spectrum_magnitude
from lasse.io.compressed_files import write_encoded_file
from lasse.io.compressed_files import read_encoded_file
import os
import matplotlib.pyplot as plt
import scipy


def processing_pipeline(x_original, Fs, num_bits, signal_type, original_num_bits, show_plot=True):
    '''
    Pipeline without downsampling.
    '''
    # define file names
    original_file_name = 'original_' + str(signal_type) + '.double'
    compressed_file_name = 'compressed_' + str(signal_type) + '.' + str(num_bits) + 'bits'

    Ts = 1.0 / Fs  # sampling interval
    N = len(x_original)  # number of samples
    duration = N*Ts

    # check unique values (that will help indicating minimum number of bits)
    unique_values = np.unique(x_original)
    num_unique = len(unique_values)

    # design an encoder (mapper) from each unique value to an integer as a dictionary, 
    # creating a continuous range of integers to help compressing the information
    # and the decoder as a simple array (to be faster)
    amplitude_encoder = {}  # dictionary in Python
    amplitude_decoder = np.zeros((num_unique,), dtype=np.float64)
    for i in range(num_unique):
        amplitude_encoder[unique_values[i]] = i  # encoder entry
        amplitude_decoder[i] = unique_values[i]  # decoder entry

    # encode (map) all samples real samples to integers
    x_encoded = np.zeros(x_original.shape, dtype=int)
    for i in range(N):
        x_encoded[i] = amplitude_encoder[x_original[i]]

    # save files and get their sizes in number of bytes
    x_original.tofile(original_file_name)  # do not use compression and save as double
    compressed_bitstream = write_encoded_file(x_encoded, num_bits, compressed_file_name)
    size_original_file = os.path.getsize(original_file_name)
    size_encoded_file = os.path.getsize(compressed_file_name)

    # read from file and check if consistent
    # x_encoded2 = np.fromfile(encoded_file_name, dtype='>f', count=-1)
    x_encoded_from_file = read_encoded_file(compressed_file_name, num_bits)

    # decode signal obtained from file
    x_reconstructed = np.zeros(x_encoded_from_file.shape, dtype=np.float64)
    for i in range(N):
        x_reconstructed[i] = amplitude_decoder[x_encoded_from_file[i]]

    # calculate mean squared-error
    error_signal = x_original - x_reconstructed
    mse = np.mean(error_signal*error_signal)

    # print useful information
    print("Signal duration in seconds", duration)
    print("Sampling frequency", Fs)
    print("# of signal samples", N)
    print("# of unique values", num_unique)
    print("unique_values", unique_values)
    print('size_original_file=', size_original_file)
    print('size_encoded_file=', size_encoded_file)
    print('actual compression ratio (original/encoded)=', size_original_file / size_encoded_file)
    print('theoretical compression ratio=', original_num_bits / num_bits)
    print('dictionary amplitude_encoder=', amplitude_encoder)
    print('x_encoded=', x_encoded)
    print('compressed_bitstream=', compressed_bitstream)
    print('x_encoded_from_file=', x_encoded_from_file)
    print('Length of x_encoded=', len(x_encoded))
    print('Length of compressed_bitstream=', len(compressed_bitstream))
    print('Length of x_encoded_from_file=', len(x_encoded_from_file))
    print('# of unique values in x_encoded=', len(np.unique(x_encoded)))
    print('# of unique values in compressed_bitstream=', len(np.unique(compressed_bitstream)))
    print('# of unique values in x_encoded_from_file=', len(np.unique(x_encoded_from_file)))
    print('MSE =', mse)

    if show_plot:
        plt.figure(1)
        plt.plot(x_reconstructed)
        plt.title('Reconstructed signal')
        #plt.show() is going to be called below
        plt.figure(2)
        plt.plot(error_signal)
        plt.title('Error (original-reconstructed) signal')
        #plt.show() is going to be called below
        plt.figure(3)
        plt.hist(x_reconstructed, 100)
        plt.title('Histogram of reconstructed signal')
        #plt.show() is going to be called below
        plt.figure(4)
        if False:
            power_spectral_density(x_reconstructed, Fs, show_plot=show_plot)
        else:
            spectrum_magnitude(x_reconstructed, Fs, show_plot=show_plot, remove_mean=True)

def processing_pipeline_with_downsampling(x_original, Fs, num_bits, signal_type, original_num_bits, original_Fs, show_plot=True):
    '''
    Pipeline with downsampling: reducing the sampling frequency we need less
    bits to represent the information.
    '''
    downsampling_factor = original_Fs / Fs
    if not (downsampling_factor == int(downsampling_factor)):
        raise Exception("Downsampling factor must be an integer! I found: " + str(downsampling_factor))

    # define file names
    original_file_name = 'original_' + str(signal_type) + '.double'
    compressed_file_name = 'compressed_' + str(signal_type) + 'down' + str(downsampling_factor) + '.' + str(num_bits) + 'bits'

    original_Ts = 1.0 / original_Fs  # sampling interval
    N_original = len(x_original)  # number of samples
    duration = N_original*original_Ts

    # save files and get their sizes in number of bytes
    x_original.tofile(original_file_name)  # do not use compression and save as double

    # downsample
    window_length = 31
    up = 1  # upsampling factor
    x_original_downsampled = scipy.signal.resample_poly(x_original, up, downsampling_factor, window=('kaiser', window_length), padtype='line')
    #print(np.unique(x_original_downsampled))
    #print('len unique =', len(np.unique(x_original_downsampled)))

    # design new quantizer
    xmin = np.min(x_original_downsampled)
    xmax = np.max(x_original_downsampled)
    quantizer = UniformQuantizer(num_bits, xmin, xmax, forceZeroLevel=False)
    x_quantized_downsampled, temp_i = quantizer.quantize_numpy_array(x_original_downsampled)

    # check unique values (that will help indicating minimum number of bits)
    unique_values = np.unique(x_quantized_downsampled)
    num_unique = len(unique_values)

    # design an encoder (mapper) from each unique value to an integer as a dictionary, 
    # creating a continuous range of integers to help compressing the information
    # and the decoder as a simple array (to be faster)
    amplitude_encoder = {}  # dictionary in Python
    amplitude_decoder = np.zeros((num_unique,), dtype=np.float64)
    for i in range(num_unique):
        amplitude_encoder[unique_values[i]] = i  # encoder entry
        amplitude_decoder[i] = unique_values[i]  # decoder entry

    # encode (map) all samples real samples to integers
    x_encoded = np.zeros(x_quantized_downsampled.shape, dtype=int)
    N_downsampled = len(x_quantized_downsampled)
    for i in range(N_downsampled):
        x_encoded[i] = amplitude_encoder[x_quantized_downsampled[i]]

    compressed_bitstream = write_encoded_file(x_encoded, num_bits, compressed_file_name)
    size_original_file = os.path.getsize(original_file_name)
    size_encoded_file = os.path.getsize(compressed_file_name)

    # read from file and check if consistent
    # x_encoded2 = np.fromfile(encoded_file_name, dtype='>f', count=-1)
    x_encoded_from_file = read_encoded_file(compressed_file_name, num_bits)

    # decode signal obtained from file
    x_reconstructed_downsampled = np.zeros(x_encoded_from_file.shape, dtype=np.float64)
    for i in range(N_downsampled):
        x_reconstructed_downsampled[i] = amplitude_decoder[x_encoded_from_file[i]]

    # upsample
    up = downsampling_factor
    x_reconstructed = scipy.signal.resample_poly(x_reconstructed_downsampled, up, 1, window=('kaiser', window_length), padtype='line')

    # calculate mean squared-error
    error_signal = x_original - x_reconstructed
    mse = np.mean(error_signal*error_signal)

    signal_power = np.mean(x_original*x_original)
    sqnr = 10.0 * np.log10(signal_power / mse)

    # print useful information
    print("Signal duration in seconds", duration)
    print("Sampling frequency", Fs)
    print("# of original signal samples", N_original)
    print("# of final signal samples", N_downsampled)
    print("# of unique values", num_unique)
    print("unique_values", unique_values)
    print('size_original_file=', size_original_file)
    print('size_encoded_file=', size_encoded_file)
    print('actual compression ratio (original/encoded)=', size_original_file / size_encoded_file)
    print('theoretical compression ratio=', downsampling_factor * original_num_bits / num_bits)
    print('dictionary amplitude_encoder=', amplitude_encoder)
    print('x_encoded=', x_encoded)
    print('compressed_bitstream=', compressed_bitstream)
    print('x_encoded_from_file=', x_encoded_from_file)
    print('Length of x_encoded=', len(x_encoded))
    print('Length of compressed_bitstream=', len(compressed_bitstream))
    print('Length of x_encoded_from_file=', len(x_encoded_from_file))
    print('# of unique values in x_encoded=', len(np.unique(x_encoded)))
    print('# of unique values in compressed_bitstream=', len(np.unique(compressed_bitstream)))
    print('# of unique values in x_encoded_from_file=', len(np.unique(x_encoded_from_file)))
    print('MSE =', mse)
    print('Signal to quantization noise ratio (SQNR) =', sqnr, 'dB')

    if show_plot:
        plt.figure(1)
        plt.plot(x_reconstructed)
        plt.title('Reconstructed signal')
        #plt.show() is going to be called below
        plt.figure(2)
        plt.plot(error_signal)
        plt.title('Error (original-reconstructed) signal')
        #plt.show() is going to be called below
        plt.figure(3)
        plt.hist(x_reconstructed, 100)
        plt.title('Histogram of reconstructed signal')
        #plt.show() is going to be called below
        plt.figure(4)
        if False:
            power_spectral_density(x_reconstructed, Fs, show_plot=show_plot)
        else:
            spectrum_magnitude(x_reconstructed, Fs, show_plot=show_plot, remove_mean=True)            