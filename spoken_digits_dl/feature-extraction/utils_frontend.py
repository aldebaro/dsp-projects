'''
This code provides functions for the feature extraction module.
All feature extractors output a matrix organized as frequency x time
(number of rows is the number of frequency points, with the row 0 (top)
representing the highest frequency).
'''
import numpy as np
import pandas as pd
import os
#from numba import jit
import librosa
import librosa.display
import sys
import h5py
import json
from numpy.fft        import fft
from scipy.io.wavfile import read, write    
from skimage.transform import resize
import matplotlib.pyplot as plt
import torch
import reassignment.reassignment_linear as reassign_lin
#from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler

#Part 1) File manipulation
'''
Read X and y from HDF5 file.
'''
def read_instances_from_file(intputHDF5FileName):
    h5pyFile = h5py.File(intputHDF5FileName, 'r')
    Xtemp = h5pyFile["X"]
    ytemp = h5pyFile["y"]
    X = np.array(Xtemp[()])
    y = np.array(ytemp[()])
    h5pyFile.close()
    return X, y

'''
Write X and y to HDF5 file.
'''
def write_instances_to_file(outputHDF5FileName, X, y):
    h5pyFile = h5py.File(outputHDF5FileName, 'w')
    h5pyFile["X"] = X #store in hdf5
    h5pyFile["y"] = y #store in hdf5
    h5pyFile.close()
    #print('==> Wrote file ', outputHDF5FileName, ' with keys X and y')

#Part 2) Time-frequency feature extraction algorithms

'''
Use Magnasco algorithm with reassign_lin.high_resolution_spectrogram.
'''
def magnasco_spectrogram(audio):

    # defining constants and parameters
    q = 2
    tdeci = 96
    over = 2
    noct = 108
    #noct = 200 #to increase dimension in frequency axis
    minf = 1.5e-2
    #minf = 1e-3
    maxf = 0.16
    #maxf = 0.5

    spectrogram = reassign_lin.high_resolution_spectrogram(audio, q, tdeci, over, noct, minf, maxf, device=torch.device('cpu'))
    # port the spectrogram to cpu
    # No need to do it if device = cpu
    spectrogram = spectrogram.cpu().numpy().T
    return spectrogram

'''
Assume Fs = 44100 Hz.
The STFT is calculated with dimension num_freq_bins x num_time_frames
but the code outputs a tranposed array with dimenstion
num_time_frames x num_freq_bins, which are called T and D in the code, respectively.
'''
def stft_spectrogram(audio, max_freq=15000):
    # STFT
    n_fft=2048 #FFT size
    win_length = n_fft
    hop_length = 100
    stft = librosa.stft(audio.astype(float),n_fft=n_fft, hop_length=hop_length,
    win_length=win_length, window='hann', center=True, dtype=None, pad_mode='constant')
    stft = np.abs(stft)**2.0
    #there is no energy above 15 kHz, so find the index
    samplerate = 44100 #this was previously checked
    df = samplerate/n_fft
    kmax = int(max_freq/df)
    stft = stft[:kmax,:]
    #make sure the y-axis starts with low frequencies and goes up to higher frequencies
    stft = np.flipud(stft)
    return stft

'''
Mel-scaled spectrogram.
https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
n_mels is the number of filter outputs, which is the number of rows in the output matrix
'''    
def melspectrogram(audio, max_freq=15000, n_mels=128):
    n_fft=2048 #FFT size    
    sr = 44100 #sampling frequency in Hz
    win_length = n_fft
    hop_length = 100
    melspectrogram = librosa.feature.melspectrogram(y=audio.astype(float),sr=sr,
    n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann',
    center=True, dtype=None, pad_mode='constant',n_mels=n_mels, fmax=max_freq)
    #make sure the y-axis starts with low frequencies and goes up to higher frequencies
    melspectrogram = np.flipud(melspectrogram)
    return melspectrogram

#Part 3) Normalization of features to feed neural networks

def normalize_as_maggie(spectrogram):
        # Original author: Jiayi (Maggie) Zhang <jiayizha@andrew.cmu.edu>
        min_value = np.min(spectrogram)
        if min_value >= 0:
            #check because it may be already in log domain
            spectrogram = np.log(spectrogram+1e-8) 
        spectrogram /= np.max(np.abs(spectrogram))
        spectrogram += 1
        return spectrogram

def normalize_to_min_max(spectrogram, min=-1, max=1):
        #numbers are now from something like -80 or -60 dB to 0 dB.
        # Move them to defined range. Default is -1 to 1
        # Using sklearn MinMaxScaler, note it works for each column, so
        # reshape to make it a single column array
        min_max_scaler = MinMaxScaler(feature_range=(min,max))    
        # Stack everything into a single column to scale by the global min / max
        original_shape = spectrogram.shape
        tmp = spectrogram.reshape(-1,1)
        spectrogram = min_max_scaler.fit_transform(tmp).reshape(original_shape)
        #print('bbbb', np.max(spectrogram), np.min(spectrogram))
        return spectrogram

def normalize_standardize_along_frequency(spectrogram):
        num_freq_bins, num_time_frames = spectrogram.shape
        for i in range(num_time_frames):
            dft = spectrogram[:,i]
            mu  = np.mean(dft)
            std = np.std(dft)
            if std > 0:
                spectrogram[:,i] = (dft - mu) / std
            else:
                spectrogram[:,i] = (dft - mu)
        return spectrogram

#Part 4) Signal statistics and plots

'''
Plot feature matrix.
'''
def plot_feature(features, title):
    plt.axis('off')
    #plt.yticks(y_range, y_axis[::-1])
    #plt.xticks(x_range, x_axis)
    plt.imshow(features, cmap='inferno')
    plt.title(title)
    plt.colorbar()
    plt.show()

'''
Plot feature matrix.
'''
def plot_feature_no_show(features, title):
    plt.clf()
    plt.axis('off')
    #plt.yticks(y_range, y_axis[::-1])
    #plt.xticks(x_range, x_axis)
    plt.imshow(features, cmap='inferno')
    title = title + ", shape=", str(features.shape)
    plt.title(title)
    plt.colorbar()

def get_stats(X):
    min_x = np.min(X)
    max_x = np.max(X)
    mean_x = np.mean(X)
    return min_x, max_x, mean_x

'''
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
'''
def plot_overall_histogram(X):
    #https://stackoverflow.com/questions/13730468/from-nd-to-1d-arrays
    all_values = X.ravel()
    plt.hist(all_values, bins=100, log=True)
    plt.title('Histogram of feature values from all examples')
    plt.xlabel('feature value')
    plt.ylabel('number of occurrences')
    plt.show()

def values_above_threshold_per_frequency(X):
        show_plots = True
        num_bins = 100
        threshold = np.min(X.ravel()) #can use a number such as 0.5

        #recall X has dimension TIME x FREQ because transpose had been used
        (num_examples, T, D) = X.shape
        if show_plots:
            plot_overall_histogram(X)
        min_x, max_x, mean_x = get_stats(X.ravel())
        #print(min_x, max_x, mean_x)
        range_all_numbers = [min_x, max_x] #define range
        occurrences_above_reference = np.zeros((D,),dtype=int)
        for i in range(D): #go over all frequencies
            values_given_frequency = X[:,:,i].ravel() #for all examples, and all time instants
            #min_x, max_x, mean_x = get_stats(values_given_frequency)
            #print(min_x, max_x, mean_x)
            n, bins = np.histogram(values_given_frequency, bins=num_bins, range=range_all_numbers)
            #find the bin corresponding to the threshold value
            indices = np.array(np.where(bins < threshold)).ravel()
            if np.any(indices):
                last_index = int(indices[-1]) #did not check this logic
            else:
                last_index = 1 #threshold is the minimum value, and accounts only for first bin
            occurrences_above_reference[i] = np.sum(n[last_index:])
        if show_plots:
            plt.plot(occurrences_above_reference)
            plt.yscale('log')
            plt.grid()
            plt.title('Histogram of values above reference = ' + str(threshold))
            plt.xlabel('frequency index (from 0 to D-1)')
            plt.show()
        return occurrences_above_reference

