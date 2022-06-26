'''
DSP - Final Project
Project Name: Extraction of instantaneous frequencies using Synchrosqueezing in Python
Student: Wilson Cosmo
Date: 22/06/2022

Script description: This script plots and extracts a single ridge curve of a loaded .wav signal. The ridge extracted is evaluated with the original frequencies of the loaded signal (ground truth) using the mean squared error metric. The ridge curve is extracted are from STFT Transform and SSQ STFT.
Input Parameters: --sf 'path_to_file.wav' --ff 'path_to_ground_truth_file.npy' --p 'penalty for frequency change'
'''

import numpy as np
import scipy.signal as sig
from ssqueezepy import ssq_cwt, ssq_stft, extract_ridges
from ssqueezepy.visuals import plot, imshow
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sf", type=str, default='./test_signals/ground_truth/crescent_decrescent/music.wav', help="signal file")
parser.add_argument("--ff", type=str, default='./test_signals/ground_truth/crescent_decrescent/music_f.npy', help="frequencies file - ground truth")
parser.add_argument("--p", type=int, default=10, help="penalty for frequency change")
opt = parser.parse_args()
print(opt)

# Visual method
def viz(x, Tf, ridge_idxs, yticks=None, tt = 'Signal', yl = 'Frequency Scale'):
    ylabel = (yl)
    title = (tt)
    ikw = dict(abs=1, cmap='turbo', title=title)
    pkw = dict(linestyle='--', color='r', xlabel="Time [samples]", ylabel=ylabel, xlims=(0, Tf.shape[1]))
    imshow(Tf, **ikw, show=0)
    plot(ridge_idxs, **pkw, show=1)

# Parameters for all signals
penalty = opt.p
nr = 1 #n of ridges
rf = 1 #rescaling factor

# load signal and ground truth
(rate,sig) = wav.read(opt.sf)
sig_f = np.load(opt.ff)

# Signal
x = sig
t = int(x.size)
plot(x, title="Original Signal", show=1, xlabel="Time [samples]", ylabel="Signal Amplitude [A.U.]") #plot original signal
Tx, Wx, ssq_freqs, scales = ssq_stft(x) # STFT

# STFT ridges
ridge_idxs = extract_ridges(Wx, scales, penalty, n_ridges=nr, bw=25, transform='stft')
t_ridges = np.transpose(ridge_idxs)
rf = np.amax(sig_f)/np.amax(t_ridges)
mse = mean_squared_error(sig_f,t_ridges[0]*rf) #mean_squared_error
print('\nSignal - STFT ridges \nNumber of frequencies = 1')
print('MSE for f1 = ' + str(mse))
viz(x, Wx, ridge_idxs, scales, tt = 'STFT Ridges')

# SSQ STFT ridges
ridge_idxs = extract_ridges(Tx, scales, penalty, n_ridges=nr, bw=4, transform='stft')
t_ridges = np.transpose(ridge_idxs)
rf = np.amax(sig_f)/np.amax(t_ridges)
mse = mean_squared_error(sig_f,t_ridges[0]*rf) #mean_squared_error
print('\nSignal - STFT ridges (with SSQ) \nNumber of frequencies = 1')
print('MSE for f1 = ' + str(mse))
viz(x, Tx, ridge_idxs, ssq_freqs, tt = 'STFT with SSQ Ridges')
