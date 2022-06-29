'''
DSP - Final Project
Project Name: Extraction of instantaneous frequencies using Synchrosqueezing in Python
Student: Wilson Cosmo
Date: 22/06/2022

Script description: This script plots ridge curves visualization of a loaded .wav signal. The ridge curves are extracted from CWT Transform and SSQ CWT.
Input Parameters: --sf 'path_to_file.wav' --p 'penalty for frequency change' --nr 'number of ridges'
'''

import numpy as np
import scipy.signal as sig
from ssqueezepy import ssq_cwt, ssq_stft, extract_ridges
from ssqueezepy.visuals import plot, imshow
from scipy.io import wavfile as wav
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sf", type=str, default='./test_signals/visualization/dolphins/SanctSound_CI01_03_dolphins_20190904T064203Z.wav', help="path to the signal file")
parser.add_argument("--p", type=int, default=300000, help="penalty for frequency change")
parser.add_argument("--nr", type=int, default=3, help="number of ridges")
opt = parser.parse_args()
print(opt)

# Visual method
def viz(x, Tf, ridge_idxs, yticks=None, tt = 'Signal', yl = 'Frequency Scale'):
    ylabel = (yl)
    title = (tt)
    ikw = dict(abs=1, cmap='turbo', title=title)
    pkw = dict(linestyle='--', color='k', xlabel="Time [samples]", ylabel=ylabel, xlims=(0, Tf.shape[1]))
    imshow(Tf, **ikw, show=0)
    plot(ridge_idxs, **pkw, show=1)

penalty = opt.p
nr = opt.nr #n of ridges

# load signal
(rate,sig) = wav.read(opt.sf)

x = sig
t = int(sig.size)
plot(x, title="Original Signal", show=1, xlabel="Time [samples]", ylabel="Signal Amplitude [A.U.]") #plot original signal

# CWT
Tx, Wx, ssq_freqs, scales = ssq_cwt(x)
ridge_idxs = extract_ridges(Wx, scales, penalty, n_ridges=nr, bw=25)
viz(x, Wx, ridge_idxs, scales, tt = 'CWT Ridges')

# SSQ CWT
ridge_idxs = extract_ridges(Tx, scales, penalty, n_ridges=nr, bw=4)
viz(x, Tx, ridge_idxs, ssq_freqs, tt = 'CWT with SSQ Ridges')
