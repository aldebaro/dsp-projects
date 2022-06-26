'''
DSP - Final Project
Project Name: Extraction of instantaneous frequencies using Synchrosqueezing in Python
Student: Wilson Cosmo
Date: 20/06/2022

Script description: This script generates three cossenoid based signals, and it's respectives ground truth frequencies. Ridge curves are extracted from each signal then evaluated with the mean squared error metric. The ridge curves are extracted are from STFT Transform and SSQ STFT.
Input Parameters: --p 'penalty for frequency change'
'''

import numpy as np
import scipy.signal as sig
from ssqueezepy import ssq_cwt, ssq_stft, extract_ridges
from ssqueezepy.visuals import plot, imshow
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--p", type=int, default=100, help="Penalty for frequency change")
opt = parser.parse_args()
print(opt)

# Visual method ##########################################################################################
def viz(x, Tf, ridge_idxs, yticks=None, tt = 'Signal', yl = 'Frequency Scale'):
    ylabel = (yl)
    title = (tt)
    ikw = dict(abs=1, cmap='turbo', title=title)
    pkw = dict(linestyle='--', color='k', xlabel="Time [samples]", ylabel=ylabel, xlims=(0, Tf.shape[1]))
    imshow(Tf, **ikw, show=0)
    plot(ridge_idxs, **pkw, show=1)

# Parameters for all signals ##########################################################################################
penalty = opt.p #penalty for frequency change
N = 500 #n samples
t = np.linspace(0, 10, N, endpoint=True)
rf = 1/10 #rescaling factor

# Signal I - sum of cossenoids ##########################################################################################
nr = 2 #n of ridges
f1 = 10
f2 = 15
x1 = np.cos(2*np.pi*f1*t) + np.cos(2*np.pi*f2*t)

# Ground truth for signal I
gt_x1 = np.zeros(N)
gt_x2 = np.zeros(N)
for i in range(N):
    gt_x1[i] = f1
    gt_x2[i] = f2

plot(x1, title="Generated signal I - sum of cossenoids", show=1, xlabel="Time [samples]", ylabel="Signal Amplitude [A.U.]") #plot original signal
Tx, Wx, ssq_freqs, scales = ssq_stft(x1, t=t) #STFT

#STFT ridges
ridge_idxs = extract_ridges(Wx, scales, penalty, n_ridges=nr, bw=25, transform='stft')
t_ridges = np.transpose(ridge_idxs)*rf
mse12 = mean_squared_error(gt_x1,t_ridges[0]) #mean_squared_error
mse11 = mean_squared_error(gt_x2,t_ridges[1]) #mean_squared_error
print('\nSignal I - STFT ridges \nNumber of frequencies = 2')
print('MSE for f1 = ' + str(mse11))
print('MSE for f2 = ' + str(mse12))
viz(x1, Wx, ridge_idxs, scales, tt = 'Signal I - STFT Ridges')

#SSQ STFT ridges
ridge_idxs = extract_ridges(Tx, scales, penalty, n_ridges=nr, bw=4, transform='stft')
t_ridges = np.transpose(ridge_idxs)*rf
mse12 = mean_squared_error(gt_x2,t_ridges[0]) #mean_squared_error
mse11 = mean_squared_error(gt_x1,t_ridges[1]) #mean_squared_error
print('\nSignal I - STFT ridges (with SSQ) \nNumber of frequencies = 2')
print('MSE for f1 = ' + str(mse11))
print('MSE for f2 = ' + str(mse12))
viz(x1, Tx, ridge_idxs, ssq_freqs, tt = 'Signal I - STFT with SSQ Ridges')

# Signal II - cossenoid with linear variant frequency ##########################################################################################
nr = 1 #n of ridges
f1 = 8
f2 = 18
x2 = sig.chirp(t, f0=f1,  f1=f2, t1=20, method='linear')

# Ground truth for signal II
gt_x3 = np.zeros(N)
for i in range(N):
    gt_x3[i] = f1 + ((f2 - f1) * t[i] / 20)

plot(x2, title="Generated signal II - cossenoid with linear variant frequency", show=1, xlabel="Time [samples]", ylabel="Signal Amplitude [A.U.]") #plot original signal
Tx, Wx, ssq_freqs, scales = ssq_stft(x2, t=t) #STFT

#STFT ridges
ridge_idxs = extract_ridges(Wx, scales, penalty, n_ridges=nr, bw=25, transform='stft')
t_ridges = np.transpose(ridge_idxs)*rf
mse22 = mean_squared_error(gt_x3,t_ridges[0]) #mean_squared_error
print('\nSignal II - STFT ridges \nNumber of frequencies = 1')
print('MSE for f1 = ' + str(mse22))
viz(x2, Wx, ridge_idxs, scales, tt = 'Signal II - STFT Ridges')

#SSQ STFT ridges
ridge_idxs = extract_ridges(Tx, scales, penalty, n_ridges=nr, bw=4, transform='stft')
t_ridges = np.transpose(ridge_idxs)*rf
mse22 = mean_squared_error(gt_x3,t_ridges[0]) #mean_squared_error
print('\nSignal II - STFT ridges (with SSQ) \nNumber of frequencies = 1')
print('MSE for f1 = ' + str(mse22))
viz(x2, Tx, ridge_idxs, ssq_freqs, tt = 'Signal II - STFT with SSQ Ridges')

# Signal III - signal I + signal II ##########################################################################################
nr = 3 #n of ridges
x3 = x1 + x2
# The ground truth os signal III correspond to the ground truth of the signals I and II
plot(x3, title="Generated signal III - signal I + signal II", show=1, xlabel="Time [samples]", ylabel="Signal Amplitude [A.U.]") #plot original signal
Tx, Wx, ssq_freqs, scales = ssq_stft(x3, t=t) #STFT

#STFT ridges
ridge_idxs = extract_ridges(Wx, scales, penalty, n_ridges=nr, bw=25, transform='stft')
t_ridges = np.transpose(ridge_idxs)*rf
mse31 = mean_squared_error(gt_x1,t_ridges[0]) #mean_squared_error
mse32 = mean_squared_error(gt_x2,t_ridges[1]) #mean_squared_error
mse33 = mean_squared_error(gt_x3,t_ridges[2]) #mean_squared_error
print('\nSignal III - STFT ridges \nNumber of frequencies = 3')
print('MSE for f1 = ' + str(mse31))
print('MSE for f2 = ' + str(mse32))
print('MSE for f3 = ' + str(mse33))
viz(x3, Wx, ridge_idxs, scales, tt = 'Signal III - STFT Ridges')

#SSQ STFT ridges
ridge_idxs = extract_ridges(Tx, scales, penalty, n_ridges=nr, bw=4, transform='stft')
t_ridges = np.transpose(ridge_idxs)*rf
mse31 = mean_squared_error(gt_x2,t_ridges[0]) #mean_squared_error
mse32 = mean_squared_error(gt_x1,t_ridges[1]) #mean_squared_error
mse33 = mean_squared_error(gt_x3,t_ridges[2]) #mean_squared_error
print('\nSignal III - STFT ridges (with SSQ) \nNumber of frequencies = 3')
print('MSE for f1 = ' + str(mse31))
print('MSE for f2 = ' + str(mse32))
print('MSE for f3 = ' + str(mse33))
viz(x3, Tx, ridge_idxs, ssq_freqs, tt = 'Signal III - STFT with SSQ Ridges')
