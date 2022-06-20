'''
Extraction of instantaneous frequencies with SSQ
Test with generated signals
Wilson Cosmo
20/06/2022
'''

import numpy as np
import scipy.signal as sig
from ssqueezepy import ssq_cwt, ssq_stft, extract_ridges
from ssqueezepy.visuals import plot, imshow
import matplotlib.pyplot as plt

#%%## Visual method #########################################################
def viz(x, Tf, ridge_idxs, yticks=None, tt = 'Signal', yl = 'Frequency Scale'):
    ylabel = (yl)
    title = (tt)
    ikw = dict(abs=1, cmap='turbo', title=title)
    pkw = dict(linestyle='--', color='k', xlabel="Time [samples]", ylabel=ylabel, xlims=(0, Tf.shape[1]))
    imshow(Tf, **ikw, show=0)
    plot(ridge_idxs, **pkw, show=1)

#define signal
N = 500 #n samples
f2 = 0.5
f3 = 2.0
penalty = 6000 #penalty for frequency change
t = np.linspace(0, 10, N, endpoint=True)
x1 = sig.chirp(t, f0=2,  f1=8, t1=20, method='linear')
x2 = np.cos(2*np.pi * f2 * t)
x3 = np.cos(2*np.pi * f3 * t)
x = x1 + x2 + x3
plot(x, title="Original Signal", show=1, xlabel="Time [samples]", ylabel="Signal Amplitude [A.U.]") #plot original signal

nr = 3 #n of ridges

gt_x1 = np.zeros(N)
gt_x2 = np.zeros(N)
gt_x3 = np.zeros(N)

for i in range(N):
    gt_x1[i] = (((8-2)/(N-1))*i) + 2
    gt_x2[i] = f2
    gt_x3[i] = f3

plt.title('Original signal frequencies')
plt.plot(gt_x1)
plt.plot(gt_x2)
plt.plot(gt_x3)
plt.show()

# CWT
Tx, Wx, ssq_freqs, scales = ssq_cwt(x, t=t)
ridge_idxs = extract_ridges(Wx, scales, penalty, n_ridges=nr, bw=25)
viz(x, Wx, ridge_idxs, scales, tt = 'CWT Ridges')

# SSQ_CWT example (note the jumps)
ridge_idxs = extract_ridges(Tx, scales, penalty, n_ridges=nr, bw=4)
print(np.shape(ridge_idxs)) #debug
viz(x, Tx, ridge_idxs, ssq_freqs, tt = 'CWT with SSQ Ridges')
