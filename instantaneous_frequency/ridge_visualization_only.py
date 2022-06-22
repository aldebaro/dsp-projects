
import numpy as np
import scipy.signal as sig
from ssqueezepy import ssq_cwt, ssq_stft, extract_ridges
from ssqueezepy.visuals import plot, imshow
from scipy.io import wavfile as wav
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sfile", type=str, default='./signals/wav/dolphins/SanctSound_CI01_03_dolphins_20190904T064203Z.wav', help="path to the signal file")
parser.add_argument("--t", type=str, default='wav', help="file type (wav, npy)")
parser.add_argument("--penalty", type=int, default=60000, help="penalty for frequency change")
parser.add_argument("--nridges", type=int, default=3, help="#number of ridges")
opt = parser.parse_args()
print(opt)

#Visual method
def viz(x, Tf, ridge_idxs, yticks=None, tt = 'Signal', yl = 'Frequency Scale'):
    ylabel = (yl)
    title = (tt)
    ikw = dict(abs=1, cmap='turbo', title=title)
    pkw = dict(linestyle='--', color='k', xlabel="Time [samples]", ylabel=ylabel, xlims=(0, Tf.shape[1]))
    imshow(Tf, **ikw, show=0)
    plot(ridge_idxs, **pkw, show=1)

penalty = opt.penalty
nr = opt.nridges #n of ridges

#load signal
if opt.t == 'wav':
    (rate,sig) = wav.read(opt.sfile)
elif opt.t == 'npy':
    sig = np.load(opt.sfile)
else:
    print('File extension not supported.')

x = sig
t = int(sig.size)
plot(x, title="Original Signal", show=1, xlabel="Time [samples]", ylabel="Signal Amplitude [A.U.]") #plot original signal

# CWT
Tx, Wx, ssq_freqs, scales = ssq_cwt(x)
ridge_idxs = extract_ridges(Wx, scales, penalty, n_ridges=nr, bw=25)
viz(x, Wx, ridge_idxs, scales, tt = 'CWT Ridges')

# SSQ_CWT example (note the jumps)
ridge_idxs = extract_ridges(Tx, scales, penalty, n_ridges=nr, bw=4)
viz(x, Tx, ridge_idxs, ssq_freqs, tt = 'CWT with SSQ Ridges')
