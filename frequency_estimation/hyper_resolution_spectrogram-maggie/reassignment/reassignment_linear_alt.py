### alternative version of reassignment with linear-scale frequency scan ###
# author: Jiayi (Maggie) Zhang <jiayizha@andrew.cmu.edu>

# based on 
# 1) the original Matlab implementation (see reassignmentgw.m),
# 2) a previous python implementation by radekosmulski
#    github repo: https://github.com/earthspecies/spectral_hyperresolution
# 3) consultation with Prof. Magnasco (co-author of the origigal paper) 

# Important note
# THIS IS AN ALTERNATIVE IMPLEMENTATION AND TAKES DIFFERENT PARAMETERS
# COMPARED TO THE MATLAB CODE.
# More specifically, instead of taking *noct* (see reassignmentgw.m, it specifies the number of divisions per octave/thus determines frequency step-size), this implementation takes in *ndiv*, which specifies directly the number of frequency bins. For example, if minf=0, maxf=5000 and ndiv=100, each bin in the histogram will aacount for 50 frequencies.
# In addition, instea of taking *q*, which indicates the temporal resolution and will be used in the calculation of *sigma* for each band of frequencies, this implementation takes in *sigma* directly. Thus *sigma* value is universal for all bands of analysis.

import torch
import torch.fft as fft
import math
import numpy as np

# for debugging
torch.set_printoptions(precision=10)

def high_resolution_spectrogram(x, sigma, tdeci, over, minf, maxf, sr, ndiv,    lint=0.2, device=torch.device('cuda'), dtype=torch.float32):
    '''
    x        signal
    sigma    a measure of temporal resolution
    tdeci    temporal step-size 
             (determines the width of the resultant spectrogram)
    over     oversampling
    minf     minimum frequency
    maxf     maximum frequency
    sr       sampling rate
    ndiv     number of divisions/bins in the frequency axis
    lint     threshold for ignoring reassignments that are too far
             (points that are >= lint away from its original location
              after the reassignment will be disgarded, default to 0.2)
    device   type of device to run the program on
             (default to torch.device('cuda') to utilize gpu)
    dtype    datatype of input, default to torch.float32
    '''
    eps = 1e-20 # bias to prevent NaN with division

    N = len(x) # assumption: input is mono-channeled
    x = torch.tensor(x, device=device, dtype=dtype)
    xf = fft.fft(x)
    
    # dimensions of the final histogram
    HT = math.ceil(N/tdeci) # time bins 
    HF = ndiv # freq bins
    df = (maxf-minf)/(ndiv-1) # frequncy step-size

    # allocate the space for the final histogram
    # histo - stores actual values, histc - stores counts of accumulation
    histo = torch.zeros(HT, HF, device=device, dtype=dtype)
    histc = torch.zeros(HT, HF, device=device, dtype=dtype)

    f = torch.arange(N, device=device) / N 
    f = torch.where(f>0.5, f-1, f)
    f *= sr
    f_steps = np.linspace(minf, maxf, HF*over+1)

    for f0 in f_steps:
        # make Gaussian window over entire frequency axis (centered at f0)
        gau = torch.exp(-torch.square(f-f0)/(2*sigma**2)) 
        # calculate eta over the entire frequency axis (centered at f0)
        gde = -1/sigma**1 * (f-f0) * gau
        
        # compute reassignment operators
        xi = fft.ifft(gau.T * xf)
        eta = fft.ifft(gde.T * xf)

        # calculate complex shift
        mp = torch.div(eta, xi+eps)
        # calculate energy
        ener = (xi.abs())**2

        # compute instantaneous time and frequency
        tins = torch.arange(1, N+1, device=device) + torch.imag(mp)/(2*math.pi*sigma)
        fins = f0 - torch.real(mp)*sigma

        # mask the reassignment result to only keep points 
        # that are within the histrogram dimensions
        mask = (torch.abs(mp)<lint) & (fins<maxf) & (fins>minf) & (tins>=1) & (tins<N)

        tins = tins[mask]
        fins = fins[mask]
        ener = ener[mask]

        #NOTE: matlab code pipes gpu array into cpu here
        itins = torch.round(tins/tdeci+0.5)-1 # so that itins >= 0 
        ifins = torch.round((maxf-fins)/df)
        idx = itins.long()*HF+ifins.long() 

        histo.put_(idx, ener, accumulate=True)
        histc.put_(idx, (0*itins+1), accumulate=True)

    mm = histc.max()
    histo[histc < torch.sqrt(mm)] = 0
    return histo



