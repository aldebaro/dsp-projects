### reassignment with linear-scale frequency scan ###
# author: Jiayi (Maggie) Zhang <jiayizha@andrew.cmu.edu>

# based on 
# 1) the original Matlab implementation (see reassignmentgw.m),
# 2) a previous python implementation by radekosmulski
#    github repo: https://github.com/earthspecies/spectral_hyperresolution
# 3) consultation with Prof. Magnasco (co-author of the origigal paper) 

# Important note
# Compared to the reassignment_linear_alt.py version, this is closer to the original Matlab implementation. Referring to the Matlab code (in case of confusion) is encouraged.

import torch
import torch.fft as fft
import math
import numpy as np

# for debugging
torch.set_printoptions(precision=10)

def high_resolution_spectrogram(x, q, tdeci, over, noct, minf, maxf, lint=0.2, device=torch.device('cuda'), dtype=torch.float32):
    '''
    x        signal
    q        a measure of temporal resolution
             (>1 yields tone-like representation, <1 yields click-like rep)
    tdeci    temporal step-size 
             (determines the width of the resultant spectrogram)
    over     oversampling
    minf     minimum frequency
    maxf     maximum frequency
             (both minf, maxf are numbers (0, 1], maxf should be <=0.5)
    lint     threshold for ignoring reassignments that are too far
             (points that are >= lint away from its original location
              after the reassignment will be disgarded, default to 0.2)
    device   type of device to run the program on
             (default to torch.device('cuda') to utilize gpu)
    dtype    datatype of input, default to torch.float32
    '''
    eps = 1e-20 # bias to prevent NaN with division
    N = len(x) # assumption: input mono sound

    x = torch.tensor(x, device=device, dtype=dtype)
    xf = fft.fft(x) # complex matrix
    
    # dimensions of the final histogram
    HT = math.ceil(N/tdeci) # time bins 
    HF = math.ceil(-noct*math.log2(minf/maxf)+1) # freq bins

    # allocate the space for the final histogram
    # histo - stores actual values, histc - stores counts of accumulation
    histo = torch.zeros(HT, HF, device=device, dtype=dtype)
    histc = torch.zeros(HT, HF, device=device, dtype=dtype)

    f = torch.arange(N, device=device) / N
    f = torch.where(f>0.5, f-1, f)

    for log2f0 in range(HF*over):
        f0 = minf*2**(log2f0/over/noct)
        sigma = f0/(2*math.pi*q)
        # make Gaussian window over the entire frequency axis (centered at f0)
        gau = torch.exp(-torch.square(f-f0)/(2*sigma**2))
        # calcualte eta over the entire frequency axis (centered at f0)
        gde = -1/sigma**1 * (f-f0) * gau

        # compute reassignment operators
        #avoid warning message from torch library (results were checked and are the same):
        #xi = fft.ifft(gau.T * xf)
        #eta = fft.ifft(gde.T * xf)
        xi = fft.ifft( gau.permute(*torch.arange(gau.ndim - 1, -1, -1))  * xf)
        eta = fft.ifft( gde.permute(*torch.arange(gde.ndim - 1, -1, -1))  * xf)

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
        #NOTE: this is so written to mirror the Matlab implementation
        itins = torch.round(tins/tdeci+0.5)-1    
        ifins = torch.round(-noct*torch.log2(fins/maxf)+1)-1
        idx = itins.long()*HF+ifins.long()

        histo.put_(idx, ener, accumulate=True)
        histc.put_(idx, (0*itins+1), accumulate=True)

    mm = histc.max()
    histo[histc < torch.sqrt(mm)] = 0 # ~ filter noise out
    return histo
