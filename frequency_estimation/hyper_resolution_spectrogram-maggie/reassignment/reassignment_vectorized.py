### vectorized implementation of the reassignment algorithm
# author: Jiayi (Maggie) Zhang <jiayizha@andrew.cmu.edu>
# author of the first implementation: Radek Osmulski

# based on 
# 1) a previous python implementation by radekosmulski
#    github repo: https://github.com/earthspecies/spectral_hyperresolution
# 2) the non-vectorized implementation in the current repository

# Important note 1
# This is an vectorized version of the algorithm, meant to speed up the calculations even more (esp. with gpu). Mr. Osmulski developed the first version; the author here adapated it to newer PyTorch methods and simplified the code (the most notable modification is the removal of unnecesarry tensor allocations).

# Import note 2
# The results from the vectorized version might be DIFFERENT from the those yielded from the iterative versions. The (relative) representations seemed to be correct (upon inspection of the resultant histogram), but the values were not the same. Use this implemenation at your discretion and double check with results from other versions, if possible.

import torch
import torch.fft as fft
import math

def high_resolution_spectrogram(x, q, tdeci, over, noct, minf, maxf, lint=0.2, device=torch.device('cuda'), dtype=torch.float32, chunks=200):
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
    chuncks  number to divide original singal into ##HERE!!!!!!!!
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

    f = torch.arange(N, device=device)/N
    f = torch.where(f>0.5, f-1, f)

    # chunking the frequency range
    log2f0ss = torch.arange(0, HF*over, device=device).chunk(chunks=chunks)

    for log2f0s in log2f0ss:
        f0s = minf*2**(log2f0s/over/noct)
        n = f0s.shape[0]
        sigma = f0s/(2*math.pi*q)

        fs = f.unsqueeze(1).expand(-1, n) # make sure dimensions match up
        # make Gaussian window over the entire frequency axis (centered at f0)
        gau = torch.exp(-torch.square(fs-f0s)/(2*sigma**2))
        # calcualte eta over the entire frequency axis (centered at f0)
        gde = -1/sigma**1 * (fs-f0s) * gau

        xf_sub = xf.unsqueeze(0).expand(n, -1) # make sure dimensions match up
        # compute reassignment operators
        xi = fft.ifft(gau.T * xf_sub)
        eta = fft.ifft(gde.T * xf_sub)

        # calculate complex shift
        mp = torch.div(eta, xi+eps)
        # calculate energy
        ener = torch.square(torch.abs(xi))

        # compute instantaneous time and frequency
        tins = torch.arange(1, N+1, device=device).unsqueeze(0).expand(n, -1) + torch.imag(mp)/(2*math.pi*sigma).unsqueeze(1)
        fins = f0s.unsqueeze(1) - torch.real(mp)*(sigma.unsqueeze(1))

        # mask the reassignment result to only keep points 
        # that are within the histrogram dimensions
        mask = (torch.abs(mp)<lint) & (fins<maxf) & (fins>minf) & (tins>=1) & (tins<N)

        tins = tins[mask]
        fins = fins[mask]
        ener = ener[mask]

        #HERE: matlab code pipes gpu array into cpu here
        #NOTE: this is so written to mirror the Matlab implementation
        itins = torch.round(tins/tdeci+0.5)-1    
        ifins = torch.round(-noct*torch.log2(fins/maxf)+1)-1
        idx = itins.long()*HF+ifins.long()  

        histo.put_(idx, ener, accumulate=True)
        histc.put_(idx, (0*itins+1), accumulate=True)

    mm = histc.max()
    histo[histc < torch.sqrt(mm)] = 0 # ~ filter noise out
    return histo



