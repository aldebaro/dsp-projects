import numpy as np
import multiprocessing as mp

#from numba import jit


#@jit(nopython=True)        
def min3(a, b, c):    
    x = a
    if b < x:
        x = b
    if c < x:
        x = c
    return x


#@jit(nopython=True)        
def imax(a, b):    
    x = a
    if b > x:
        x = b
    return x


#@jit(nopython=True)        
def imin(a, b):    
    x = a
    if b < x:
        x = b
    return x


BAND = 0.1


#@jit(nopython=True)        
def dtw(x, y, band=BAND, gap_penalty = 0.0):
    N = x.shape[0]
    M = y.shape[0]
    w = int(imax(N, M) * band)
    w = imax(w, abs(N - M)) + 2
    dp = np.ones((N + 1, M + 1)) * np.Infinity   
    dp[0, 0] = 0.0    
    for i in range(1, N + 1):    
        for j in range(imax(1, i - w), imin(M + 1, i + w)):            
            dist = np.sqrt(np.sum(np.square(x[i - 1] - y[j - 1])))
            dp[i, j] = dist + min3(
                dp[i - 1, j] + gap_penalty,
                dp[i - 1, j - 1],
                dp[i, j - 1] + gap_penalty
            )
    return dp[N, M] / (N * M)


#@jit
def dtw_distances(X, band=BAND, gap_penalty = 0.0):
    N = len(X)
    d = np.zeros((N, N))
    for i in range(0, N):
        if i % 100 == 0:
            print(" ... distances {} / {} = {}".format(i, N, i / N))
        for j in range(i + 1, N):
            distance = dtw(X[i], X[j], band, gap_penalty)
            d[i, j] = distance
            d[j, i] = d[i, j]
    return d
        
