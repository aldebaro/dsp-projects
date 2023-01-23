import numpy as np

#from numba import jit
from collections import namedtuple
from lib_dolphin.dtw import * 


Symbol = namedtuple("Symbol", "id type")


#@jit
def levenstein(x, y):
    n = len(x)
    m = len(y)     
    w = int(imax(n, m) * BAND)
    w = imax(w, abs(n - m)) + 2
    d = np.ones((n + 1, m + 1)) * np.Infinity
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(imax(1, i - w), imin(m + 1, i + w)):
            error = 0
            if x[i - 1] != y[j - 1]:
                error += 1
            d[i, j] = min([
                d[i - 1, j] + 1,
                d[i, j - 1] + 1,
                d[i - 1, j - 1] + error
            ])
    return d[n, m] / (n * m)


#@jit
def levenstein_distances(X):
    N = len(X)
    d = np.zeros((N, N))
    for i in range(0, N):
        if i % 100 == 0:
            print(" ... distances {} / {} = {}".format(i, N, i / N))
        for j in range(i + 1, N):
            distance = levenstein(X[i], X[j])
            d[i, j] = distance
            d[j, i] = d[i, j]
    return d


def symbols(clusters, classifications):
    return [Symbol(c, l) for c, l in zip(clusters, classifications)]

