import numpy as np

class CorrelatedComplexNoiseGenerator:
    def __init__(self, corrMat):
        self.CorrelatedComplexNoiseGenerator(corrMat)
    
    def CorrelatedComplexNoiseGenerator(self, covMat):
        D,V = np.linalg.eig(covMat)
        self.corrMat = V * np.sqrt(D)
        return self
    
    def generate(self, nSamples):
        d = np.shape(self.corrMat)[0]
        complexNoise1D = np.sqrt(0.5) * np.matmul(np.random.randn(d * nSamples, 2), np.array([1, 1j]))
        r = np.matmul(self.corrMat, complexNoise1D.reshape((d, nSamples)))
        return r
    
def nmser(x,y):
    z=0
    if len(x)==len(y):
        for k in range(len(x)):
            z = z + (((x[k]-y[k])**2)/x[k])    
            z = z/(len(x))
    return z


## time
N = 10**5

# covariance matrix
dimension = 4
corrMat = np.random.randn(dimension, dimension) + 1j * np.random.randn(dimension, dimension)
R_bcu = np.matmul(corrMat, corrMat.conj().T)

#noise generator
corrNoiseGen = CorrelatedComplexNoiseGenerator(R_bcu)

channelsOverTime = np.zeros((dimension, N), dtype=complex) 

#initial channel at t = 1
chan = corrNoiseGen.generate(1)
channelsOverTime[:,0] = chan[:,0]

alpha = 0.5
corrNoiseGen = CorrelatedComplexNoiseGenerator( (1 - alpha**2) * R_bcu)

for t in range(1, N):
    channelsOverTime[:,t] = alpha * channelsOverTime[:, t-1] + corrNoiseGen.generate(1)[:,0]

print('estimated covariance matrix ')
R_estimated = np.matmul(channelsOverTime, channelsOverTime.conj().T) / N
print(R_estimated)

print('true covarience matrix')
print(R_bcu)

#print('difference between the matrices')
#print(R_bcu-R_estimated)

print('difference between the matrices')
erro = nmser(R_bcu, R_estimated)
print(erro)