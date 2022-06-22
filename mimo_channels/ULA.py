import numpy as np
#import GeneralFunctions as GF
from numpy import linalg as LA

import matplotlib.pyplot as plt

def ak_fftmtx(N, option=1):
    # function [Ah, A] = ak_fftmtx(N, option)
    #FFT direct Ah and inverse A matrices with 3 options for
    #the normalization factors:
    #1 (default) ->orthonormal matrices
    #2->conventional discrete-time (DT) Fourier series
    #3->used in Matlab/Octave, corresponds to samples of DTFT
    #Example that gives the same result as Matlab/Octave:
    # Ah=ak_fftmtx(4,3); x=[1:4]'; Ah*x, fft(x)
    W = np.exp(-1j*2*np.pi/N) #twiddle factor W_N
    Ah=np.zeros((N,N), dtype=complex) #pre-allocate space
    for n in range(0, N):#n=0:N-1 #create the matrix with twiddle factors
        for k in range(0, N):#k=0:N-1
            Ah[k,n] = W ** (n*k)
    #choose among three different normalizations
    if option == 1: #orthonormal (also called unitary)
        Ah = Ah/np.sqrt(N)
        A = np.conj(Ah)
    elif option == 2: #read X(k) in Volts in case x(n) is in Volts
        A = np.conj(Ah)
        Ah = Ah/N
    elif option == 3: #as in Matlab/Octave: Ah = Ah;
        A = np.conj(Ah)/N
    else:
        print(['Invalid option value: ', str(option)])
    return Ah, A

class ULA:
    def __init__(self, Na, phi, normDistance, ph0, w, r_array):
        self.phi = phi
        self.Na = Na
        self.normDistance = normDistance
        self.ph0 = ph0
        self.w = w
        self.r_array = r_array

    def cos_gamma(self, phi = np.linspace(-np.pi,np.pi,1000), theta = np.pi/2):
        if self.r_array == 1:
            r = np.sin(theta)*np.cos(phi)
        elif self.r_array == 2:
            r = np.sin(theta)*np.sin(phi)
        elif self.r_array == 3:
            r = np.cos(phi) #theta
        else:
            print('Invalid arguments')
        
        return r

    def psiULA(self):
        theta = np.pi/2
        directional_cossine = self.cos_gamma(self.phi, theta)
        self.psi = 2*np.pi*self.normDistance*directional_cossine
        return self.psi.reshape(-1,1)

    def steeringvectorsForULA(self):
        self.psi = self.psiULA()
        self.steeringvectors = np.exp(np.kron(np.arange(0, self.Na), -1j*self.psi))
        return self.steeringvectors


    def steeringweightsULA(self):
        theta0 = np.pi/2
        directional_cossine= self.cos_gamma(self.ph0, theta0)
        beta = 2 * np.pi * self.normDistance * directional_cossine             # scanning phase in psi-space
        self.wsteer = np.exp(1j*np.arange(0, self.Na)*beta)

        return self.wsteer

    def randomweights(self):
        self.w =  np.random.randn(1,self.Na) + 1j* np.random.randn(1,self.Na)
        # w = w.reshape(-1,1) / LA.norm(w) #make it a column vector and normalize it
        self.w = self.w / LA.norm(self.w) #make it a column vector and normalize it
        return self.w

    def dftweights(self, f):
        self.dftMatrix, *_ = f(self.Na,3)    
        return self.dftMatrix

    def arrayfactor(self):
        self.steeringvectors = self.steeringvectorsForULA()
        self.wsteer = self.steeringweightsULA()
        self.arrayfactorULA = np.matmul(self.steeringvectors, self.wsteer)/np.sqrt(self.Na)

        return self.arrayfactorULA

    def PlotCodeBook(self, f, ind=-1):
        self.steeringvectors = self.steeringvectorsForULA()
        DFTmatrix = self.dftweights(f)
        self.arrayfactorULA = np.zeros((self.steeringvectors.shape[0], 1, self.Na))
        for i in range(0, self.Na):
            wx = DFTmatrix[:,i].reshape(-1,1)
            
            self.arrayfactorULA[:,:,i] = np.matmul(self.steeringvectors, wx)/np.sqrt(Na)

            #plt.polar(phi.T,np.absolute(self.arrayfactorULA[:,:,i]))
            #plt.show()
        
        if ind == -1:
            pass
        elif ind > self.Na:
            print("Invalid Index")
        else:
            plt.polar(phi.T,np.absolute(self.arrayfactorULA[:,:,ind-1]))
            plt.show()
    
    def get_los_geometric_channel(self, a):
        self.arrayfactorULA = self.arrayfactor()
        h = a * np.exp(-1j * 2 * np.pi * self.normDistance) * np.transpose(self.arrayfactorULA)
        return h

    def CorrelatedComplexNoiseGenerator(self, covMat):
        D,V = np.linalg.eig(covMat)
        self.corrMat = V * np.sqrt(D)
        return self
    
    def generate(self, nSamples):
        d = np.shape(self.corrMat)[0]
        complexNoise1D = np.sqrt(0.5) * np.matmul(np.random.randn(d * nSamples, 2), np.array([1, 1j]))
        r = np.matmul(self.corrMat, complexNoise1D.reshape((d, nSamples)))
        return r

######### Parameters #########
Na=8 #num antennas
Npointing=4 # num of pointing angles

N=1000 #grid resolution
phi=np.linspace(-np.pi,np.pi,N) #angles
phi=phi[np.newaxis, :]

d=1
lambdac=2
normDistance=d/lambdac # distance between the antennas
r_array = 3

w = 1#np.ones(Na)

RxPos = np.array([5, np.pi/4]) # RxPos[0]: Module       RxPos[1]: Angle in rad
pointingAngle = RxPos[1]
a = 1 # Gain

######### Create my Class #########
test1 = ULA(Na, phi, normDistance, pointingAngle, w, r_array)

######### Plot arrayfactor #########
arrayfactor = test1.arrayfactor()
phi2=np.linspace(-np.pi,np.pi,N)

# Points to draw RxPos and TxPos on the graph
RxPos_Module = RxPos[0]
RxPos_angle = RxPos[1]

Tx_module = np.arange(0, Na)
Tx_angle = np.repeat(np.pi/2, Na)

# Create the figure
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection='polar')

# Draw the array
ax.plot(Tx_angle, Tx_module, '--bo', label='Transmit array')

# Draw Receptor
ax.plot(RxPos_angle, RxPos_Module, 'o', color = 'y', label='Rx position')

# Draw Beamforming
ax.plot(phi2.T, np.absolute(arrayfactor))
ax.legend()
plt.show()


######### Get LOS channel #########
h_LOS = test1.get_los_geometric_channel(a)

######### Get channel with correlation #########
# time
N = 10**5

# covariance matrix
dimension = 4
corrMat = np.random.randn(dimension, dimension) + 1j * np.random.randn(dimension, dimension)
R_bcu = np.matmul(corrMat, corrMat.conj().T)

#noise generator
corrNoiseGen = test1.CorrelatedComplexNoiseGenerator(R_bcu)

channelsOverTime = np.zeros((dimension, N), dtype=complex) 

#initial channel at t = 1
chan = corrNoiseGen.generate(1)
channelsOverTime[:,0] = chan[:,0]

alpha = 0.5
corrNoiseGen = test1.CorrelatedComplexNoiseGenerator( (1 - alpha**2) * R_bcu)

for t in range(1, N):
    channelsOverTime[:,t] = alpha * channelsOverTime[:, t-1] + corrNoiseGen.generate(1)[:,0]

print('estimated covariance matrix ')
R_estimated = np.matmul(channelsOverTime, channelsOverTime.conj().T) / N
print(R_estimated)

print('true covarience matrix')
print(R_bcu)

print('difference between the matrices')
dif = R_bcu-R_estimated
print(dif)