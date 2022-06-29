import numpy as np
from scipy.linalg import toeplitz
#from sympy import *
from numpy import linalg as LA
from scipy.integrate import quad
import matplotlib.pyplot as plt

def functionRlocalscattering(M,theta,ASDdeg,antennaSpacing=1/2,distribution='Gaussian'):
    #Compute the ASD in radians based on input
    ASD = ASDdeg*np.pi/180

    #The correlation matrix has a Toeplitz structure, so we only need to
    #compute the first row of the matrix
    firstRow = np.zeros((M,1), dtype=complex)

    #Go through all the columns of the first row
    for column in range(0, M):#column = 1:M
        
        #Distance from the first antenna
        distance = antennaSpacing*(column)
        
        
        #For Gaussian angular distribution
        if distribution == 'Gaussian':
            
            #Define integrand of (2.23)
            #Delta = Symbol('Delta')
            #F = exp(1j*2*np.pi*distance*sin(theta+Delta))*exp((-Delta**2)/(2*(ASD**2)))/(np.sqrt(2*np.pi)*ASD)
            
            #Compute the integral in (2.23) by including 20 standard deviations
            Real = quad(lambda Delta: (np.exp(1j*2*np.pi*distance*np.sin(theta+Delta))*np.exp((-Delta**2)/(2*(ASD**2)))/(np.sqrt(2*np.pi)*ASD)).real, -20*ASD, 20*ASD)[0]
            Imag = quad(lambda Delta: (np.exp(1j*2*np.pi*distance*np.sin(theta+Delta))*np.exp((-Delta**2)/(2*(ASD**2)))/(np.sqrt(2*np.pi)*ASD)).imag, -20*ASD, 20*ASD)[0]
            firstRow[column,:] = Real + 1j*Imag
            
        #For uniform angular distribution
        elif distribution == 'Uniform':
            
            #Set the upper and lower limit of the uniform distribution
            limits = np.sqrt(3)*ASD
            
            #Define integrand of (2.23)
            #Delta = Symbol('Delta')
            #F = np.exp(1j*2*np.pi*distance*np.sin(theta+Delta))/(2*limits)
            
            #Compute the integral in (2.23) over the entire interval
            Real = quad(lambda Delta: (np.exp(1j*2*np.pi*distance*np.sin(theta+Delta))/(2*limits)).real, -limits, limits)[0]#integrate(F, (Delta, -limits, limits))#integral(F,-limits,limits);
            Imag = quad(lambda Delta: (np.exp(1j*2*np.pi*distance*np.sin(theta+Delta))/(2*limits)).imag, -limits, limits)[0]
            firstRow[column,:] = Real + 1j*Imag

            
        #For Laplace angular distribution
        elif distribution == 'Laplace':
            
            #Set the scale parameter of the Laplace distribution
            LaplaceScale = ASD/np.sqrt(2)
            
            #Define integrand of (2.23)
            #Delta = Symbol('Delta')
            #F = np.exp(1j*2*np.pi*distance*np.sin(theta+Delta))*np.exp(-abs(Delta)/LaplaceScale)/(2*LaplaceScale)
            
            #Compute the integral in (2.23) by including 20 standard deviations
            Real = quad(lambda Delta: (np.exp(1j*2*np.pi*distance*np.sin(theta+Delta))*np.exp(-abs(Delta)/LaplaceScale)/(2*LaplaceScale)).real, -20*ASD, 20*ASD)[0]#integrate(F, (Delta, -20*ASD, 20*ASD))#integral(F,-20*ASD,20*ASD);
            Imag = quad(lambda Delta: (np.exp(1j*2*np.pi*distance*np.sin(theta+Delta))*np.exp(-abs(Delta)/LaplaceScale)/(2*LaplaceScale)).imag, -20*ASD, 20*ASD)[0]
            firstRow[column,:] = Real + 1j*Imag

    #Compute the spatial correlation matrix by utilizing the Toeplitz structure
    R = toeplitz(firstRow)
    return R

#Number of BS antennas
M = 100

#Set the angle of the UE
theta = np.pi/6

#Set the ASD
ASD = 10

#Define the antenna spacing (in number of wavelengths)
antennaSpacing = 1/2 #Half wavelength distance

#Compute spatial correlation matrix with local scattering model and
#different angular distributions
R_Gaussian = functionRlocalscattering(M,theta,ASD,antennaSpacing,'Gaussian')
R_Uniform = functionRlocalscattering(M,theta,ASD,antennaSpacing,'Uniform')
R_Laplace = functionRlocalscattering(M,theta,ASD,antennaSpacing,'Laplace')

#Channel correlation matrix with uncorrelated fading
R_uncorrelated = np.eye(M)

#Extract the eigenvalues and place them in decreasing order
eigenvalues_Gaussian = np.flip(np.sort(LA.eigvals(R_Gaussian).real))
eigenvalues_Uniform = np.flip(np.sort(LA.eigvals(R_Uniform).real))
eigenvalues_Laplace = np.flip(np.sort(LA.eigvals(R_Laplace).real))
eigenvalues_uncorr = np.flip(np.sort(LA.eigvals(R_uncorrelated).real))

#Replace negative eigenvalues with a small positive number 
#(since the correlation matrices should be Hermitian)
for i in range(0, M):
    if eigenvalues_Gaussian[i] < 0 : eigenvalues_Gaussian[i] = 10**(-16)
    if eigenvalues_Uniform[i] < 0 : eigenvalues_Uniform[i] = 10**(-16)
    if eigenvalues_Laplace[i] < 0 : eigenvalues_Laplace[i] = 10**(-16)


## Plot the simulation results
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot()

ax.plot(np.arange(1,M+1),10*np.log10(eigenvalues_Laplace),'r--', label='Laplace')
ax.plot(np.arange(1,M+1),10*np.log10(eigenvalues_Uniform),'b-.', label='Uniform')
ax.plot(np.arange(1,M+1),10*np.log10(eigenvalues_Gaussian),'k', label='Gaussian')
ax.plot(np.arange(1,M+1),10*np.log10(eigenvalues_uncorr),'k:', label='Uncorrelated fading')

ax.set_xlabel('Eigenvalue number in decreasing order')
ax.set_ylabel('Normalized eigenvalue [dB]')
ax.set_ylim([-50, 10])
ax.set_yticks(np.arange(-50,11,5))
ax.set_xticks(np.arange(0,101,10))
ax.legend()

plt.show()