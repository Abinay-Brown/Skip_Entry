import numpy as np
import scipy.io
from Dynamics import  F, A, B
from Constants import *
from numba import guvectorize, float64


def Linearize(X, U):
    Amat = np.zeros((NUM, nx, nx))
    Bmat = np.zeros((NUM, nx, nu))
    for i in range(NUM):
        #print(i)
        r, theta, phi, Vr, y, psi = Xnom[:, i]
        alpha, sigma = Unom[:, i]
        Amat[i, :, :] = A(r, theta, phi, Vr, y, psi, alpha, sigma)
        Bmat[i, :, :] = B(r, theta, phi, Vr, y, psi, alpha, sigma)
    return Amat, Bmat



@guvectorize(['void(float64[:,:], float64[:,:], float64, float64[:,:], float64[:,:])'], 
             '(n,m),(n,p),()->(n,m),(n,p)', nopython=True, cache=True)
def Discretize_guv_zoh(A, B, dt, Ad, Bd):
    dim1 = 6
    dim2 = 2  
    iter_count = 30  
    N = 500  
    dtau = dt / N  

    for i in range(dim1):
        for j in range(dim1):
            Ad[i, j] = 1.0 if i == j else 0.0

    

    Apow = np.eye(dim1, dtype=np.float64)

    fact = 1.0  # Factorial
    for k in range(1, iter_count + 1):
        
        Apow_new = np.zeros((dim1, dim1), dtype=np.float64)
        for r in range(dim1):
            for s in range(dim1):
                for t in range(dim1):
                    Apow_new[r, s] += Apow[r, t] * A[t, s]
        
        Apow = Apow_new.copy()
        fact *= k
        
        for r in range(dim1):
            for s in range(dim1):
                Ad[r, s] += (Apow[r, s] * dt) / fact


    # Initialize B1d and B2d as zero matrices
    B1d = np.zeros((dim1, dim2), dtype=np.float64)
    
    for i in range(dim1):
        for j in range(dim2):
            Bd[i, j] = 0
            
    # Numerical integration to compute B1d and B2d
    for step in range(1, N + 1):
        tau = step * dtau
        Texp = dt - tau

        # Initialize expm as identity matrix
        expm = np.eye(dim1, dtype=np.float64)
        Apow = np.eye(dim1, dtype=np.float64)
        fact = 1.0

        # Compute expm using Taylor series
        for k in range(1, iter_count + 1):
            Apow_new = np.zeros((dim1, dim1), dtype=np.float64)
            for r in range(dim1):
                for s in range(dim1):
                    for t in range(dim1):
                        Apow_new[r, s] += Apow[r, t] * A[t, s]
            Apow = Apow_new.copy()
            fact *= k
            for r in range(dim1):
                for s in range(dim1):
                    Apow[r, s] *= Texp / fact
            for r in range(dim1):
                for s in range(dim1):
                    expm[r, s] += Apow[r, s]

        # Multiply expm with B: expmB = expm * B
        expmB = np.zeros((dim1, dim2), dtype=np.float64)
        for i in range(dim1):
            for j in range(dim2):
                for k in range(dim1):
                    expmB[i, j] += expm[i, k] * B[k, j]

        # Add the scaled contribution to B1d and B2d
        for i in range(dim1):
            for j in range(dim2):
                B1d[i, j] += expmB[i, j] * dtau
                

    for i in range(dim1):
        for j in range(dim2):
            Bd[i, j] = B1d[i, j] 
            

'''
# Load Nominal trajectory
Nominal = scipy.io.loadmat('Nominal.mat')
Xnom = Nominal['Xnom'].T
Unom = Nominal['Unom'].T
Tnom = Nominal['Tnom']

Amat, Bmat = Linearize(Xnom, Unom)
Ad, Bd = Discretize_guv_zoh(Amat, Bmat, Tnom/(N-1))
Ad = Ad.reshape((NUM, nx, nx))
#Bd = Bd.reshape((NUM, nx, nu))
print(Bd.shape)
'''