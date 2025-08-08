import numpy as np
import scipy.io
from numpy import pi
from Dynamics import F, A, B, H1, H2, H3, H4, H5, H6
from Constants import *
'''
X: nominal state   [nx, N]
U: nominal control [nu, N]
X0_set: AABB [nx, N]
W_set: Exogenous Disturbance    [nx, N]
Wl_set: Linearization Error set [nx, N]

'''
def Reach_set_solver(X, U, ):

    return

'''
X0_set: +/- limits (nx, 1)
'''
def Reach_Linearization_Errors(X0_set, Phi= 0, method = 'sum'):
    
    M = 5000 # sampling lengths
    S = 50000 # Total samples
    H = [H1, H2, H3, H4, H5, H6]
    Xk_set = X0_set
    mu = np.zeros((nx, N))
        
    # Loop through trajectory points
    for i in range(N):
        
        # Sample points from the tube
        r_set       = np.linspace(Xk_set[0, 0], Xk_set[0, 1], M)
        theta_set   = np.linspace(Xk_set[1, 0], Xk_set[1, 1], M)
        phi_set     = np.linspace(Xk_set[2, 0], Xk_set[2, 1], M)
        Vr_set      = np.linspace(Xk_set[3, 0], Xk_set[3, 1], M)
        y_set       = np.linspace(Xk_set[4, 0], Xk_set[4, 1], M)
        psi_set     = np.linspace(Xk_set[5, 0], Xk_set[5, 1], M)
        alpha_set   = np.linspace(Xk_set[6, 0], Xk_set[6, 1], M)
        sigma_set   = np.linspace(Xk_set[7, 0], Xk_set[7, 1], M)
        
        # Sample different indices 
        rng = np.random.default_rng()
        ind = rng.integers(0, M, size=(nx + nu, S))
        #break
        # Loop through the Hessians 1 to 6
        for k in range(nx):
            hess_max  = 0
            for j in range(S):
                Hess = H[k](r_set[ind[0, j]], theta_set[ind[1, j]], phi_set[ind[2, j]], Vr_set[ind[3, j]], y_set[ind[4, j]], psi_set[ind[5, j]], alpha_set[ind[6, j]], sigma_set[ind[7, j]])
                
                if method == 'fro':
                    val = 0.5 * np.linalg.norm(Hess, 'fro')
                    
                elif method == 'sum':
                    val = 0.5 * np.sum(np.abs(Hess))
                    
                if j == 0:
                    hess_max = val
                elif val > hess_max:
                    hess_max = val
            mu[k, i] = hess_max
        break       
        
    return mu[:, 0]

# Load Nominal trajectory
Nominal = scipy.io.loadmat('Nominal.mat')
Xnom = Nominal['Xnom'].T
Unom = Nominal['Unom'].T
Tnom = Nominal['Tnom']
X0_set = np.zeros((nx + nu, 2))
delta = np.array([100, 0.0001*(pi/180), 0.0001*(pi/180), 50, 0.5*(pi/180), 0.5*(pi/180), 0.1*(pi/180),  0.1*(pi/180)]).reshape(nx+nu, 1)
X0_set[:, 0] = np.vstack((Xnom[:, 0].reshape(nx,1), Unom[:, 0].reshape(nu, 1)))[:, 0] + delta[:, 0]
X0_set[:, 1] = np.vstack((Xnom[:, 0].reshape(nx, 1), Unom[:, 0].reshape(nu, 1)))[:, 0] - delta[:, 0]
print(Xnom[:, 0])
#print(X0_set)
ans1 = Reach_Linearization_Errors(X0_set, 0, 'fro')
ans2 = Reach_Linearization_Errors(X0_set, 0, 'sum')
print(ans1)
print(ans2)
