import scipy.io
import numpy as np
import cvxpy as cp
from Dynamics import F, A, B
from Constants import *


class SLS_optimizer:

    def __init__(self):

        return

    def initParameters(self, Xnom, Unom):
        Amat, Bmat, Zmat = self.Linearize(Xnom, Unom)
        self.Ablk = cp.Parameter((NUM * nx, NUM * nx))
        self.Bblk = cp.Parameter((NUM * nx, NUM * nu))
        self.Zblk = cp.Parameter((NUM * nx, NUM * nx))
        self.Iblk = cp.Parameter((NUM * nx, NUM * nx))

        self.Ablk.value = Amat
        self.Bblk.value = Bmat
        self.Zblk.value = Zmat
        self.Iblk.value = np.eye((NUM * nx))

        # Need to setup E matrix
        # Need to setup mu: for that solve hessian maximization problem
        # Need to setup c to select elements
        # Need to setup b for bounds
        return True

    def initDecisionVariables(self):
        self.z = cp.Variable((nx, NUM))
        self.v = cp.Variable((nu, NUM))
        
        self.Phi_x = cp.Variable((NUM * nx, NUM * nx))
        self.Phi_u = cp.Variable((NUM * nu, NUM * nx))
        self.tau   = cp.Variable((NUM)) # Double check the dimension if NUM + 1
        
        return True
    def initConstraints(self):
        self.Constraints = []
        # Boundary Conditions
        # Dynamics Constraints
        # Control limits
        
        # Phi_x & Phi_u upper diagonal constraint
        for i_blk in range(NUM):        
            for j_blk in range(NUM):    
                if j_blk > i_blk:     
                    self.constraints += [Phi_x[i_blk*nx:(i_blk+1)*nx, j_blk*nx:(j_blk+1)*nx] == 0]
                    self.constraints += [Phi_u[i_blk*nu:(i_blk+1)*nu, j_blk*nx:(j_blk+1)*nx] == 0]
    
                    
        # [I - ZA - ZB]*Phi = [I] Constraint
        Phi = cp.vstack([self.Phi_x, self.Phi_u])
        self.Constraints += [(self.Iblk - self.Zblk@self.Ablk - self.Zblk@self.Bblk)@phi == self.Iblk]
        # 27d constraint
        # 27e constraint
            
        return
    
    def updateParameters(self):
        
        return True 

    def Linearize(self, Xnom, Unom):
        Ablk = np.zeros((NUM * nx, NUM * nx))
        Bblk = np.zeros((NUM * nx, NUM * nu))
        Zblk = np.zeros((NUM * nx, NUM * nx))
        idx = nx 
        idu = nu
        for i in range(NUM):
            
            r, theta, phi, Vr, y, psi = Xnom[:, i]
            alpha, sigma = Unom[:, i]
            Ablk[idx-nx : idx, idx-nx : idx] = A(r, theta, phi, Vr, y, psi, alpha, sigma)
            Bblk[idx-nx : idx, idu-nu : idu] = B(r, theta, phi, Vr, y, psi, alpha, sigma)
            if i > 0:
                Zblk[idx-nx:idx, idx-2*nx:idx-nx] = np.eye(nx)
            idx = idx + nx
            idu = idu + nu
        return Ablk, Bblk, Zblk
    
Nominal = scipy.io.loadmat('Nominal.mat')
Xnom = Nominal['Xnom'].T
Unom = Nominal['Unom'].T
Tnom = Nominal['Tnom']
    
opt = SLS_optimizer()
Ablk, Bblk, Zblk = opt.Linearize(Xnom, Unom)
