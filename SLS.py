from Constants import *
from Dynamics import A, B 
import scipy.io
import numpy as np

# Load Nominal trajectory
Nominal = scipy.io.loadmat('Nominal.mat')
Xnom = Nominal['Xnom'].T
Unom = Nominal['Unom'].T
Tnom = Nominal['Tnom']

### Setup Block Diagonal Matrices

# Lower Block Shift Matrix
Zblk = np.kron(np.eye(4, k=-1), np.eye(nx))
print(Zblk.shape)
