import numpy as np
from numpy import *

"""
function D=l2distance(X,Z)
	
Computes the Euclidean distance matrix. 
Syntax:
D=l2distance(X,Z)
Input:
X: dxn data matrix with n vectors (columns) of dimensionality d
Z: dxm data matrix with m vectors (columns) of dimensionality d

Output:
Matrix D of size nxm 
D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
"""

def l2distance(X,Z):
    d, n = X.shape
    dd, m = Z.shape
    assert d == dd, 'First dimension of X and Z must be equal in input to l2distance'
    
    D = np.zeros((n, m))

    X_square = (X**2).sum(axis=0)

    Z_square = (Z**2).sum(axis=0)

    first = np.tile(X_square, (m,1)).transpose()
    second = np.tile(Z_square, (n,1))
    third = -2 * np.dot(X.transpose(),Z)

    D = first + second + third
    D[D<0]=0
    D = np.sqrt(D)
    D[D<0]=0
    
    return D


