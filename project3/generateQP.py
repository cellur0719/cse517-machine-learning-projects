"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
C : regularization constant

Output:
Q,p,G,h,A,b as defined in cvxopt.solvers.qp

A call of cvxopt.solvers.qp(Q, p, G, h, A, b) should return the optimal nx1 vector of alphas
of the SVM specified by K, yTr, C. Just make these variables np arrays and keep the return 
statement as is so they are in the right format. See this reference to assign variables:
https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf.
"""	
import numpy as np
from cvxopt import matrix

def generateQP(K, yTr, C):
    yTr = yTr.astype(np.double)
    n = yTr.shape[0]
    
    Q=np.dot(yTr,yTr.transpose())*K
    p = -1 * np.ones((n,1))

    A=yTr.transpose()
    b = 0.00

    h = np.concatenate((np.zeros((n, 1)), C * np.ones((n, 1))), axis = 0)
    G = np.concatenate((-1 * np.identity(n), np.identity(n)), axis = 0)
    
            
    return matrix(Q), matrix(p), matrix(G), matrix(h), matrix(A), matrix(b)

