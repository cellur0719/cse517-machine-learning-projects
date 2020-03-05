import math
import numpy as np

'''

    INPUT:
    xTr dxn matrix (each column is an input vector)
    yTr 1xn matrix (each entry is a label)
    w weight vector (default w=0)

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [d,n]=size(xTr);
'''
def logistic(w,xTr,yTr):
	multiplier=yTr*np.dot(w.transpose(),xTr)
	loss=np.sum(np.log(1+np.exp(-multiplier)))
	gradient=np.sum(-xTr*yTr/(1+np.exp(multiplier)),axis=1).reshape(w.shape[0],1)
	return loss,gradient
