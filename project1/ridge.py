import numpy as np


def ridge(w,xTr,yTr,lambdaa):
#
# INPUT:
# w weight vector (default w=0)
# xTr:dxn matrix (each column is an input vector)
# yTr:1xn matrix (each entry is a label)
# lambdaa: regression constant
#
# OUTPUTS:
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
#
# [d,n]=size(xTr);
    l1=w.transpose().dot(xTr)-yTr #try
    loss=l1.dot(l1.transpose())+w.transpose().dot(w)*lambdaa
    gradient=2*xTr.dot(np.dot(xTr.transpose(),w))-2*xTr.dot(yTr.transpose())+2*lambdaa*w
    return loss,gradient
