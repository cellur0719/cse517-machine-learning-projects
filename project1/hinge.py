from numpy import maximum
import numpy as np


def hinge(w,xTr,yTr,lambdaa):
#
#
# INPUT:
# xTr dxn matrix (each column is an input vector)
# yTr 1xn matrix (each entry is a label)
# lambda: regression constant
# w weight vector (default w=0)
#
# OUTPUTS:
#
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
	judge=1-yTr*np.dot(w.transpose(),xTr)
	af_judge=np.maximum(judge,np.zeros((1,yTr.shape[1])))
	loss=af_judge.sum()+np.dot(w.transpose(),w)*lambdaa
	case=(judge>0)
	gradient=2*lambdaa*w-np.dot(xTr,(yTr*case).transpose())
	return loss,gradient

