
import numpy as np
def grdescent(func,w0,stepsize,maxiter,tolerance=1e-02):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector
	eps = 2.2204e-14 #minimum step size for gradient descent
	w=w0
	if stepsize<eps:
		stepsize=eps
	loss,gradient=func(w)
	i=0
	while np.linalg.norm(gradient)>=tolerance and i<maxiter:
		w-=stepsize*gradient
		newloss,gradient=func(w)
		if newloss>loss:
			stepsize = stepsize*0.5
		elif newloss<loss:
			stepsize*=1.01
		#if stepsize<eps:
			#stepsize=eps
		i+=1
		loss=newloss
	return w

