# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:03:09 2019

@author: Jerry Xing
"""
import numpy as np
def grdescent(func, w0, stepsize, maxiter, tolerance = 1e-2):
#% function [w]=grdescent(func,w0,stepsize,maxiter,tolerance)
#%
#% INPUT:
#% func function to minimize
#% w0 = initial weight vector 
#% stepsize = initial gradient descent stepsize 
#% tolerance = if norm(gradient)<tolerance, it quits
#%
#% OUTPUTS:
#% 
#% w = final weight vector
#%
    w = w0
    ## << Insert your solution here
    eps = 2.2204e-14 #minimum step size for gradient descent
    loss, gradient = func(w)
    i = 0
    while i < maxiter and np.linalg.norm(gradient)>tolerance:

        w -= stepsize * gradient
        newloss = loss
        loss, gradient = func(w)
        #print(i)
        i += 1
        if loss<=newloss:
            stepsize = stepsize*1.01
        else:
            stepsize = stepsize*0.5

    ## >>    
    return w