# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:30:29 2019

@author: remus
"""
import numpy as np
def forward_pass(W, xTr, trans_func):
#% function [as,zs]=forward_pass(W,xTr,trans_func)
#%
#% INPUT:
#% W weights (list of numpy ndarray)
#% xTr dxn numpy ndarray (each column is an input vector)
#% trans_func transition function to apply for inner layers
#%
#% OUTPUTS:
#%
#% aas = result of forward pass 
#% zzs = result of forward pass (zs[0] output layer of the forward pass) 
#%
    n = np.shape(xTr)[1]    
    
    # First, we add the constant weight
    zzs = [None]*(len(W)+1);   zzs[-1] = np.vstack((xTr, np.ones([1, n])))
    aas = [None]*(len(W)+1);   aas[-1] = xTr
    
    # Do the forward process here
    for i in range(len(W)-1, 0-1, -1):
        aas[i-len(W)-1] = np.dot(W[i-len(W)], zzs[i-len(W)])
        zzs[i-len(W)-1] = np.vstack((trans_func(aas[i-len(W)-1]), np.ones([1, n])))
        
    # INSERT CODE: (last one is special, no transition function)
    zzs[i-len(W)-1] = aas[i-len(W)-1]
    return aas, zzs