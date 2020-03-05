#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
"""

import numpy as np

def naivebayesPXY(x, y):
# =============================================================================
#    function [posprob,negprob] = naivebayesPXY(x,y);
#
#    Computation of P(X|Y)
#    Input:
#    x : n input vectors of d dimensions (dxn)
#    y : n labels (-1 or +1) (1xn)
#    
#    Output:
#    posprob: probability vector of p(x|y=1) (dx1)
#    negprob: probability vector of p(x|y=-1) (dx1)
# =============================================================================


    
    # Convertng input matrix x and y into NumPy matrix
    # input x and y should be in the form: 'a b c d...; e f g h...; i j k l...'
    X = np.matrix(x) #128*1200
    Y = np.matrix(y) #1*1200
    
    # Pre-configuring the size of matrix X
    d,n = X.shape
    
    # Pre-constructing a matrix of all-ones (dx2)
    X0 = np.ones((d,2))
    Y0 = np.matrix('-1, 1')
    
    # add one all-ones positive and negative example
    Xnew = np.hstack((X, X0)) #stack arrays in sequence horizontally (column-wise) #128*1202
        #Xnew = np.concatenate((X, X0), axis=1) #concatenate to column 
    Ynew = np.hstack((Y, Y0)) #1*1202
    
    # Re-configuring the size of matrix Xnew
    d,n = Xnew.shape
    
# =============================================================================
# fill in code here

    Ynew=Ynew+1
    Ynew=Ynew/2.0
    
    numerator=np.dot(Xnew,Ynew.transpose())
    sum_features=np.zeros((1,n))
    for i in range(n):
        sum_features[0,i]=np.sum(Xnew[:,i])
    denominator=np.dot(sum_features,Ynew.transpose())
    posprob=numerator/denominator

    Ynew2=1-Ynew
    numerator=np.dot(Xnew,Ynew2.transpose())
    denominator=np.dot(sum_features,Ynew2.transpose())
    negprob=numerator/denominator

    return posprob,negprob

# =============================================================================
