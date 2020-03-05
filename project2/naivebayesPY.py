#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
"""

import numpy as np

def naivebayesPY(x, y):
    # function [pos,neg] = naivebayesPY(x,y);
    #
    # Computation of P(Y)
    # Input:
    # x : n input vectors of d dimensions (dxn)
    # y : n labels (-1 or +1) (1xn)
    #
    # Output:
    # pos: probability p(y=1)
    # neg: probability p(y=-1)
    #
    
    # Convertng input matrix x and y into NumPy matrix
    # input x and y should be in the form: 'a b c d...; e f g h...; i j k l...'
    X = np.matrix(x) #128*1200
    Y = np.matrix(y) #1*1200
    
    # Pre-configuring the size of matrix X
    d,n = X.shape
    
    # Pre-constructing a matrix of all-ones (dx2)
    X0 = np.ones((d,2)) #128*2
    Y0 = np.matrix('-1, 1') #1*2
    
    # add one all-ones positive and negative example
    Xnew = np.hstack((X, X0)) #stack arrays in sequence horizontally (column-wise) #128*1202
    Ynew = np.hstack((Y, Y0)) #1*1202

    # Re-configuring the size of matrix Xnew
    d,n = Xnew.shape
    
    ## fill in code here
    Ynew=Ynew+1
    Ynew=Ynew/2.0
    pos=np.mean(Ynew)
    neg=1-pos
    return pos,neg