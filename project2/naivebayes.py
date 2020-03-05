#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
"""

import numpy as np
from naivebayesPY import naivebayesPY
from naivebayesPXY import naivebayesPXY

def naivebayes(x, y, x1):
# =============================================================================
#function logratio = naivebayes(x,y,x1);
#
#Computation of log P(Y|X=x1) using Bayes Rule
#Input:
#x : n input vectors of d dimensions (dxn)
#y : n labels (-1 or +1)
#x1: input vector of d dimensions (dx1)
#
#Output:
#logratio: log (P(Y = 1|X=x1)/P(Y=-1|X=x1))
# =============================================================================


    
    # Convertng input matrix x and x1 into NumPy matrix
    # input x and y should be in the form: 'a b c d...; e f g h...; i j k l...'
    
    
# =============================================================================
# fill in code here
    X = np.matrix(x)
    X1= np.matrix(x1)
    d,n=X.shape
    posprob,negprob=naivebayesPXY(x,y)
    pos,neg=naivebayesPY(x,y)
    logratio = (x1.T).dot(np.log(posprob))+np.log(pos)-(x1.T).dot(negprob)-np.log(neg)

    return logratio
# =============================================================================
