"""
INPUT:  
xTr : dxn input vectors
yTr : 1xn input labels
ktype : (linear, rbf, polynomial)
Cs   : interval of regularization constant that should be tried out
paras: interval of kernel parameters that should be tried out

Output:
bestC: best performing constant C
bestP: best performing kernel parameter
lowest_error: best performing validation error
errors: a matrix where allvalerrs(i,j) is the validation error with parameters Cs(i) and paras(j)

Trains an SVM classifier for all combination of Cs and paras passed and identifies the best setting.
This can be implemented in many ways and will not be tested by the autograder. You should use this
to choose good parameters for the autograder performance test on test data. 
"""

import numpy as np
from numpy import mat
import math
from trainsvm import trainsvm
import pandas as pd
import os
from sklearn import metrics
from scipy.stats import zscore
from sklearn.model_selection import KFold


def crossvalidate(xTr, yTr, ktype, Cs, paras):
    bestC, bestP, lowest_error = 0, 0, 0
    df_xTr = mat(xTr.transpose())
    df_yTr = mat(yTr)

    errors = np.zeros((len(paras),len(Cs)))

    
    kf = KFold(10)
    
    fold = 0

    for i in range(len(Cs)):
        for j in range(len(paras)):
            errors[i,j]=0

            for train, test in kf.split(df_xTr):
                fold+=1
                    
                x_train = df_xTr[train]
                y_train = df_yTr[train]
                x_test = df_xTr[test]
                y_test = df_yTr[test]

                x_train = x_train.getA()
                y_train = y_train.getA()
                x_test = x_test.getA()
                y_test = y_test.getA()
                svmclassify = trainsvm(x_train.T,y_train, Cs[i],ktype,paras[j])
                train_preds = svmclassify(x_test.T)
                errors[i,j] += np.sum(train_preds != y_test)
            errors[i,j]=errors[i,j]/xTr.shape[1]

    minerror=np.min(errors)
    print(minerror)
    index = np.where(errors == minerror)
    bestC=Cs[index[0][3]]
    bestP=paras[index[1][3]]
    lowest_error=minerror
    
    return bestC, bestP, lowest_error, errors
    