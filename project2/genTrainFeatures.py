import numpy as np
from name2features import name2features

def genTrainFeatures():
# =============================================================================
# function [x,y]=genTrainFeatures()
# 
# This function converts names into feature vectors and loads in the training data. 
# 
# 
# Output: 
# x: n feature vectors of dimensionality d [d,n]
# y: n labels (+1 = girl, -1 = boy)
# =============================================================================
    girls = []
    with open('girls.train', 'r') as f:
        girls_train = f.read().splitlines()
        num_girls = len(girls_train)
        for name in girls_train:
            feature_vector = name2features(name)
            girls.append(feature_vector)
    girls = np.array(girls) #600*128
    
    boys = []
    with open('boys.train', 'r') as f:
        boys_train = f.read().splitlines()
        num_boys = len(boys_train)
        for name in boys_train:
            feature_vector = name2features(name)
            boys.append(feature_vector)
    boys = np.array(boys) #600*128
    
    # subprocess.call('cat girls.train | python name2features.py > girls.csv ', shell=True)
    # subprocess.call('cat boys.train | python name2features.py > boys.csv ', shell=True)
    
    x = np.vstack((girls,boys)).T  #128*1200
    y = np.vstack((np.ones((num_girls,1)), -np.ones((num_boys,1)))).T #1*1200
        
    # shuffle data into random order
    ii = np.random.permutation(y.size)

    x = x[:,ii]
    y = y[:,ii]
    
    return x,y