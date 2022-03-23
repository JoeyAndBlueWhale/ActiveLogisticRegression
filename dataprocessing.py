#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 22:48:35 2022

@author: JoeyZhou
"""
import numpy as np
import uci_dataset as dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def importDataSet(name,method):
    '''
    X = np.genfromtxt(r'./dataset/'+name)
    y = X[:,-1]
    X = X[:,0:-1]
    y = (y == 1) * 2 - 1
    #y = y-3
    '''
    X = dataset.load_chess().to_numpy()
    y = X[:,-1]
    y = (y == 'won') * 2 - 1
    X = X[:,0:-1]
    X_1 = np.delete(X, [12, 14, -1], axis=1)
    X_1 = (X_1 == 't').astype(int)
    
    le = LabelEncoder()
    encoded = le.fit_transform(X[:,12])
    
    X_1 = np.append(X_1, encoded.reshape(-1,1), axis=1)
    
    encoded = le.fit_transform(X[:,-1])
    
    X_1 = np.append(X_1, encoded.reshape(-1,1), axis=1)
    
    encoded = le.fit_transform(X[:,14])
    encoded = encoded.reshape(len(encoded), -1)
    one = OneHotEncoder(sparse=False)
    
    X_1 = np.append(X_1, one.fit_transform(encoded), axis=1)
    
    X = X_1
    
    
    X -= np.mean(X, axis=0)
    if method == "standardization":
        X /= np.std(X, axis=0)
    else:
        mini = np.min(X, axis=0)
        maxi = np.max(X, axis=0)
        X /= maxi - mini
    return X, y

         

