#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 22:48:35 2022

@author: JoeyZhou
"""
import numpy as np

def importDataSet(name,method):
    X = np.genfromtxt(r'./dataset/'+name)
    y = X[:,-1]
    X = X[:,0:-2]
    y = (y == 1) * 2 - 1
    #y = y-3
    X -= np.mean(X, axis=0)
    if method == "standardization":
        X /= np.std(X, axis=0)
    else:
        mini = np.min(X, axis=0)
        maxi = np.max(X, axis=0)
        X /= maxi - mini
    return X, y


         

