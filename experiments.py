#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 23:12:37 2022

@author: JoeyZhou
"""
import numpy as np
from algorithms import maxentropy
from algorithms import maxerrorreduction
from algorithms import minlossincrease
from algorithms import maxmodelchange
from algorithms import varianceReduction
from algorithms import uniformrandom
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from dataprocessing import importDataSet


def experiments(name, maxiter, initsize, k, lammy):
    
    X, y = importDataSet(name, "standardization")
    kf = KFold(n_splits=k, random_state=None)
    performance = np.zeros([4, k, maxiter+1])
    foldindex = 0
    
    for train_index , test_index in kf.split(X):
        
        X_train , X_test = X[train_index,:], X[test_index,:]
        y_train , y_test = y[train_index], y[test_index]
        
        L, Ly = X_train[:initsize,:], y_train[:initsize]
        U, Uy = X_train[initsize:,:], y_train[initsize:]
        
        performance[0, foldindex, :] = maxentropy(L, Ly, U, Uy, lammy, maxiter, X_test, y_test)
        print("Max entropy finished!")
        performance[1, foldindex, :] = maxerrorreduction(L, Ly, U, Uy, lammy, maxiter, X_test, y_test)
        print("Max error reduction finished!")
        performance[2, foldindex, :] = minlossincrease(L, Ly, U, Uy, lammy, maxiter, X_test, y_test)
        print("Min loss increase finished!")
        performance[3, foldindex, :] = maxmodelchange(L, Ly, U, Uy, lammy, maxiter, X_test, y_test)
        print("Max model change finished!")
        
        foldindex += 1
        
    return np.sum(performance, axis=1)/k

def experimentsbatch(name, num, initsize, k):
    
    X, y = importDataSet(name, "standardization")
    kf = KFold(n_splits=k, shuffle=True)
    performance = np.zeros([2, k, 2])
    foldindex = 0
    
    for train_index , test_index in kf.split(X):
        
        X_train , X_test = X[train_index,:], X[test_index,:]
        y_train , y_test = y[train_index], y[test_index]
        
        L, Ly = X_train[:initsize,:], y_train[:initsize]
        U, Uy = X_train, y_train
        
        performance[0, foldindex, :] = varianceReduction(L, Ly, U, Uy, num, U, Uy)
        performance[1, foldindex, :] = uniformrandom(L, Ly, U, Uy, num, U, Uy)
        
        foldindex += 1
        
    return np.sum(performance, axis=1)/k
    

def graphplotter(performance):
    n = performance.shape[1]
    x = np.array(range(n))
    plt.plot(x, performance[0,:], label="Max entropy")
    plt.plot(x, performance[1,:], label="Max error reduction")
    plt.plot(x, performance[2,:], label="Min loss increase")
    plt.plot(x, performance[3,:], label="Max model change")
    plt.legend()
    plt.show()
    

    
performance = experimentsbatch("australian.dat", 600, 30, 10)
print(performance)
