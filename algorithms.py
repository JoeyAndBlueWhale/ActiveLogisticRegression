#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 22:48:57 2022

@author: JoeyZhou
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from dataprocessing import importDataSet
import cvxpy as cp
import mosek
import random
from numpy.linalg import norm 

def maxentropy(L, Ly, U, Uy, lammy, maxiter, X_test, y_test):
    classifier = LogisticRegression(C=1/lammy)
    classifier.fit(L, Ly)
    performance = np.zeros(maxiter+1)
    pred = classifier.predict(X_test)
    performance[0] = np.sum(pred==y_test)/len(y_test)
    
    for i in range(maxiter):
        prob = classifier.predict_proba(U)
        prob = prob[:,0]
        prob2 = 1-prob
        entropy = np.multiply(prob2, np.log(prob2))+np.multiply(prob, np.log(prob))
        entropy = -1*entropy
        index = np.argmax(entropy)
        newdata = U[index,:]
        newlabel = Uy[index]
        L = np.insert(L, 0, newdata, axis=0)
        Ly = np.insert(Ly, 0, newlabel)
        U = np.delete(U, index, axis=0)
        Uy = np.delete(Uy, index)
        classifier.fit(L, Ly)
        
        pred = classifier.predict(X_test)
        performance[i+1] = np.sum(pred==y_test)/len(y_test)
    
    return performance

def maxerrorreduction(L, Ly, U, Uy, lammy, maxiter, X_test, y_test):
    classifier = LogisticRegression(C=1/lammy)
    classifier.fit(L, Ly)
    performance = np.zeros(maxiter+1)
    pred = classifier.predict(X_test)
    performance[0] = np.sum(pred==y_test)/len(y_test)
    print("iteration: 0")
    print("accuracy: ", performance[0])
    
    for j in range(maxiter):
        errorreduction = np.zeros(len(U))
        for i in range(len(U)):
            Lplus = np.insert(L, 0, U[i], axis=0)
            Lyplus1 = np.insert(Ly, 0, 1)
            Lyplus2 = np.insert(Ly, 0, -1)
            
            classifier.fit(Lplus, Lyplus1)
            prob = classifier.predict_proba(U)
            prob = prob[:,0]
            prob2 = 1-prob
            entropy1 = np.multiply(prob2, np.log(prob2))+np.multiply(prob, np.log(prob))
            entropy1 = -1*entropy1
            entropy1 = np.sum(entropy1)
            
            classifier.fit(Lplus, Lyplus2)
            prob = classifier.predict_proba(U)
            prob = prob[:,0]
            prob2 = 1-prob
            entropy2 = np.multiply(prob2, np.log(prob2))+np.multiply(prob, np.log(prob))
            entropy2 = -1*entropy2
            entropy2 = np.sum(entropy2)
            
            errorreduction[i] = min(entropy1, entropy2)
        
            
        index = np.argmin(errorreduction)
        newdata = U[index,:]
        newlabel = Uy[index]
        L = np.insert(L, 0, newdata, axis=0)
        Ly = np.insert(Ly, 0, newlabel)
        U = np.delete(U, index, axis=0)
        Uy = np.delete(Uy, index)
        classifier.fit(L, Ly)
        
        pred = classifier.predict(X_test)
        performance[j+1] = np.sum(pred==y_test)/len(y_test)
        print("iteration: ", j+1)
        print("accuracy: ", performance[j+1])
    
    return performance

def minlossincrease(L, Ly, U, Uy, lammy, maxiter, X_test, y_test):
    classifier = LogisticRegression(C=1/lammy)
    classifier.fit(L, Ly)
    performance = np.zeros(maxiter+1)
    pred = classifier.predict(X_test)
    performance[0] = np.sum(pred==y_test)/len(y_test)
    print("iteration: 0")
    print("accuracy: ", performance[0])
    
    for j in range(maxiter):
        lossincrease = np.zeros(len(U))
        for i in range(len(U)):
            Lplus = np.insert(L, 0, U[i], axis=0)
            Lyplus1 = np.insert(Ly, 0, 1)
            Lyplus2 = np.insert(Ly, 0, -1)
            
            classifier.fit(Lplus, Lyplus1)
            w = classifier.coef_.flatten()
            loss1 = np.log(1+np.exp(-1*np.multiply(np.dot(Lplus,w), Lyplus1)))
            loss1 = np.sum(loss1)
            loss1 = lammy/2*norm(w,2)**2+loss1
            
            
            classifier.fit(Lplus, Lyplus2)
            w = classifier.coef_.flatten()
            loss2 = np.log(1+np.exp(-1*np.multiply(np.dot(Lplus,w), Lyplus2)))
            loss2 = np.sum(loss2)
            loss2 = lammy/2*norm(w,2)**2+loss2
            
            lossincrease[i] = max(loss1, loss2)
        
            
        index = np.argmin(lossincrease)
        newdata = U[index,:]
        newlabel = Uy[index]
        L = np.insert(L, 0, newdata, axis=0)
        Ly = np.insert(Ly, 0, newlabel)
        U = np.delete(U, index, axis=0)
        Uy = np.delete(Uy, index)
        classifier.fit(L, Ly)
        
        pred = classifier.predict(X_test)
        performance[j+1] = np.sum(pred==y_test)/len(y_test)
        print("iteration: ", j+1)
        print("accuracy: ", performance[j+1])
    
    return performance

def maxmodelchange(L, Ly, U, Uy, lammy, maxiter, X_test, y_test):
    classifier = LogisticRegression(C=1/lammy)
    classifier.fit(L, Ly)
    performance = np.zeros(maxiter+1)
    pred = classifier.predict(X_test)
    performance[0] = np.sum(pred==y_test)/len(y_test)
    
    for i in range(maxiter):
        w = classifier.coef_.flatten()
        innerprod = np.dot(U, w)
        change = np.multiply(1/(1+np.exp(-1*innerprod)), 1/(1+np.exp(innerprod)))
        change = 2*np.multiply(change, norm(U, axis=1))
        
            
        index = np.argmax(change)
        newdata = U[index,:]
        newlabel = Uy[index]
        L = np.insert(L, 0, newdata, axis=0)
        Ly = np.insert(Ly, 0, newlabel)
        U = np.delete(U, index, axis=0)
        Uy = np.delete(Uy, index)
        classifier.fit(L, Ly)
        
        pred = classifier.predict(X_test)
        performance[i+1] = np.sum(pred==y_test)/len(y_test)
    
    return performance


def sdpsolver(U,w,num):
    print("Number of samples: ",num)
    n = len(U)
    print(n)
    m = U.shape[1]
    a = cp.Variable(n)
    S = np.zeros([m,m])
    S = cp.reshape(S,(m,m))
    
    for i in range(n):
        data = U[i,:]
        inner = w @ data
        matrix = cp.kron(cp.reshape(data,(m,1)), cp.reshape(data,(1,m)))
        S += a[i]*(cp.exp(inner)/((1+cp.exp(inner))**2))*matrix
        
    cost = 0
    
    for i in range(n):
        data = U[i,:]
        inner = w @ data
        cost += (cp.exp(inner)/((1+cp.exp(inner))**2))*cp.matrix_frac(data,S)
        
    constraints = [0 <= a, a <= 1, cp.sum(a) == num]
    prob = cp.Problem(cp.Minimize(cost),constraints)
    #prob.solve(solver="MOSEK", verbose=True, mosek_params = {mosek.dparam.intpnt_co_tol_near_rel:  1e+5})
    prob.solve(solver="CVXOPT")
    
    print(np.sum(a.value))
    print(prob.status)
    
    return a.value/n
    
def varianceReduction(L, Ly, U, Uy, num, X_test, y_test):
    classifier = LogisticRegression(penalty='none')
    classifier.fit(L, Ly)
    performance = np.zeros(2)
    pred = classifier.predict(X_test)
    performance[0] = np.sum(pred==y_test)/len(y_test)
    print("accuracy: ", performance[0])
    w = classifier.coef_.flatten()
    
    gamma = sdpsolver(U, w, num)
    uniform = np.ones(len(U))/len(U)
    alpha = 1 - np.power(float(num),-1/6)
    gamma = alpha*gamma+(1-alpha)*uniform
    
    
    indexlist = np.array(range(len(U)))
    
    index = random.choices(indexlist, weights=gamma, k=num)
    data = U[index,:]
    label = Uy[index]
    
    classifier.fit(data, label)
    
    pred = classifier.predict(X_test)
    performance[1] = np.sum(pred==y_test)/len(y_test)
    print("accuracy: ", performance[1])
    
    return performance

def uniformrandom(L, Ly, U, Uy, num, X_test, y_test):
    classifier = LogisticRegression(penalty='none')
    classifier.fit(L, Ly)
    performance = np.zeros(2)
    pred = classifier.predict(X_test)
    performance[0] = np.sum(pred==y_test)/len(y_test)
    print("accuracy: ", performance[0])  
    
    indexlist = np.array(range(len(U)))
    
    index = random.choices(indexlist, k=num)
    data = U[index,:]
    label = Uy[index]
    
    classifier.fit(data, label)
    
    pred = classifier.predict(X_test)
    performance[1] = np.sum(pred==y_test)/len(y_test)
    print("accuracy: ", performance[1])
    
    return performance

'''
m = 100
n = 15
U = np.random.rand(m,n)
w = np.random.rand(n)

sdpsolver(U,w,30)
'''
    
    
    
    
    



