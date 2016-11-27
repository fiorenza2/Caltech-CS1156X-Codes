# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 19:32:44 2016

@author: philip.ball
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def create_Points(n):
    X1 = np.random.uniform(-1,1,n)
    X2 = np.random.uniform(-1,1,n)
    Y = classify(X1,X2)
    X = np.column_stack((X1,X2))
    return(X,Y)
    
def classify(X1,X2):
    c = X2 - X1 + 0.25*np.sin(np.pi * X1)
    return(np.sign(c))

# where the h denotes hard margin
def train_SVM_hRBF(X,Y,g):
    svc = SVC(kernel = 'rbf', C = float('inf'), gamma = g, shrinking = False)
    model_RBF = svc.fit(X,Y)
    return(model_RBF)

def test_SVM(X,Y,model):
    Y_hyp = model.predict(X)
    return((Y_hyp != Y).mean())

def getClassLine(X,model):
    X1 = np.linspace(X[:,0].min(),X[:,0].max())
    X2 = np.linspace(X[:,1].min(),X[:,1].max())
    xx,yy = np.meshgrid(X1,X2)
    Z = model.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)    
    return(xx,yy,Z)
    
its = 1

sep_yn = np.zeros(its)

for i in range(0,its):
    X,Y = create_Points(100)
    model = train_SVM_hRBF(X,Y,1.5)
    E_in = test_SVM(X,Y,model)
    if E_in == 0:
        sep_yn[i] = 0
    else:
        sep_yn[i] = 1

X1 = X[:,0]
X2 = X[:,1]
xx,yy,Z = getClassLine(X,model)
plt.contourf(xx,yy,Z,cmap = plt.cm.coolwarm, alpha = 0.8)
plt.scatter(X1[Y == 1],X2[Y == 1], c = 'green')
plt.scatter(X1[Y == -1], X2[Y == -1], c = 'red')
plt.show()

print(sep_yn.mean())
    