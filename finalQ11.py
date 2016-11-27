# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 15:20:14 2016

@author: philip.ball
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def train_SVM_hard(X,Y,deg):
    svc = SVC(kernel = 'poly', C = float('inf'), degree = deg, shrinking = False, gamma = 1.0, coef0 = 1)
    y_poly = svc.fit(X,Y)
    return(y_poly)

def train_SVM_hard_RBF(X,Y):
    svc = SVC(kernel = 'rbf', C = float('inf'), gamma = 1.0, shrinking = False)
    y_RBF = svc.fit(X,Y)
    return(y_RBF)
    
def transformZ(X):
    Z1 = X[:,1]**2 - 2*X[:,0] - 1
    Z2 = X[:,0]**2 - 2*X[:,1] + 1
    return(np.column_stack((Z1,Z2)))

def getClassLine(X,model):
    X1 = np.linspace(X[:,0].min(),X[:,0].max())
    X2 = np.linspace(X[:,1].min(),X[:,1].max())
    xx,yy = np.meshgrid(X1,X2)
    Z = model.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)    
    return(xx,yy,Z)
    
X = np.array([[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]])
Y = np.array([-1,-1,-1,1,1,1,1])

Z = transformZ(X)

# train the polynomial model in the X space for Q12
modelX = train_SVM_hard(X,Y,2)
print("SVs for X space: " + str(modelX.n_support_.sum()))

# train also a monomial model in the Z space for fun
modelZ = train_SVM_hard(Z,Y,1)
print("SVs for Z space linear: " + str(modelZ.n_support_.sum()))

# train also a monomial model in the Z space for fun
modelR = train_SVM_hard_RBF(X,Y)
print("SVs for X space RBF: " + str(modelR.n_support_.sum()))

# plot the Z space model
print("Transformed into Z space:")
xx,yy,Zs = getClassLine(Z,modelZ)
plt.contourf(xx,yy,Zs,cmap = plt.cm.coolwarm, alpha = 0.8)
plt.scatter(Z[Y==-1][:,0],Z[Y==-1][:,1],c = 'red')
plt.scatter(Z[Y==1][:,0],Z[Y==1][:,1],c = 'green')
plt.show()

# plot the X space model, to verify SVs
print("\nNontransformed (in X space):")
xx,yy,Zs = getClassLine(X,modelX)
plt.contourf(xx,yy,Zs,cmap = plt.cm.coolwarm, alpha = 0.8)
plt.scatter(X[Y==-1][:,0],X[Y==-1][:,1],c = 'red')
plt.scatter(X[Y==1][:,0],X[Y==1][:,1],c = 'green')
plt.show()

# plot the X space RBF model for fun
print("\nNontransformed (in X space):")
xx,yy,Zs = getClassLine(X,modelR)
plt.contourf(xx,yy,Zs,cmap = plt.cm.coolwarm, alpha = 0.8)
plt.scatter(X[Y==-1][:,0],X[Y==-1][:,1],c = 'red')
plt.scatter(X[Y==1][:,0],X[Y==1][:,1],c = 'green')
plt.show()