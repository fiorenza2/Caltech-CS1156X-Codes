# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 23:47:52 2016

@author: philip.ball
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def SVM_Poly(x_train,y_train,Deg,Reg):
    svc = SVC(kernel = 'poly', C = Reg, degree = Deg, shrinking = False)
    y_poly = svc.fit(x_train,y_train)
    return(y_poly)
    
def oneVall(num_class,x_train,y_train,Deg,Reg):
    entries = (y_train[:] == num_class)
    y_oVa = np.ones(y_train.size)
    y_oVa[~entries] = -1
    fitted = SVM_Poly(x_train,y_oVa,Deg,Reg)
    return fitted, y_oVa

def oneVone(num_class,num_not,x_train,y_train,Deg,Reg):
    entries = (y_train[:] == num_class) | (y_train[:] == num_not)
    y_new = y_train[entries]
    x_new = x_train[entries]
    neg = (y_new[:] == num_not)
    y_oVo = np.ones(y_new.size)
    y_oVo[neg] = -1
    fitted = SVM_Poly(x_new,y_oVo,Deg,Reg)
    return(fitted, y_oVo)

def EinCalc(x_train,y_train,Deg,Reg,num_class,num_not = False):
    if num_not == False:
        fitted,y_act = oneVall(num_class,x_train,y_train,Deg,Reg)
    else:
        fitted,y_act = oneVone(num_class,num_not,x_train,y_train,Deg,Reg)
    y_hyp = fitted.predict(x_train)
    Ein = (y_act != y_hyp)
    No_SV = fitted.n_support_
    return(Ein.mean(),No_SV)
    
feat_test = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW8Q2 Data\\features.test.txt')
feat_train = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW8Q2 Data\\features.train.txt')

x_test = feat_test[:,1:]
y_test = feat_test[:,0]
x_train = feat_train[:,1:]
y_train = feat_train[:,0]

#dig = [0,2,4,6,8]
dig = [1,3,5,7,9]

Deg = 2
Reg = 0.01

for num in dig:
    Ein,SVs = EinCalc(num_class = num, x_train = x_train, y_train = y_train, Deg = Deg, Reg = Reg)
    print('error of '+ str(num) + ' vs all: ' + str(Ein))
    print('SVs of '+ str(num) + ' vs all: ' + str(SVs))

#EinCalc(x_train,y_train,Deg,Reg,0,3)