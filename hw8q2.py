# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 23:47:52 2016

@author: philip.ball
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

def SVM_Poly(x_test,y_test,Deg,Reg):
    svr = SVR(kernel = 'poly', C = Reg, degree = Deg)
    y_poly = svr.fit(x_test,y_test)
    return(y_poly)

def oneVall(num_class,x_test,y_test,Deg,Reg):
    entries = (y_test[:] == num_class)
    y_oVa = np.ones(y_test.size)
    y_oVa[~entries] = -1
    fitted = SVM_Poly(x_test,y_oVa,Deg,Reg)
    return fitted, y_oVa
   
def EinCalc(num_class,x_test,y_test,Deg,Reg):
    fitted,y_act = oneVall(num_class,x_test,y_test,Deg,Reg)
    y_hyp = np.sign(fitted.predict(x_test))
    Ein = (y_act != y_hyp)
    return(Ein.mean())
    
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
    print('error of '+ str(num) + ' vs all: ' + str(EinCalc(num,x_test,y_test,Deg,Reg)))
    