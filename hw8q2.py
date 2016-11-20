# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 23:47:52 2016

@author: philip.ball
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from scipy.stats import mode

def SVM_Poly(x_train,y_train,Deg,Reg):
    svc = SVC(kernel = 'poly', C = Reg, degree = Deg, shrinking = False, gamma = 1.0, coef0 = 1)
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
    return(fitted, y_oVo,x_new)

def EinCalc(x_train,y_train,Deg,Reg,num_class,num_not = False):
    if num_not == False:
        fitted,y_act = oneVall(num_class,x_train,y_train,Deg,Reg)
        x_train = x_train
    else:
        fitted,y_act,x_new = oneVone(num_class,num_not,x_train,y_train,Deg,Reg)
        x_train = x_new
    y_hyp = fitted.predict(x_train)
    Ein = (y_act != y_hyp)
    No_SV = fitted.n_support_.sum()
    return(Ein.mean(),No_SV,fitted)

def EoutCalc(x_test, y_test, model, num_class, num_not = False):
    if num_not == False:
        y_act = np.ones(y_test.size)
        y_act[y_test != num_class] = -1
    else:
        entries = (y_test == num_class) | (y_test == num_not)
        y_new = y_test[entries]
        y_act = np.ones(y_new.size)
        y_act[y_new == num_not] = -1
        x_new = x_test[entries]
    y_hyp = model.predict(x_new)
    Eout = (y_act != y_hyp)
    return(Eout.mean())

# the below assumes you have already turned your outputs into 1/-1
def CrossVal_1v5(x_train,y_train,Deg,Reg):
    group = 156
    E_out = np.zeros(10)
    for i in range(0,10):
        start = i*group
        end = start + group
        if i == 9:  # as the last group will have 1 extra element
            end = end + 1
        x_l_out = x_train[start:end,:]
        y_l_out = y_train[start:end]
        x_keep = np.vstack((x_train[:start,:],x_train[end:,:]))
        y_keep = np.hstack((y_train[:start],y_train[end:]))
        model = SVM_Poly(x_keep,y_keep,Deg,Reg)
        y_l_hyp = model.predict(x_l_out)
        E_out[i] = (y_l_hyp != y_l_out).mean()
    E_CV = E_out.mean()
    return(E_CV)

def CrossVal_exp(runs,x_train,y_train,Deg):
    C_vals = [0.0001,0.001,0.01,0.1,1]
    C_choice = np.zeros(runs)
    entries = (y_train == 1) | (y_train == 5)
    y_new = y_train[entries]
    x_new = x_train[entries]
    y_class = np.ones(entries.sum())
    y_class[y_new == 5] = -1
    comb = np.column_stack((y_class,x_new))
    for i in range(0,runs):
        np.random.shuffle(comb)
        y_new_s = comb[:,0]
        x_new_s = comb[:,1:]
        CV_min = 1
        for C in C_vals:
            E_CV = CrossVal_1v5(x_new_s,y_new_s,Deg,C)
            if CV_min > E_CV:
                C_choice[i] = C
    C_com,numb = mode(C_choice)
    return(C_com)
            
        
    
feat_test = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW8Q2 Data\\features.test.txt')
feat_train = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW8Q2 Data\\features.train.txt')

x_test = feat_test[:,1:]
y_test = feat_test[:,0]
x_train = feat_train[:,1:]
y_train = feat_train[:,0]

#dig = [0,2,4,6,8]
#dig = [1,3,5,7,9]

Deg = 2
Reg = 1
#Reg = [0.001,0.01,0.1,1]

#for num in dig:
#    Ein,SVs,model = EinCalc(x_train, y_train, Deg, Reg, num)
#    print('error of '+ str(num) + ' vs all: ' + str(Ein))
#    print('SVs of '+ str(num) + ' vs all: ' + str(SVs))
#
#Ein,SVs,model = EinCalc(x_train,y_train,Deg,Reg,1,5)
#Eout = EoutCalc(x_test,y_test,model,1,5)
#print(Ein)
#print(SVs)
#print(Eout)

#for C in Reg:
#    Ein,SVs,model = EinCalc(x_train,y_train,Deg,C,1,5)
#    Eout = EoutCalc(x_test,y_test,model,1,5)
#    print('in sample error of C = '+ str(C) + ': ' + str(Ein))
#    print('out of sample error of C = '+ str(C) + ': ' + str(Eout))
#    print('SVs of C = '+ str(C) + ': ' + str(SVs))

print(CrossVal_exp(10,x_train,y_train,Deg))