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

# note that for RBF, the deg value is ignored
def SVM_RBF(x_train,y_train,Deg,Reg):
    svc = SVC(kernel = 'rbf', C = Reg, gamma = 1.0, shrinking = False)
    y_RBF = svc.fit(x_train,y_train)
    return(y_RBF)
    
def oneVall(num_class,SV_trainer,x_train,y_train,Deg,Reg):
    entries = (y_train[:] == num_class)
    y_oVa = np.ones(y_train.size)
    y_oVa[~entries] = -1
    fitted = SV_trainer(x_train,y_oVa,Deg,Reg)
    return fitted, y_oVa

def oneVone(num_class,num_not,SV_trainer,x_train,y_train,Deg,Reg):
    entries = (y_train[:] == num_class) | (y_train[:] == num_not)
    y_new = y_train[entries]
    x_new = x_train[entries]
    neg = (y_new[:] == num_not)
    y_oVo = np.ones(y_new.size)
    y_oVo[neg] = -1
    fitted = SV_trainer(x_new,y_oVo,Deg,Reg)
    return(fitted, y_oVo,x_new)

def EinCalc(x_train,y_train,SV_trainer,Deg,Reg,num_class,num_not = False):
    if num_not == False:
        fitted,y_act = oneVall(num_class,SV_trainer,x_train,y_train,Deg,Reg)
        x_train = x_train
    else:
        fitted,y_act,x_new = oneVone(num_class,num_not,SV_trainer,x_train,y_train,Deg,Reg)
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
    group = 156 # for 1 v 5, there are 1561 data points, so we group into 156 (and 157 for the last)
    E_out = np.zeros(10)    # this is a from 10, leave one out algorithm, so we store 10 errors
    for i in range(0,10):   
        start = i*group     # starting index
        end = start + group     # ending index 
        if i == 9:
            end = end + 1   # as the last group will have 1 extra element
        x_l_out = x_train[start:end,:]  # select the leave out elements
        y_l_out = y_train[start:end]    # same, but for y
        x_keep = np.vstack((x_train[:start,:],x_train[end:,:])) # select the remained elements
        y_keep = np.hstack((y_train[:start],y_train[end:])) # same, but for y
        model = SVM_Poly(x_keep,y_keep,Deg,Reg) # tran the model on the remained elements
        y_l_hyp = model.predict(x_l_out)    # predict the left out y vals
        E_out[i] = (y_l_hyp != y_l_out).mean()  # return the error of that prediction
    E_CV = E_out.mean() # find the CV error by taking the mean of the stored errors
    return(E_CV)

def CrossVal_exp(runs,x_train,y_train,Deg):
    C_vals = [0.001]  # list containing all the different C vals we wish to test
    C_choice = np.zeros(runs)   # vector containing the selected C vals for each run
    CV_vals = np.zeros(runs)    # vector containing the errors vals of the selected C vals
    entries = (y_train == 1) | (y_train == 5)   # boolean vector saying if 1 or 5 is in that entry
    y_new = y_train[entries]    # select only the 1 or 5 y vals
    x_new = x_train[entries]    # same for x vals
    y_class = np.ones(entries.sum())    # create the classification vector (1 vs -1)
    y_class[y_new == 5] = -1    # set all 5 vals to -1
    comb = np.column_stack((y_class,x_new)) # create a combined y and x matrix for shuffling
    for i in range(0,runs):
        np.random.shuffle(comb) # shuffle the order of rows
        y_new_s = comb[:,0] # re-extract the y values
        x_new_s = comb[:,1:]    # same for x values
        CV_min = 1  # now set a dummy CV error (the error should always be smaller than 1)
        for C in C_vals:
            E_CV = CrossVal_1v5(x_new_s,y_new_s,Deg,C)  # run CV on the input value of C
            if CV_min > E_CV:   # if the error is less than the smallest error so far, then
                CV_min = E_CV   # set the new min value
                CV_vals[i] = CV_min  # store the min error for the optimal C
                C_choice[i] = C     # select this value of C as being optimal for this run
    C_com,numb = mode(C_choice)     # take the mode of the C choices vector as being the most selected C
    CV_val_ave = CV_vals.mean()
    return(C_com, CV_val_ave)   # return the most selected C value
    
feat_test = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW8Q2 Data\\features.test.txt')
feat_train = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW8Q2 Data\\features.train.txt')

x_test = feat_test[:,1:]
y_test = feat_test[:,0]
x_train = feat_train[:,1:]
y_train = feat_train[:,0]

#dig = [0,2,4,6,8]
#dig = [1,3,5,7,9]

Deg = 2
#Reg = 1
#Reg = [0.001,0.01,0.1,1]
Reg = [0.01,1,100,10**4,10**6]

#for num in dig:
#    Ein,SVs,model = EinCalc(x_train, y_train, SVM_Poly, Deg, Reg, num)
#    print('error of '+ str(num) + ' vs all: ' + str(Ein))
#    print('SVs of '+ str(num) + ' vs all: ' + str(SVs))
#
#Ein,SVs,model = EinCalc(x_train,y_train, SVM_Poly,Deg,Reg,1,5)
#Eout = EoutCalc(x_test,y_test,model,1,5)
#print(Ein)
#print(SVs)
#print(Eout)

for C in Reg:
    Ein,SVs,model = EinCalc(x_train,y_train,SVM_RBF,Deg,C,1,5)
    Eout = EoutCalc(x_test,y_test,model,1,5)
    print('in sample error of C = '+ str(C) + ': ' + str(Ein))
    print('out of sample error of C = '+ str(C) + ': ' + str(Eout))
    print('SVs of C = '+ str(C) + ': ' + str(SVs))

#print(CrossVal_exp(100,x_train,y_train,Deg))