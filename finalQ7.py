# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 23:25:03 2016

@author: philip.ball
"""

import numpy as np
import matplotlib.pyplot as plt

def w_reg(Z,y,lamb):
    n,m = Z.shape
    Z_tr = np.transpose(Z)
    bracket = np.dot(Z_tr,Z) + lamb * np.identity(m)
    pre = np.dot(np.linalg.inv(bracket),Z_tr)
    w = np.dot(pre,y)
    return(w)
    
def lin_OvA(Z_in,y_in,lamb,num):
    ind = y_in == num   # vector which tells you which rows have the required number
    bias = np.ones(y_in.size)
    y_new = np.ones(y_in.size)  # instantiate the classification vector
    y_new[~ind] = -1    # set the appropriate values to -1 (ie: the rows where the required number didn't appear)
    Z_in = np.column_stack((bias,Z_in))  # add the bias terms
    w = w_reg(Z_in,y_new,lamb)
    y_hyp = np.sign(np.dot(w,np.transpose(Z_in)))
    bin_err = y_hyp != y_new
    #plot_OvA(Z_in,y_new,w)
    all0_ans = ind.sum()/ind.size
    return(bin_err.mean(),all0_ans)

def plot_OvA(Z_in,y_in,w):
    x = np.linspace(Z_in[:,1].min(),Z_in[:,1].max())
    y = -w[0]/w[2] - w[1]/w[2] * x
    plt.plot(x,y)
    ind_pos = (y_in == 1)
    plt.scatter(Z_in[~ind_pos][:,1],Z_in[~ind_pos][:,2],c = "green")
    plt.scatter(Z_in[ind_pos][:,1],Z_in[ind_pos][:,2],c = "red")
    plt.show()
        
# the below code was used to verify that my code was working as expected
#y_test = np.array([1,1,1,1,-1,-1,-1,-1])
#Z_test = np.array([1,2,1,2,5,6,5,6])
#Z_test = np.column_stack((np.ones(y_test.size),Z_test))
#
#w = w_reg(Z_test,y_test,0)
#y_hyp = np.sign(np.dot(w,np.transpose(Z_test)))
#print((y_hyp != y_test).mean())
#
#x = np.linspace(0,7)
#x_b = np.column_stack((np.ones(x.size),x))
#y = np.dot(w,np.transpose(x_b))
#y_dummy = np.linspace(-2,2)
#x_dummy = np.ones(y_dummy.size)
#
#x_sep = -w[0]/w[1]
#
#x_dummy = x_sep * x_dummy
#
#plt.scatter(Z_test[:,1],y_test)
#plt.plot(x,y)
#plt.plot(x_dummy,y_dummy)

# more test code below

#feat_test = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW8Q2 Data\\features.test.txt')
#feat_train = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW8Q2 Data\\features.train.txt')
#
#y_in = feat_train[:,0]
#Z_in = feat_train[:,1:]
#Z_in[:,1] = Z_in[:,1]**2
#
#nums = [0,1,2,3,4]
#lamb = 0.5
#
#for n in nums:
#    err,test = lin_OvA(Z_in,y_in,lamb,n)
#    print("In sample error for " + str(n)+ ":" + str(err))
#    print("All -1 guess: "+  str(test))


#here begins the code for the finals question

feat_test = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW8Q2 Data\\features.test.txt')
feat_train = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW8Q2 Data\\features.train.txt')

y_in = feat_test[:,0]
Z_in = feat_test[:,1:]

nums = [5,6,7,8,9]
lamb = 1

for n in nums:
    err,test = lin_OvA(Z_in,y_in,lamb,n)
    print("In sample error for " + str(n)+ ":" + str(err))

