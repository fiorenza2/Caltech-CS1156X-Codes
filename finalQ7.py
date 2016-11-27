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
    
def lin_OvA(Z_in,y_in,Z_out,y_out,lamb,num):
    ind = y_in == num   # vector which tells you which rows have the required number
    y_new = np.ones(y_in.size)  # instantiate the classification vector
    y_new[~ind] = -1    # set the appropriate values to -1 (ie: the rows where the required number didn't appear)
    w = w_reg(Z_in,y_new,lamb)
    y_hyp = np.sign(np.dot(w,np.transpose(Z_in)))
    bin_err_in = y_hyp != y_new
    y_new_o,y_hyp_o = lin_OoS(Z_out,y_out,num,w)
    bin_err_out = y_hyp_o != y_new_o
    #plot_OvA(Z_in,y_new,w)
    all0_ans = ind.sum()/ind.size
    return(bin_err_in.mean(),bin_err_out.mean(),all0_ans)

def lin_OvO(Z_in,y_in,Z_out,y_out,lamb,num1,num2):
    ind = (y_in == num1) | (y_in == num2)     # total indices where either num1 or num2 appear
    y_filt = y_in[ind]    # only the numbers we want filtered out
    Z_filt = Z_in[ind,:]    # only the rows we want filtered out
    y_new = np.ones(y_filt.size)
    ind2 = y_filt == num2
    y_new[ind2] = -1
    w = w_reg(Z_filt,y_new,lamb)
    y_hyp = np.sign(np.dot(w,np.transpose(Z_filt)))
    bin_err_in = y_new != y_hyp
    all0_ans = ind2.sum()/ind.sum()
    y_new_o,y_hyp_o = lin_OoS_OvO(Z_out,y_out,num1,num2,w)
    bin_err_out = y_hyp_o != y_new_o
    plot_OvA(Z_filt,y_new,w)
    return(bin_err_in.mean(),bin_err_out.mean(),all0_ans)
    
def lin_OoS(Z_out,y_out,num,w):
    ind = y_out == num
    y_new_o = np.ones(ind.size)
    y_new_o[~ind] = -1
    y_hyp_o = np.sign(np.dot(w,np.transpose(Z_out)))
    return(y_new_o,y_hyp_o)

def lin_OoS_OvO(Z_out,y_out,num1,num2,w):
    ind = (y_out == num1) | (y_out == num2)     # total indices where either num1 or num2 appear
    y_filt = y_out[ind]
    Z_filt = Z_out[ind,:]
    y_new_o = np.ones(y_filt.size)
    ind2 = y_filt == num2
    y_new_o[ind2] = -1
    y_hyp_o = np.sign(np.dot(w,np.transpose(Z_filt)))
    return(y_new_o,y_hyp_o)

def plot_OvA(Z_in,y_in,w):
    x = np.linspace(Z_in[:,1].min(),Z_in[:,1].max())
    y = -w[0]/w[2] - w[1]/w[2] * x
    plt.plot(x,y)
    ind_pos = (y_in == 1)
    plt.scatter(Z_in[~ind_pos][:,1],Z_in[~ind_pos][:,2],c = "green")
    plt.scatter(Z_in[ind_pos][:,1],Z_in[ind_pos][:,2],c = "red")
    plt.show()

# where n is the number of terms, order x0,x1,x2,x1x2,x1sq,x2sq
def transform_Z(Z,n):
    r,c = Z.shape
    Z0 = np.ones(r)
    Z_sq = Z**2
    Z_cr = Z[:,0]*Z[:,1]
    Z_trans = np.column_stack((Z0,Z,Z_cr,Z_sq))
    return(Z_trans[:,0:n])
    
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
#r,c = Z_in.shape
#bias = np.ones(r)
#Z_in = np.column_stack((bias,Z_in))
#
#nums = [0,1,2,3,4]
#lamb = 0.5
#
#for n in nums:
#    err,erroos,test = lin_OvA(Z_in,y_in,Z_in,y_in,lamb,n)
#    print("In sample error for " + str(n)+ ":" + str(err))
#    print("All -1 guess: "+  str(test))


#here begins the code for the finals question 7

#feat_test = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW8Q2 Data\\features.test.txt')
#feat_train = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW8Q2 Data\\features.train.txt')
#
#y_in = feat_train[:,0]
#Z_in = transform_Z(feat_train[:,1:],3)
#y_out = feat_test[:,0]
#Z_out = transform_Z(feat_test[:,1:],3)
#
#nums = [5,6,7,8,9]
#lamb = 1
#
#for n in nums:
#    err,erroos,test = lin_OvA(Z_in,y_in,Z_out,y_out,lamb,n)
#    print("In sample error for " + str(n)+ ":" + str(err))
#    print("All 0 error: " + str(test))

# finals question 8

#feat_test = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW8Q2 Data\\features.test.txt')
#feat_train = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW8Q2 Data\\features.train.txt')
#
#y_in = feat_train[:,0]
#Z_in = transform_Z(feat_train[:,1:],6)
#y_out = feat_test[:,0]
#Z_out = transform_Z(feat_test[:,1:],6)
#
#nums = [0,1,2,3,4]
#lamb = 1
#
#for n in nums:
#    err,erroos,test = lin_OvA(Z_in,y_in,Z_out,y_out,lamb,n)
#    print("Out of sample error for " + str(n)+ ":" + str(erroos))
#    #print("All 0 error: " + str(test))

# finals quetsion 9

#feat_test = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW8Q2 Data\\features.test.txt')
#feat_train = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW8Q2 Data\\features.train.txt')
#
#y_in = feat_train[:,0]
#Z_in_3 = transform_Z(feat_train[:,1:],3)
#Z_in_6 = transform_Z(feat_train[:,1:],6)
#y_out = feat_test[:,0]
#Z_out_3 = transform_Z(feat_test[:,1:],3)
#Z_out_6 = transform_Z(feat_test[:,1:],6)
#
#nums = list(range(10))
#lamb = 1
#
#for n in nums:
#    err_3,erroos_3,zeros3 = lin_OvA(Z_in_3,y_in,Z_out_3,y_out,lamb,n)
#    err_6,erroos_6,zeros6 = lin_OvA(Z_in_6,y_in,Z_out_6,y_out,lamb,n)
#    print("In sample error for " + str(n) + " with 3 features: " + str(err_3))
#    print("In sample error for " + str(n) + " with 6 features: " + str(err_6))
#    print("Out of sample error for " + str(n) + " with 3 features: " + str(erroos_3))
#    print("Out of sample error for " + str(n) + " with 6 features: " + str(erroos_6))
#    
# finals question 10

feat_test = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW8Q2 Data\\features.test.txt')
feat_train = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW8Q2 Data\\features.train.txt')

y_in = feat_train[:,0]
Z_in = transform_Z(feat_train[:,1:],6)
y_out = feat_test[:,0]
Z_out = transform_Z(feat_test[:,1:],6)

lamb = [0.01,1]

for l in lamb:
    err,erroos,test = lin_OvO(Z_in,y_in,Z_out,y_out,l,1,5)
    print("In sample error for lambda " + str(l)+": " + str(err))
    print("Out of sample error for lambda " + str(l)+ ": " + str(erroos))
    #print("All 0 error: " + str(test))