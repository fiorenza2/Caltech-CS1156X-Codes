# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:48:04 2016

@author: philip.ball
"""

import numpy as np
import matplotlib.pyplot as plt

def lr_pt(x1,x2,l_x1,l_x2):
    lr = np.sign((l_x1[1] - l_x1[0]) * (x2 - l_x2[0]) - 
                 (l_x2[1] - l_x2[0]) * (x1 - l_x1[0]))
    return lr
    
def instan(N):  # an instantiation of an experiment with N points
    x1 = np.random.uniform(-1,1,N)  # x1 coords
    x2 = np.random.uniform(-1,1,N)  # x2 coords
    l_x1 = np.random.uniform(-1,1,2)  # x1 coords for line
    l_x2 = np.random.uniform(-1,1,2)  # x2 coords for line
    y_d = lr_pt(x1,x2,l_x1,l_x2)    # get whether the points were on the left or right
    return(x1,x2,l_x1,l_x2,y_d)

def grad_Ein(x_row,y,w):
    grad = -x_row*y/(1 + np.exp(y*np.dot(x_row,w)))
    return(grad)
    
def sgd_epoch(x_mat,y,w,eta):
    N = y.size  # get the number of points we need to iterate across
    ind_ran = np.random.permutation(np.array(range(0,N)))   # random index order
    for ind in ind_ran:
        w = w - eta*grad_Ein(x_mat[ind,:],y[ind],w)
    return(w)

def stoch_gd(x_mat,y):
    eta = 0.01  # set learning rate
    epoch_c = 0 # epoch counter
    w = np.array([0,0,0]) # initialies weights to 0
    w_prev = w + 100  # create a previous weights vector to compare with (+100 to ensure greater than 0.01 mag difference)
    while np.linalg.norm(w - w_prev) >= 0.01:   # while the mag of the diff between weights is still high
        w_prev = w  # set the previous w
        w = sgd_epoch(x_mat,y,w,eta)    # calculate the new w
        epoch_c = epoch_c + 1
    return w, epoch_c

def Ecalc(x_mat,y,w):
    dot_w_x = np.dot(x_mat,w)
    Err = np.sum(np.log(1 + np.exp(-y * dot_w_x)))/y.size   # calculate the cross entropy error
    return Err    

def exper(n):   # run the logistic regression experiment n times
    N = 100 # number of in sample points in an instantiation
    outsize = 100000 # out of sample size
    E_out = np.zeros(n) # instantiate the out of sample error vector
    epoch_cnt = np.zeros(n) # instantiate the epoch counting vector
    x0 = np.ones(N)
    x0_oos = np.ones(outsize)
    for i in range(0,n):
        x1,x2,l_x1,l_x2,y_d = instan(N)
        x_mat = np.column_stack([x0,x1,x2])
        w,epoch_cnt[i] = stoch_gd(x_mat,y_d)
        x1_oos = np.random.uniform(-1,1,outsize)    # out of samples x1 coords
        x2_oos = np.random.uniform(-1,1,outsize)    # out of samples x2 coords
        y_D = lr_pt(x1_oos, x2_oos, l_x1, l_x2)     # get the y_out of the out of samples
        x_mat_oos= np.column_stack([x0_oos,x1_oos,x2_oos])
        E_out[i] = Ecalc(x_mat_oos,y_D,w)
    return np.mean(E_out),np.mean(epoch_cnt)

print(exper(100))