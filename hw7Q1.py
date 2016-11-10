# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:16:34 2016

@author: philip.ball
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#function to create the feature matrix given the input [x1,x2] and feature number k (where k = 0 is the bias feature)
def featurise(x_feat,k):
    r_i,c_i = x_feat.shape    #get size attributes of the in-sample data
    ones_i = np.ones(r_i)   #create the ones feature (bias term)
    x_i = x_feat[:,:2]        #create the power to the one terms (x1,x2)
    x_sq_i = x_feat[:,:2]**2  #create the square terms
    x_cr_i = x_feat[:,0]*x_feat[:,1]    # create the cross terms (x1x2)
    x_mods_i = np.fabs(x_feat[:,0] - x_feat[:,1])   # create the difference, but absolute
    x_modp_i = np.fabs(x_feat[:,0] + x_feat[:,1])   # create the addition, but absolute
    psi_x = np.column_stack((ones_i,x_i,x_sq_i,x_cr_i,x_mods_i,x_modp_i))
    psi_x = psi_x[:,:(k+1)]
    return(psi_x)

# function to create the non-regularised solution for optimal weights
def w_lin(psi_x,y):
    psi_tr = np.transpose(psi_x)
    psi_sq = np.dot(psi_tr,psi_x)
    invert = np.linalg.inv(psi_sq)
    w = np.dot(np.dot(invert,psi_tr),y)
    return(w)
    
def train_error_asses(in_s,out_s,w_train,k):
    x_feat_i = in_s[:,:2]   # get the features of the in sample
    x_feat_o = out_s[:,:2]  # get the features of the out sample
    y_i = in_s[:,2]     # get the classification of the in sample
    y_o = out_s[:,2]    # get the classification of the out sample
    psi_i = featurise(x_feat_i,k) # get the transformed feature vector in sample
    psi_o = featurise(x_feat_o,k) # get the transformed feature vector out sample
    w_t = w_train(psi_i,y_i)
    y_i_pred = np.sign(np.dot(psi_i,w_t))    # calculate the in sample classification predictions
    y_o_pred = np.sign(np.dot(psi_o,w_t))    # calculate the out sample classification predictions
    class_e_i = np.mean(y_i != y_i_pred)
    class_e_o = np.mean(y_o != y_o_pred)
    #plot_bound(x_feat_i,y_i,x_feat_o,y_o,w_t)
    return(class_e_i,class_e_o)

in_s = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW6Q2 Data\\in.dta')
out_s = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW6Q2 Data\\out.dta')

train = in_s[:25,:]
val = in_s[25:,:]

train_r = val
val_r = train

# Q1 answer below
print("Q1:")
for k in range(3,8):
    print("k = " + str(k) + " " + str(train_error_asses(train,val,w_lin,k)))

print("\nQ2:")
# Q2 answer below
for k in range(3,8):
    print("k = " + str(k) + " " + str(train_error_asses(train,out_s,w_lin,k)))

print("\nQ3:")
# Q3 answer below
for k in range(3,8):
    print("k = " + str(k) + " " + str(train_error_asses(train_r,val_r,w_lin,k)))


print("\nQ4:")
# Q3 answer below
for k in range(3,8):
    print("k = " + str(k) + " " + str(train_error_asses(train_r,out_s,w_lin,k)))