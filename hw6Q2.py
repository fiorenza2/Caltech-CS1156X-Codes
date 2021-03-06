# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:28:47 2016

@author: philip.ball
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#function to create the feature matrix given the input [x1,x2] (but can also have y term)
def featurise(x_feat):
    r_i,c_i = x_feat.shape    #get size attributes of the in-sample data
    ones_i = np.ones(r_i)   #create the ones feature (bias term)
    x_i = x_feat[:,:2]        #create the power to the one terms (x1,x2)
    x_sq_i = x_feat[:,:2]**2  #create the square terms
    x_cr_i = x_feat[:,0]*x_feat[:,1]    # create the cross terms (x1x2)
    x_mods_i = np.fabs(x_feat[:,0] - x_feat[:,1])   # create the difference, but absolute
    x_modp_i = np.fabs(x_feat[:,0] + x_feat[:,1])   # create the addition, but absolute
    psi_x = np.column_stack((ones_i,x_i,x_sq_i,x_cr_i,x_mods_i,x_modp_i))
    return(psi_x)

# function to create the non-regularised solution for optimal weights
def w_lin(psi_x,y):
    psi_tr = np.transpose(psi_x)
    psi_sq = np.dot(psi_tr,psi_x)
    invert = np.linalg.inv(psi_sq)
    w = np.dot(np.dot(invert,psi_tr),y)
    return(w)

def w_aug(psi_x,y,lamb):
    psi_tr = np.transpose(psi_x)
    psi_sq = np.dot(psi_tr,psi_x)
    row,col = psi_sq.shape
    reg_term = np.identity(row) * lamb
    invert = np.linalg.inv(psi_sq + reg_term)
    w = np.dot(np.dot(invert,psi_tr),y)
    return(w)
    
# function which returns the in and out of sample errors for a given set of inputs and training schema
    #note that if there is no lambda, set this to be "NA"
def train_error_asses(in_s,out_s,w_train,lamb):
    x_feat_i = in_s[:,:2]   # get the features of the in sample
    x_feat_o = out_s[:,:2]  # get the features of the out sample
    y_i = in_s[:,2]     # get the classification of the in sample
    y_o = out_s[:,2]    # get the classification of the out sample
    psi_i = featurise(x_feat_i) # get the transformed feature vector in sample
    psi_o = featurise(x_feat_o) # get the transformed feature vector out sample
    if lamb == "NA":
        w_t = w_train(psi_i,y_i)
    else:
        w_t = w_train(psi_i,y_i,lamb)
    y_i_pred = np.sign(np.dot(psi_i,w_t))    # calculate the in sample classification predictions
    y_o_pred = np.sign(np.dot(psi_o,w_t))    # calculate the out sample classification predictions
    class_e_i = np.mean(y_i != y_i_pred)
    class_e_o = np.mean(y_o != y_o_pred)
    plot_bound(x_feat_i,y_i,x_feat_o,y_o,w_t)
    return(class_e_i,class_e_o)

# function that plots the separation boundary for a given weight set
def plot_bound(x_in,y_in,x_out,y_out,weight):
    s = 1000
    Z = np.zeros((s,s))
    x1 = np.linspace(x_out[:,0].min(),x_out[:,0].max(),s)
    x2 = np.linspace(x_out[:,1].min(),x_out[:,1].max(),s)
    X1,X2 = np.meshgrid(x1,x2)
    for i in range(0,s):
        x_cols = np.column_stack((X1[:,i],X2[:,i]))
        Z[:,i] = np.dot(featurise(x_cols),weight)
    plt.contour(X1,X2,Z,levels=[0])
    plt.scatter(x_in[y_in[:]==1][:,0],x_in[y_in[:]==1][:,1])
    plt.scatter(x_in[y_in[:]==-1][:,0],x_in[y_in[:]==-1][:,1], c = "red")
    plt.title("In sample data with separation contour")
    plt.show()
    plt.contour(X1,X2,Z,levels=[0])
    plt.scatter(x_out[y_out[:]==1][:,0],x_out[y_out[:]==1][:,1])
    plt.scatter(x_out[y_out[:]==-1][:,0],x_out[y_out[:]==-1][:,1], c = "red")
    plt.title("Out of sample data with separation contour")
    plt.show()

#load data
in_s = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW6Q2 Data\\in.dta')
out_s = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW6Q2 Data\\out.dta')

k = -1

lamb = 10**k

print("Non-regularized")
print(train_error_asses(in_s,out_s,w_lin,"NA"))
print("\nRegularized with lambda " + str(lamb))
print(train_error_asses(in_s,out_s,w_aug,lamb))