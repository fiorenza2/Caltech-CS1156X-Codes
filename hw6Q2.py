# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:28:47 2016

@author: philip.ball
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#load data
in_s = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW6Q2 Data\\in.dta')
out_s = np.loadtxt('C:\\Users\\philip.ball\\Documents\\AI-DS\\edX CS1156x\\Python Scripts\\HW6Q2 Data\\out.dta')

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
    return w

