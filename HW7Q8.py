# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 20:45:36 2016

@author: philip.ball
"""

import numpy as np
import matplotlib.pyplot as plt
import quadprog as qp

# creates a random instantiation of an experiment with N points, returning the point locations and their values
def rand_inst(N):
    x1 = np.random.uniform(-1,1,N)
    x2 = np.random.uniform(-1,1,N)
    y = np.zeros(N)
    while np.unique(y).size == 1:
        l_x1 = np.random.uniform(-1,1,2)
        l_x2 = np.random.uniform(-1,1,2)
        y = LorR(x1,x2,l_x1,l_x2)
    return(x1,x2,y,l_x1,l_x2)
        
# returns the vector which states which side the points generated are on
def LorR(x1,x2,l_x1,l_x2):
    side = (l_x1[1] - l_x1[0])*(x2 - l_x2[0]) - (l_x2[1] - l_x2[0])*(x1 - l_x1[0])
    y = np.sign(side)
    return y

# uses a perceptron algorithm to learn a separation line for the data set
def percep(x1,x2,y):
    x0 = np.ones(y.size)
    x = np.column_stack((x0,x1,x2))    # crete a feature matrix
    w = np.zeros(3)    # instantiate the weights vector
    it = 100000 # max number of PLA iterations
    for i in range(0,it):
        y_pred = np.sign(np.dot(w,np.transpose(x)))
        y_check = (y_pred == y)
        if y_check.all():
            break
        x_mis = x[y_check == 0]
        y_mis = y[y_check == 0]
        h = len(y_mis)
        p = np.random.randint(0,h)
        w = w + y_mis[p]*x_mis[p,:]
    return w

def test_percep(N):
    x1,x2,y,l_x1,l_x2 = rand_inst(10)
    w = percep(x1,x2,y)
    sep_line = np.zeros(2)
    sep_line[0] = -w[1]/w[2]
    sep_line[1] = -w[0]/w[2]
    x1_co = np.linspace(-1,1,100)
    x2_co = np.poly1d(sep_line)(x1_co)
    plt.plot(x1_co,x2_co)
    plt.scatter(x1[y==1],x2[y==1],c = "red")
    plt.scatter(x1[y==-1],x2[y==-1])
    plt.xlim(-1,1)  # axis display limits
    plt.ylim(-1,1)
    plt.gca().set_aspect('equal', adjustable='box') # equalise the x and y
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
    
test_percep(10)