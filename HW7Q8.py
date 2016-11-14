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
    y[y==0] == 1
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


# uses the SVM with a hard margin and quadprog to learn the separaion line
def SVM_hm(x1,x2,y):
    x = np.column_stack((x1,x2))
    Gy = np.outer(y,y)
    Gx = np.dot(x,np.transpose(x))
    G = np.multiply(Gy,Gx)
    G = G + 1e-8*np.identity(y.size)
    a = np.ones(y.size)
    Ident = np.identity(y.size)
    C = np.column_stack((y,Ident))
    #C = y[:,np.newaxis]
    b = np.zeros(y.size + 1)    
    #b = np.array([0.0])
    a_l, f, xu, iters, lagr, iact = qp.solve_qp(G,a,C,b,1)
    ay = a_l*y
    w = np.dot(np.transpose(x),ay[:,np.newaxis])
    b_c = get_b(y,a_l,x,w)
    return(a_l,w,b_c)
    
def get_b(y,a,x,w):
    SV_ind = np.argmax(a) # index of max alpha
    y_sv = y[SV_ind]    
    x_sv = x[SV_ind,:]
    s = 0
    for i in range(0,y.size):
        if a[i]>0:
            s = s + y[i]*a[i]*np.dot(x[i,:],x_sv)
    b = y_sv - s
    return b

# function to test the perceptron algorithm
def test_percep(N):
    x1,x2,y,l_x1,l_x2 = rand_inst(N)
    w = percep(x1,x2,y)
    plt_percep_hyp(x1,x2,y,w)

# plots the separation line given the inputs and outputs
def plt_percep_hyp(x1,x2,y,w):
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

def test_SVM(N):
    x1,x2,y,l_x1,l_x2 = rand_inst(N)
    a,w,b = SVM_hm(x1,x2,y)
    plt_SVM_hyp(x1,x2,y,w,b)
    return((a>1/(y.size)**2).sum())

def plt_SVM_hyp(x1_t,x2_t,y,w,b):
    s = 1000
    x1 = np.linspace(-1,1,s)
    x2 = np.linspace(-1,1,s)
    X1,X2 = np.meshgrid(x1,x2)
    Z = np.zeros((s,s))
    for i in range(0,s):
        x_cols = np.column_stack((X1[:,i],X2[:,i]))
        Z[:,i] = (np.dot(x_cols,w) + b).flatten()
    plt.contour(X1,X2,Z,levels = [0])
    plt.scatter(x1_t[y==1],x2_t[y==1],c = "red")
    plt.scatter(x1_t[y==-1],x2_t[y==-1])
    
def compare(N):
    x1,x2,y,l_x1,l_x2 = rand_inst(N)    # create a random instantiation
    a,w_S,b_S = SVM_hm(x1,x2,y) # train the SVM
    w_P = percep(x1,x2,y)   # train the Percep
    x1_test = np.random.uniform(-1,1,1000)
    x2_test = np.random.uniform(-1,1,1000)
    X_test = np.column_stack((x1_test,x2_test))
    X_test_0 = np.column_stack((np.ones(1000),X_test))
    y_test = LorR(x1_test,x2_test,l_x1,l_x2)
    y_SVM = np.sign(np.dot(X_test,w_S) + b_S).flatten()
    y_P = np.sign(np.dot(X_test_0,w_P))
    Err_SVM = (y_test != y_SVM).sum()
    Err_Per = (y_test != y_P).sum()
    return(Err_SVM, (a>1).sum(), Err_Per)

def compare_exper(N,runs):
    Err_S_a = np.zeros(runs)
    Err_P_a = np.zeros(runs)
    SV_Cnt_a = np.zeros(runs)
    SVM_Win = 0
    for i in range(0,runs):
        Err_S_a[i], SV_Cnt_a[i], Err_P_a[i] = compare(N)
        if Err_S_a[i]<Err_P_a[i]:
            SVM_Win = SVM_Win + 1
    print(SVM_Win/runs)
    print(Err_S_a.mean())
    print(Err_P_a.mean())
    print(SV_Cnt_a.mean())

compare_exper(100,1000)