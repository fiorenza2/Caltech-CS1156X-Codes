# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 10:41:09 2016

@author: philip.ball
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb
import random as rand

def leftorright(point1, point2, testp):         # determines if a point lies on the right or left of a line given by two points
    lorr = np.sign((point1[0]-point2[0]) * (testp[1]-point2[1]) - 
                                 (point1[1]-point2[1]) * (testp[0]-point2[0]))
    return lorr

def nrandpts(N,lo=-1,hi=1):
    return np.random.uniform(lo,hi,N)

def pseudoinv(M):
    Msq = np.dot(np.transpose(M),M)
    return np.dot(np.linalg.inv(Msq),np.transpose(M))

def y_out(x1,x2,pt1,pt2):   # returns the vector of y outputs, 1 if on one side, 0 on the other of a line defined by pt1 and pt2
    N = len(x1)
    y = [0]*N
    for i in range(0,N):
        test_pt = [x1[i],x2[i]]
        y[i] = leftorright(pt1,pt2,test_pt)
    return np.array(y)

def wttolin(wt):    # converts the parametric weight vector into cartesian
    m = -wt[1]/wt[2]
    c = -wt[0]/wt[2]
    return [m,c]
    
def plotexp(x1,x2,y,pt1,pt2,wt_hyp):  # plots an instance of an experiment with lin reg fit
    plt.scatter(x1[y==1],x2[y==1],color = "blue")
    plt.scatter(x1[y==-1],x2[y==-1], color = "red")
    #to plot the line which cuts through pt1 and pt2, we need to transform into xy space
    sep_line = np.polyfit([pt1[0],pt2[0]],[pt1[1],pt2[1]],deg=1)    # y = mx + c
    x_co = np.linspace(-1,1,100)
    y_co = np.poly1d(sep_line)(x_co)
    plt.plot(x_co,y_co)
    yh_co = np.poly1d(wttolin(wt_hyp))(x_co)
    plt.plot(x_co,yh_co)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(-1,1)  # axis display limits
    plt.ylim(-1,1)
    plt.gca().set_aspect('equal', adjustable='box') # equalise the x and y
    plt.show()

def exper(N):   # an instance of the experiment to fit a lin reg hypothesis
    x0 = [1]*N      # offset terms
    x1 = nrandpts(N,-1,1)   # x1 rand gen
    x2 = nrandpts(N,-1,1)   # etc.
    ln_pt1 = nrandpts(2,-1,1)   # pick a random x1,x2 for the sep line
    ln_pt2 = nrandpts(2,-1,1)   # and another
    y = y_out(x1,x2,ln_pt1,ln_pt2)  # generate the corresponding required outputs
    X_feat = np.column_stack((x0,x1,x2))  # create the feature vector
    X_pi = pseudoinv(X_feat)    # get the pseudoinverse
    w = np.dot(X_pi,y)  # calculate the weights directly
    yh = np.sign(np.dot(w,np.transpose(X_feat)))  # get the predicted y values
    #plotexp(x1,x2,y,ln_pt1,ln_pt2,w)
    E1 = 1 - np.mean(yh==y)
    #print("The Error was: " + str(E1))
    return [E1,w,ln_pt1,ln_pt2,x1,x2,y]

#[E1,w] = exper(100)
#N = 10
#E1s = [0]*N
#ws = [0]*N
#pt1s = [0]*N
#pt2s = [0]*N

#for i in range(0,N):
#    E1s[i],ws[i],pt1s[i],pt2s[i] = exper(100)

#x0t = [1]*1000
#x1t = nrandpts(1000,-1,1)
#x2t = nrandpts(1000,-1,1)
#Xt_feat = np.column_stack((x0t,x1t,x2t))
#Eout = [0]*1000
#
#for i in range(0,1000):
#    y = y_out(x1t,x2t,pt1s[i],pt2s[i])
#    yh = np.sign(np.dot(ws[i],np.transpose(Xt_feat)))
#    Eout[i] = 1 - np.mean(y==yh)
#
#np.mean(np.array(Eout))

def perceptronLR(N):    # runs the perceptron algo with initial weights decided by lin reg
    [E1,w,l1,l2,x1,x2,y] = exper(N)   # get the LR results
    # perceptron learning algo
    wt = w  # initialise to the LR output
    x0_co = [1]*N
    x_ar = np.column_stack((x0_co,x1,x2))   # stack the x values, and add the 1s

    for z in range(1,1000000):
        y_h = np.sign(np.dot(wt,np.transpose(x_ar)))
        chk = y_h == y #produce a boolean vector which shows which y values matched
        if chk.all():    # if we have perfect classification, ie: all elemnts are true
            return z  # print the number of iterations it took and break
            break
        x_nm = x_ar[chk == False]   # extract the x vals which were misclassified
        y_nm = y[chk == False]   # extract the y vals of those misclassified xvals
        #ri = round(np.random.uniform(0,len(y_nm)-1))
        ri = rand.randint(0,len(y_nm)-1)    # now randomly select the index one of the miscalissifed vals (uses a different rng to give us different iteration paths)
        wt = wt + y_nm[ri]*x_nm[ri]     # iterate the weight using the PLA
    return z

its = [0]*1000
    
for i in range(0,1000):
    its[i] = perceptronLR(10)