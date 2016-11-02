# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 20:02:27 2016

@author: philip.ball
"""

import numpy as np
import random as rnd
import matplotlib.pyplot as plt

def targetFunc(point):
    return np.sign(point[0]**2 + point[1]**2 - 0.6)

def noise(sign_out):
    r = rnd.randint(1,10)
    if r == 1:
        sign_out = -sign_out
    return sign_out
    
def y_out(x1,x2,noisy=True):
    y = np.array([0]*len(x1))
    for i in range(0,len(x1)-1):
        y[i] = targetFunc([x1[i],x2[i]])
        if noisy == True:
            y[i] = noise(y[i])
    return y
  
def wttolin(wt):    # converts the parametric weight vector into cartesian
    m = -wt[1]/wt[2]
    c = -wt[0]/wt[2]
    return [m,c]

def nrandpts(N,lo=-1,hi=1):
    return np.random.uniform(lo,hi,N)

def gettarg():
    t = np.linspace(0,2*np.pi,1000)
    x = np.sqrt(0.6)*np.cos(t)
    y = np.sqrt(0.6)*np.sin(t)
    return [x,y]
    
def pseudoinv(M):
    Msq = np.dot(np.transpose(M),M)
    return np.dot(np.linalg.inv(Msq),np.transpose(M))

def plotexp(x1,x2,y,wt_hyp):  # plots an instance of an experiment with lin reg fit
    plt.scatter(x1[y==1],x2[y==1],color = "blue")
    plt.scatter(x1[y==-1],x2[y==-1], color = "red")
    [x_t,y_t] = gettarg()
    plt.plot(x_t,y_t)
    x_co = np.linspace(-2,2,100)
    plt.plot(x_t,y_t)
    yh_co = np.poly1d(wttolin(wt_hyp))(x_co)
    plt.plot(x_co,yh_co)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(-1,1)  # axis display limits
    plt.ylim(-1,1)
    plt.gca().set_aspect('equal', adjustable='box') # equalise the x and y
    plt.show()
    
def exper_nosquare(N):
    x0 = [1]*N
    x1 = nrandpts(N)
    x2 = nrandpts(N)
    X_feat = np.column_stack((x0,x1,x2))
    y = y_out(x1,x2,True)
    w = np.dot(pseudoinv(X_feat),y)
    yh = np.sign(np.dot(w,np.transpose(X_feat)))
    Ein = 1 - np.mean(y==yh)
    #print("Error is: " + str(Ein))
    #plotexp(x1, x2, y, w)
    return Ein

def exper_square(N):
    x0 = [1]*N
    x1 = nrandpts(N)
    x2 = nrandpts(N)
    x1s = x1**2
    x2s = x2**2
    x1x2 = x1*x2
    y = y_out(x1,x2,True)
    X_feat = np.column_stack((x0,x1,x2,x1x2,x1s,x2s))
    w = np.dot(pseudoinv(X_feat),y)
    yh = np.sign(np.dot(w,np.transpose(X_feat)))
    y = y_out(x1,x2,True)
    Ein = 1 - np.mean(y==yh)
    return [Ein, w]
    
Eins = [0]*1000
w = [0]*1000
    
for i in range(0,1000):
    Eins[i],w[i] = exper_square(1000)
    

print(np.mean(np.array(Eins)))
print(sum(w)/len(w))
#noise_out = [0]*1000
#
#for i in range(0,1000):
#    noise_out[i] = noise(1)
#

Eouts = [0]*1000
x0 = [1]*1000
x1 = nrandpts(1000)
x2 = nrandpts(1000)
x1s = x1**2
x2s = x2**2
x1x2 = x1*x2
y = y_out(x1,x2,True)
X_feat = np.column_stack((x0,x1,x2,x1x2,x1s,x2s))
for i in range(0,1000):
    yh = np.sign(np.dot(w[i],np.transpose(X_feat)))
    Eouts[i] = 1 - np.mean(y == yh)
print(np.mean(np.array(Eouts)))
