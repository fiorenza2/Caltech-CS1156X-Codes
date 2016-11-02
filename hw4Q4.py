# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 17:30:28 2016

@author: philip.ball
"""

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

def get_hyp(N):
    g_hyp = np.zeros(N)
    #x = np.linspace(-1,1)
    for i in range(0,N):
        g_hyp[i],x1,x2,y1,y2 = fit_exper()
        #plt.plot(x,g_hyp[i]*x)
    return(np.mean(g_hyp))
        
def fit_exper():
    x1 = rnd.uniform(-1,1) # one random x
    x2 = rnd.uniform(-1,1) # another random x
    y1 = np.sin(x1*np.pi)      # get the corresponding y
    y2 = np.sin(x2*np.pi)
    a = (x1*y1 + x2*y2)/(x1**2 + x2**2) # min mse wrt a
    return(a,x1,x2,y1,y2)

a_fin = get_hyp(50000)
#    
print("The mean hypothesis weight value is " + str(a_fin))

def plotcomp(a_fin):
    x = np.linspace(-1,1,10000)
    y = np.sin(np.pi * x)
    y_e = a_fin * x
    y_a = 3/np.pi * x
    plt.plot(x,y,x,y_e,x,y_a)
    plt.show()
    mse_s = np.trapz((x*a_fin-y)**2,x) / 2
    mse_d = np.trapz((x*3/np.pi-y)**2,x) / 2
    print("mse of sample method: " + str(mse_s))
    print("mse of direct fit (analytical): " + str(mse_d))
    return(mse_s, mse_d)
    
print("\nI also calculated a direct value for a fit for our 'ax' model to sine. I wanted to compare this to the fit obtained through the sampling method:")
bias,fit = plotcomp(a_fin)
    
def plotcheck():
    x = np.linspace(-1,1,10000)
    y = np.sin(np.pi * x)
    a,x1,x2,y1,y2 = fit_exper()
    plt.scatter([x1,x2],[y1,y2])
    plt.scatter(0,0, c = 'r')
    plt.plot(x,a*x,x,y)
    plt.axes().set_aspect('equal')
    plt.axis([-1, 1, -1, 1])
    plt.show()
    print("mse: " + str(0.5*((x1*a - y1)**2 + (x2*a - y2)**2)))
    b = a + 0.01
    c = a - 0.01
    print("mse with a + 0.01: " + str(0.5*((x1*b - y1)**2 + (x2*b - y2)**2)))
    print("mse with a - 0.01: " + str(0.5*((x1*c - y1)**2 + (x2*c - y2)**2)))
    #print(x1, x2, y1, y2, a)

print("\nBelow is a simulation of one iteration")
plotcheck()

def calcvar(a_fin):
    N = 50000
    x = np.linspace(-1,1)
    y_sin = np.sin(np.pi * x)
    g_hyp = np.zeros(N)
    for i in range(0,N):
        g_hyp[i],x1,x2,y1,y2 = fit_exper()
    g_hyp_ar = np.array(g_hyp)
    g_var_mse = (np.outer(g_hyp_ar,x) - a_fin * x)**2
    g_var_split = np.trapz(g_var_mse,x) / 2
    g_var = np.mean(g_var_split)
    g_x = np.outer(g_hyp_ar,x)
    g_mse_split = np.trapz((g_x-y_sin)**2,x) / 2
    g_mse = np.mean(g_mse_split)
    return(g_var,g_mse)

var, mse_t = calcvar(a_fin)
    
print("\nThe variance of this model is: " + str(var))
print("\nThe total error of this model is " + str(mse_t) + ", which should be close to " + str(var + bias))

