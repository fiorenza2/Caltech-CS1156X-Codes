# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 18:14:11 2016

@author: philip.ball
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd

def checksquare_exper():
    x1 = rnd.uniform(-1,1)
    x2 = rnd.uniform(-1,1)
    y1 = np.sin(np.pi * x1)
    y2 = np.sin(np.pi * x2)
    a = (x1**2*y1 + x2**2*y2)/(x1**4 + x2**4)
    return(a, x1, x2, y1, y2)

def checksquare(N):
    g_h = np.zeros(N)
    x = np.linspace(-1,1)
    y = np.sin(np.pi * x)
    for i in range(0,N):
        g_h[i],x1,x2,y1,y2 = checksquare_exper()
#        plt.plot(x,g_h[i]*x**2)
#        plt.plot(x,np.sin(x*np.pi))
#        plt.scatter([x1,x2],[y1,y2])
#        plt.show()
    y_h = np.outer(g_h,x**2)
    mse_all = np.trapz((y_h - y)**2, x) / 2
    return(np.mean(mse_all))
        
a = checksquare(10000)
print(a)