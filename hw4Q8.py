# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 00:11:54 2016

@author: philip.ball
"""

import numpy as np
import scipy.misc as mi
import matplotlib.pyplot as plt

def growth_func(N,q):
    x = 0
    for i in range(1,N):
        x = x + 2**(N-i-1)*mi.comb(i,q, exact=True)
    g = 2**N - x
    return(g)

def growth_bound(N,d_vc):
    x = 0
    for i in range(0,d_vc+1):
        x = x + mi.comb(N,i)
    return(x)

qrng = 42
    
x = range(1,qrng)
gf = np.zeros(qrng-1)
gb = np.zeros(qrng-1)

N = 50

for i in x:
    q = i
    d_vc = i
    gf[i-1] = growth_func(N,q)
    gb[i-1] = growth_bound(N,d_vc)

plt.plot(x,gb,x,gf)

#print(gf)
#print(gb)
print("The min difference between the bounds is: " + str((gb-gf).min()))