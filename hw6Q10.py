# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 19:03:23 2016

@author: philip.ball
"""

import numpy as np
import matplotlib.pyplot as plt

def get_Nw(ni,s1,s2,no):
    return(ni*(s1-1)+s1*(s2-1)+s2*no)

Nw = np.zeros(36)
    
for i in range(1,36):
    s1 = i
    s2 = 36 - s1
    Nw[i] = get_Nw(10,s1,s2,1)

plt.plot(Nw)