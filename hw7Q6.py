# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:47:26 2016

@author: philip.ball
"""

import numpy as np
import matplotlib.pyplot as plt

def e_min():
    e1 = np.random.uniform(0,1)
    e2 = np.random.uniform(0,1)
    e = np.min([e1,e2])
    return(e)

def e_exper(n):
    e_val = np.zeros(n)
    for i in range(0,n):
        e_val[i] = e_min()
    plt.pdf(e_val,100)
    return(np.mean(e_val))
    
print(e_exper(100000))