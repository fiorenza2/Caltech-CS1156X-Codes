# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import random as rnd
import matplotlib.pyplot as plt

def mean(numlist):
    return float(sum(numlist)/max(len(numlist),1))

def headortale():
    hort = rnd.randint(0,1)
    return hort
    
def flip10():
    ht = [0]*10
    for i in range(0,10):
        ht[i] = headortale()
    return ht
    
def experiment():
    sim = [0]*1000
    for a in range(0,1000):
        sim[a] = mean(flip10())
    return sim

def getmeans(sim_list):
    nu_1 = sim_list[0]
    nu_min = min(sim_list)
    r_var = rnd.randint(0,len(sim_list)-1)
    nu_rand = sim_list[r_var]
    return nu_1, nu_min, nu_rand
    
def multexp():
    first_list = [0]*10000
    min_list = [0]*10000
    rand_list = [0]*10000
    for z in range(0,10000):
        first_list[z], min_list[z], rand_list[z] = getmeans(experiment())
    return first_list, min_list, rand_list
    
#first, mini, random = multexp()
    
def f_eps(eps, delta):
    return np.sum(delta>eps)/np.size(delta)
    
def f_hoeff(eps,N):
    return 2*np.exp(-2*N*eps**2)

plt.plot(eps,P, label ="rand")
plt.plot(eps,Hoeff,label="Hoeff")
plt.plot(eps,Pm,label="min")
plt.legend()
plt.xlabel("tolerance")
plt.ylabel("probability the in and out sample is greater than tolerance")
plt.show()