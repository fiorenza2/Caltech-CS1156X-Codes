# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 10:35:01 2016

@author: philip.ball
"""

import numpy as np

def err_func(u,v):
    return (u*np.exp(v) - 2*v*np.exp(-u))**2
    
def pd_wrt_u(u,v):
    return 2*(np.exp(v) + 2*v*np.exp(-u))*(u*np.exp(v) - 2*v*np.exp(-u))
    
def pd_wrt_v(u,v):
    return 2*(u*np.exp(v) - 2*np.exp(-u))*(u*np.exp(v) - 2*v*np.exp(-u))
    
def iteration_gd(u,v,eta):
    u_new = u - eta*pd_wrt_u(u,v)
    v_new = v - eta*pd_wrt_v(u,v)
    return u_new,v_new

def iteration_cd(u,v,eta):
    u_new = u - eta*pd_wrt_u(u,v)
    v_new = v - eta*pd_wrt_v(u_new,v)
    return u_new,v_new
    
def exper_graddesc(start, eta):
    err = err_func(start[0],start[1])
    u = start[0]
    v = start[1]
    it = 0
    while err > 10**-14:
        u,v = iteration_gd(u,v,eta)
        it = it + 1
        err = err_func(u,v)
    return it, u , v

def exper_coorddesc(start,eta):
    u = start[0]
    v = start[1]
    for it in range(0,15):
        u,v = iteration_cd(u,v,eta)
    return err_func(u,v)
    
print(exper_graddesc([1,1],0.1))
print(exper_coorddesc([1,1],0.1))