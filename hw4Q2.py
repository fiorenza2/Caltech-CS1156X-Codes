# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:50:41 2016

@author: philip.ball
"""

import numpy as np
import matplotlib.pyplot as plt

def app_gf(N,dvc):
    return(N**dvc)

def VC_Bound(N,dvc,g_func,delta):
    eta_upper = np.sqrt(8/N * np.log(4*g_func(2*N,dvc)/delta))
    return(eta_upper)
    
def Rad_Bound(N,dvc,g_func,delta):
    eta_upper = (np.sqrt(2/N * np.log(2*N*g_func(N,dvc)/1)) + 
                         np.sqrt(2/N * np.log(1/delta)) + 1/N)
    return(eta_upper)

def PVdB_Bound(N,dvc,g_func,delta):
    x = 6*g_func(2*N,dvc)/delta
    eta_upper = (1 + np.sqrt(1 + N*np.log(x)))/N
    return(eta_upper)

def Dev_Bound(N,dvc,g_func,delta):
    z = np.log(4*g_func(N**2,dvc)/delta)
    eta_upper = (2 + np.sqrt(4 + (2*N-4)*z))/(2*N - 4)
    return(eta_upper)

dvc = 10
delta = 0.05
N = np.linspace(1,10000,100)

y_VC = VC_Bound(N,dvc,app_gf,delta)
y_Rad = Rad_Bound(N,dvc,app_gf,delta)
y_PVdB = PVdB_Bound(N,dvc,app_gf,delta)
y_Dev = Dev_Bound(N,dvc,app_gf,delta)

fig, ax = plt.subplots()
ax.plot(N,y_VC,'b', label = 'VC')
ax.plot(N,y_Rad, 'k', label = 'Rademacher')
ax.plot(N,y_PVdB, 'g', label = 'P and VdB')
ax.plot(N,y_Dev, 'r', label = 'Devroye')
ax.set_yscale('log')
ax.set_xlabel('Number of samples (N)')
ax.set_ylabel('Generalization error upper bound at 95% confidence')
legend = ax.legend(loc='upper right', shadow=True)
plt.show()

N = 10000

print("Different bounds at N = 10000...")
print("VC: " + str(VC_Bound(N,dvc,app_gf,delta)))
print("Rad: " + str(Rad_Bound(N,dvc,app_gf,delta)))
print("PVdB: " + str(PVdB_Bound(N,dvc,app_gf,delta)))
print("Dev: " + str(Dev_Bound(N,dvc,app_gf,delta)))

N = 5

print("\n"+"Different bounds at N = 5...")
print("VC: " + str(VC_Bound(N,dvc,app_gf,delta)))
print("Rad: " + str(Rad_Bound(N,dvc,app_gf,delta)))
print("PVdB: " + str(PVdB_Bound(N,dvc,app_gf,delta)))
print("Dev: " + str(Dev_Bound(N,dvc,app_gf,delta)))