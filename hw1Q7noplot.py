# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
#import matplotlib.pyplot as plt
import random as rand

def leftorright(point1, point2, testp):
    lorr = np.sign((point1[0]-point2[0]) * (testp[1]-point2[1]) - 
                                 (point1[1]-point2[1]) * (testp[0]-point2[0]))
    return lorr

N = 100  # number of points
it = 10000 # number of iterations for PLA

it_list = [0]*100  # placeholder for the 1000 different seeds we'll try
prob_list = [0]*100

for s in range(1,100):          
    
    np.random.seed()    # set the random seed to be consistent
    
    x1_co = np.random.uniform(-1,1,N)
    x2_co = np.random.uniform(-1,1,N)
    # now generate random points for monte carlo modelling later
    x1r_co = np.random.uniform(-1,1,1000)
    x2r_co = np.random.uniform(-1,1,1000)
    x0r_co = np.array([1]*1000)
    xr_ar = np.column_stack((x0r_co,x1r_co,x2r_co))
    yr = [0]*1000
    #end montecarlo bit
    rand1 = round(np.random.uniform(0,N-1))
    rand2 = rand1 # initialise rand2 to be rand1
    while rand1 == rand2:       # while rand2 and rand1 are the same
        rand2 = round(np.random.uniform(0,N-1))     # set rand2 to be a different number
    randx1 = [x1_co[rand1],x1_co[rand2]]
    randx2 = [x2_co[rand1],x2_co[rand2]]
    line = np.polyfit(randx1, randx2, deg=1)  # find the line between the xs and ys (ie: the m and the c)
    x1_ext = np.linspace(-2,2,100)   # get a linspace of x for -2 to 2
    x2_ext = np.poly1d(line)(x1_ext)  # get the ycoords given our m and c values
    y = [0]*N
    
    for i in range(0,N):
        lorr = leftorright([randx1[0],randx2[0]],[randx1[1],randx2[1]],
                           [x1_co[i],x2_co[i]])
        if (lorr != -1):
            y[i] = 1
        else:
            y[i] = -1
    
    y_ar = np.array(y)
        
    # perceptron learning algo
    wt = np.array([0,0,0])  # initialise to zeros, including w0
    x0_co = np.array([1]*N)
    x_ar = np.column_stack((x0_co,x1_co,x2_co))   # stack the x values, and add the 1s
    
    for z in range(1,it):
        y_h = np.sign(np.dot(wt,np.transpose(x_ar)))
        chk = y_h == y_ar #produce a boolean vector which shows which y values matched
        if chk.all():    # if we have perfect classification, ie: all elemnts are true
            it_list[s] = z   # print the number of iterations it took and break
            for i in range(0,1000):
                lorr = leftorright([randx1[0],randx2[0]],[randx1[1],randx2[1]],
                               [x1r_co[i],x2r_co[i]])
                if (lorr != -1):
                    yr[i] = 1
                else:
                    yr[i] = -1
        
            y_ar_r = np.array(yr)
            y_h_r = np.sign(np.dot(wt,np.transpose(xr_ar)))
            prob_list[s] = (y_ar_r == y_h_r).mean()
            break
        x_nm = x_ar[chk == False]   # extract the x vals which were misclassified
        y_nm = y_ar[chk == False]   # extract the y vals of those misclassified xvals
        ri = rand.randint(0,len(y_nm)-1)    # now randomly select the index one of the miscalissifed vals (uses a different rng to give us different iteration paths)
        wt = wt + y_nm[ri]*x_nm[ri]     # iterate the weight using the PLA

    
print(sum(it_list)/len(it_list))
print(1-sum(prob_list)/len(prob_list))