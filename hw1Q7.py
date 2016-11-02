# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rand

def leftorright(point1, point2, testp):
    lorr = np.sign((point1[0]-point2[0]) * (testp[1]-point2[1]) - 
                                 (point1[1]-point2[1]) * (testp[0]-point2[0]))
    return lorr

N = 100  # number of points
it = 10000 # number of iterations

np.random.seed()    # set the random seed to be consistent

x1_co = np.random.uniform(-1,1,N)
x2_co = np.random.uniform(-1,1,N)
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

# the below plots the scatter and the demarcation line
plt.scatter(x1_co[y_ar==1],x2_co[y_ar==1],color = 'b')
plt.scatter(x1_co[y_ar==-1],x2_co[y_ar==-1],color = 'r')
plt.plot(x1_ext,x2_ext)
plt.xlim(-1,1)  # axis display limits
plt.ylim(-1,1)
plt.gca().set_aspect('equal', adjustable='box') # equalise the x and y
plt.xlabel("x1")
plt.ylabel("x2")

# perceptron learning algo
wt = np.array([0,0,0])  # initialise to zeros, including w0
x0_co = np.array([1]*N)
x_ar = np.column_stack((x0_co,x1_co,x2_co))   # stack the x values, and add the 1s

for z in range(1,it):
    y_h = np.sign(np.dot(wt,np.transpose(x_ar)))
    chk = y_h == y_ar #produce a boolean vector which shows which y values matched
    if chk.all():    # if we have perfect classification, ie: all elemnts are true
        print("Iterations: " + str(z))  # print the number of iterations it took and break
        break
    x_nm = x_ar[chk == False]   # extract the x vals which were misclassified
    y_nm = y_ar[chk == False]   # extract the y vals of those misclassified xvals
    #ri = round(np.random.uniform(0,len(y_nm)-1))
    ri = rand.randint(0,len(y_nm)-1)    # now randomly select the index one of the miscalissifed vals (uses a different rng to give us different iteration paths)
    wt = wt + y_nm[ri]*x_nm[ri]     # iterate the weight using the PLA

# show the perceptron plot
sep_line = [0]*2
sep_line[0] = -wt[1]/wt[2]
sep_line[1] = -wt[0]/wt[2]
x2h_ext = np.poly1d(sep_line)(x1_ext)
plt.plot(x1_ext,x2h_ext)
plt.show()