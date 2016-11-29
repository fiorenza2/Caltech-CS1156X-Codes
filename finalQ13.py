# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 19:32:44 2016

@author: philip.ball
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import warnings

warnings.simplefilter("error")

def create_Points(n):
    X1 = np.random.uniform(-1,1,n)
    X2 = np.random.uniform(-1,1,n)
    X = np.column_stack((X1,X2))
    Y = classify_targ(X)
    X = np.column_stack((X1,X2))
    return(X,Y)
    
def classify_targ(X):
    X1 = X[:,0]
    X2 = X[:,1]
    c = X2 - X1 + 0.25*np.sin(np.pi * X1)
    return(np.sign(c))

# where the h denotes hard margin
def train_SVM_hRBF(X,Y,g):
    svc = SVC(kernel = 'rbf', C = float('inf'), gamma = g, shrinking = False)
    model_RBF = svc.fit(X,Y)
    return(model_RBF)

# return RBF weights given inputs
def train_RBF(X,Y,K,gam):
    cluster, mu = lloyds_alg(X,K)
    rx,cx = X.shape
    rm,cm = mu.shape
    psi = np.zeros((rx,rm))
    for i in range(0,rx):
        for j in range(0,rm):
            psi[i,j] = (np.linalg.norm(X[i,:] - mu[j,:]))**2
    psi = np.exp(-gam * psi)
    psi = np.column_stack((np.ones(rx),psi))     # add the bias term
    psi_t = np.transpose(psi)
    pre = np.linalg.pinv(np.dot(psi_t,psi))
    w_RBF = np.dot(np.dot(pre,psi_t),Y)
    return(w_RBF,cluster,mu)

# function to calculate the classification hypothesis given some weights of an RBF
def RBF_hyp_class(X,w,mu,gam):
    rx,cx = X.shape
    b = w[0]
    w = w[1:]
    Y = np.zeros(rx)
    for i in range(0,w.size):
        Xsub = X - mu[i,:]
        temp = (np.linalg.norm(Xsub,axis = 1))**2
        temp = np.exp(temp * -gam) * w[i]
        Y = Y + temp
    Y = Y + b
    Y = np.sign(Y)
    return(Y)
        
def lloyds_alg(X,K):
    r,c = X.shape
    mu_init_X1 = np.random.uniform(-1,1,K)
    mu_init_X2 = np.random.uniform(-1,1,K)
    mu = np.column_stack((mu_init_X1,mu_init_X2))
    mu_prev = mu + 0.5        # instantiate a "previous" mu vector for convergence checking purposes
    dist_prev = np.ones(r) * 100    # initialise the vector of previous euclidian distances
    cluster = np.zeros(r)   # the vector containing which cluster a X value belongs to
    while (mu_prev != mu).any():
        mu_prev = mu
        for i in range(0,K):
            X_minus = X - mu[i,:]	# X_minus contains the difference between the X's and the cluster i
            X_dist = np.linalg.norm(X_minus, axis = 1)  # calculate the Euclidian distance
            less_ind = dist_prev > X_dist   # create an index which tells you which values were less than the previously smallest value
            dist_prev[less_ind] = X_dist[less_ind]  # rewrite the previously smallest value for all entries where the current is smaller
            cluster[less_ind] = i   # for these entries, the cluster they belong to must be i
        if np.unique(cluster).size != K:
            break
        for i in range(0,K):
            c_ind = (cluster == i)    # get the index of all things belonging to cluster i
            mu[i,:] = np.mean(X[c_ind], axis = 0) # take the mean of all the X-coords of this cluster, and set this to be the new mu
    if np.unique(cluster).size != K:    # if we had at least a cluster with no X points...
        cluster, mu = lloyds_alg(X,K)   # rerun the algorithm
    return(cluster, mu)

# returns the classification error
def test_SVM(X,Y,model):
    Y_hyp = model.predict(X)
    return((Y_hyp != Y).mean())

# returns the classification error
def test_RBF(X,Y,w,mu,gam):
    Y_h = RBF_hyp_class(X,w,mu,gam)
    return((Y_h != Y).mean())

def getClassLine(X,model):
    X1 = np.linspace(X[:,0].min(),X[:,0].max())
    X2 = np.linspace(X[:,1].min(),X[:,1].max())
    xx,yy = np.meshgrid(X1,X2)
    Z = model.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)    
    return(xx,yy,Z)

def plot_RBF(X,Y,w,mu,gam):
    x1 = np.linspace(-1,1)
    x2 = np.linspace(-1,1)
    xx,yy = np.meshgrid(x1,x2)
    Z = RBF_hyp_class(np.c_[xx.ravel(),yy.ravel()],w,mu,gam)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z, alpha = 0.8,cmap = plt.cm.coolwarm)
    plt.scatter(X[:,0][Y==1],X[:,1][Y==1],c = 'green')
    plt.scatter(X[:,0][Y==-1],X[:,1][Y==-1],c = 'red')
    plt.scatter(mu[:,0],mu[:,1],c = 'yellow')
    plt.show()
    
X,Y = create_Points(100)

X_o,Y_o = create_Points(5000)

gamma = 1.5

######################################## test code for lloyds

#clu,mu = lloyds_alg(X,2)
#
#plt.scatter(X[:,0][clu == 1],X[:,1][clu == 1], c = 'red')
#plt.scatter(X[:,0][clu == 0],X[:,1][clu == 0], c = 'green')
#plt.scatter(mu[:,0],mu[:,1], c = 'yellow')


####################################### below is the code for Q13
#its = 1
#
#sep_yn = np.zeros(its)
#
#for i in range(0,its):
#    X,Y = create_Points(100)
#    model = train_SVM_hRBF(X,Y,1.5)
#    E_in = test_SVM(X,Y,model)
#    if E_in == 0:
#        sep_yn[i] = 0
#    else:
#        sep_yn[i] = 1
#
#X1 = X[:,0]
#X2 = X[:,1]
#xx,yy,Z = getClassLine(X,model)
#plt.contourf(xx,yy,Z,cmap = plt.cm.coolwarm, alpha = 0.8)
#plt.scatter(X1[Y == 1],X2[Y == 1], c = 'green')
#plt.scatter(X1[Y == -1], X2[Y == -1], c = 'red')
#plt.show()
#
#print(sep_yn.mean())

####################################### below is test code for RBF

#w,cluster,mu = train_RBF(X,Y,9,1.5)
#Y_hyp = RBF_hyp_class(X,w,mu,1.5)
#print((Y != Y_hyp).mean())
#plot_RBF(X,Y,w,mu,1.5)

############################# below is code for Q14 and Q15

#runs = 1000
#
#SVM_win = np.zeros(runs)
#
#for i in range(0,runs):
#    X,Y = create_Points(100)
#    Xo,Yo = create_Points(5000)
#    model = train_SVM_hRBF(X,Y,gamma)   # get the SVM model
#    w,cluster,mu = train_RBF(X,Y,12,gamma)   # get the RBF model
#    E_out_SVM = test_SVM(Xo,Yo,model)
#    E_out_RBF = test_RBF(Xo,Yo,w,mu,gamma)
#    if E_out_SVM < E_out_RBF:
#        SVM_win[i] = 1
#
#print(SVM_win.mean())

############################ below is the code for Q16

#runs = 100
#
#K_12_Ein_win = np.zeros(runs)
#K_12_Eout_win = np.zeros(runs)
#
#for i in range(0,runs):
#    X,Y = create_Points(100)
#    Xo,Yo = create_Points(5000)
#    w9,cluster9,mu9 = train_RBF(X,Y,9,gamma)
#    w12,cluster12,mu12 = train_RBF(X,Y,12,gamma)
#    E_in_9 = test_RBF(X,Y,w9,mu9,gamma)
#    E_in_12 = test_RBF(X,Y,w12,mu12,gamma)
#    E_out_9 = test_RBF(Xo,Yo,w9,mu9,gamma)
#    E_out_12 = test_RBF(Xo,Yo,w12,mu12,gamma)
#    if E_in_12 < E_in_9:
#        K_12_Ein_win[i] = 1
#    if E_out_12 < E_out_9:
#        K_12_Eout_win[i] = 1
#
#print("Proportion 12 wins on E_in: " + str(K_12_Ein_win.mean()))
#print("Proportion 12 wins on E_out: " + str(K_12_Eout_win.mean()))

########################### below is the code for Q17

#runs = 100
#
#K = 9
#
#G_2_Ein_win = np.zeros(runs)
#G_2_Eout_win = np.zeros(runs)
#
#for i in range(0,runs):
#    X,Y = create_Points(100)
#    Xo,Yo = create_Points(5000)
#    w1h,cluster1h,mu1h = train_RBF(X,Y,K,1.5)
#    w2,cluster2,mu2 = train_RBF(X,Y,K,2)
#    E_in_1h = test_RBF(X,Y,w1h,mu1h,gamma)
#    E_in_2 = test_RBF(X,Y,w2,mu2,gamma)
#    E_out_1h = test_RBF(Xo,Yo,w1h,mu1h,gamma)
#    E_out_2 = test_RBF(Xo,Yo,w2,mu2,gamma)
#    if E_in_2 < E_in_1h:
#        G_2_Ein_win[i] = 1
#    if E_out_2 < E_out_1h:
#        G_2_Eout_win[i] = 1
#
#print("Proportion Gamma 2 wins on E_in: " + str(G_2_Ein_win.mean()))
#print("Proportion Gamma 2 wins on E_out: " + str(G_2_Eout_win.mean()))

########################## below is the code for Q18

runs = 1000

K = 9
gamma = 1.5

E_in_vec = np.zeros(runs)

for i in range(0,runs):
    X,Y = create_Points(100)
    w,cluster,mu = train_RBF(X,Y,K,gamma)
    E_in_vec[i] = test_RBF(X,Y,w,mu,gamma)  # in sample error

print("Proportion with no errors: " + str((E_in_vec == 0).sum()/E_in_vec.size))