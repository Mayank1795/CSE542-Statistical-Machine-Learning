
#%%
import numpy as np 
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style = "darkgrid")

#%%

data_points = [[-5.01, -8.12, -3.68, -0.91, -0.18, -0.05, 5.35, 2.26, 8.13],
                [-5.43, -3.48, -3.54, 1.30, -2.06, -3.53, 5.12, 3.22, -2.66],
                [1.08, -5.52, 1.66, -7.75, -4.54, -0.95, -1.34, -5.31, -9.87],
                [0.86, -3.78, -4.11, -5.47, 0.50, 3.92, 4.48, 3.42, 5.19],
                [-2.67, 0.63, 7.39, 6.14, 5.72, -4.85, 7.11, 2.39, 9.21],
                [4.94, 3.29, 2.08, 3.60, 1.26, 4.36, 7.17, 4.33, -0.98],
                [-2.51, 2.09, -2.59, 5.37, -4.63, -3.65, 5.75, 3.97, 6.65],
                [-2.25, -2.13, -6.94, 7.18, 1.46, -6.66, 0.77, 0.27, 2.41],
                [5.56, 2.86, -2.26, -7.39, 1.17, 6.30, 0.90, -0.43, -8.71],
                [1.03, -3.33, 4.33, -7.50, -6.32,  -0.31, 3.52, -0.36, 6.43]]

data = np.array(data_points, dtype=float)
 

 #%%

#%%

def setParam(n):
    global mean_vecs, sigmas

    if(n == 1):
        for i in range(0, c):
            mean_vecs[i] = np.mean(data[:, 3*i])
            sigmas[i] = np.var(data[:, 3*i])
    
    elif(n == 2):
        for i in range(0, c):
            mean_vecs[i] = np.mean(data[:, 3*i:3*(i+1) - 1], axis = 0)
            mean_vecs[i] = np.reshape(mean_vecs[i], (-1,1))
            sigmas[i] = np.cov(data[:, 3*i:3*(i+1) - 1], rowvar = False)

    elif(n == 3):
        for i in range(0, c):
            mean_vecs[i] = np.mean(data[:, 3*i:3*(i+1)], axis = 0)
            mean_vecs[i] = np.reshape(mean_vecs[i], (-1,1))
            sigmas[i] = np.cov(data[:, 3*i:3*(i+1)], rowvar = False)
   
    return
        

#%%

def g(x, ui, sigma, prior):  
    
    v = x - ui
    
    if(sigma.ndim == 0):
        gi_x = (-1/2)*v[0][0]*(1/sigma)*v[0][0] + (-1/2)*math.log(sigma, math.e) + math.log(prior, math.e)

    else:
        v_tran = v.T    
        det = np.linalg.det(sigma)
        sigma_inv = np.linalg.inv(sigma)
        v = np.dot(sigma_inv, v)
        gi_x = (-1/2)*np.dot(v_tran,v) + (-1/2)*math.log(det, np.e) + math.log(prior, np.e)
    
    # print(x.shape, ui.shape, v.shape, sigma.shape)
    return gi_x
    

#%%

def dichotomizer(x):
    global result
    t1 = g(x, mean_vecs[0], sigmas[0], priors[0]) 
    t2 = g(x, mean_vecs[1], sigmas[1], priors[1]) 
    
    # print(t1,':',t2)
    if(t1 > t2):
        return 0
    else:
        return 1   


#%%

def classify(n):
    nr = data.shape[0]
    predicted_labels = np.zeros((nr, 2))

    if(n == 1):
        for i in range(0, nr):    
            x_1 = data[i, 0].reshape(-1,1)
            x_2 = data[i, 3].reshape(-1,1)
            # print(x_1, x_2)
            predicted_labels[i, 0] = dichotomizer(x_1)
            predicted_labels[i, 1] = dichotomizer(x_2)

    elif(n == 2):
        for i in range(0, nr):    
            x_1 = data[i, 0:2].reshape(-1,1)
            x_2 = data[i, 3:5].reshape(-1,1)
            predicted_labels[i, 0] = dichotomizer(x_1)
            predicted_labels[i, 1] = dichotomizer(x_2)
        
    elif(n == 3):       
        for i in range(0, nr):    
            x_1 = data[i, 0:3].reshape(-1,1)
            x_2 = data[i, 3:6].reshape(-1,1)
            predicted_labels[i, 0] = dichotomizer(x_1)
            predicted_labels[i, 1] = dichotomizer(x_2)
        

    return predicted_labels




#%%

d = 3 # no. of dimensions
c = 2  # no. of classes

# priors = np.zeros((1, c))
priors = [0.5, 0.5, 0]

mean_vecs = [ np.zeros((d, 1)) for i in range(0, c) ]
sigmas = [ np.zeros((d, d)) for i in range(0, c) ]

result = []

def findError(p):
    setParam(p)
    predict1 = classify(p)
    # print(predict1)

    unq1, cnt1 = np.unique(predict1[:, 0], return_counts = True)  # check no. of points miscalssified of class 0
    unq2, cnt2 = np.unique(predict1[:, 1], return_counts = True)  # check no. of points miscalssified of class 1

    print(unq1, cnt1)
    print(unq2, cnt2)
    print('Empirical training error : ', (cnt1[1] + cnt2[0])/(2*data.shape[0]))

    if(p==1):

        k1 = mean_vecs[1] - mean_vecs[0]
        sg = (sigmas[0] + sigmas[1])/2
        sg_inv = 1/sg
       
        val = (1/8) * np.dot(k1, np.dot(sg_inv, k1)) + (1/2)*math.log(sg/math.sqrt(sigmas[0]*sigmas[1]), math.e)

    else:
        k1 = mean_vecs[1] - mean_vecs[0]
        k = np.reshape(mean_vecs[1] - mean_vecs[0], (-1, p))
        sg = (sigmas[0] + sigmas[1])/2
        sg_inv = np.linalg.inv(sg)
        det0 = np.linalg.det(sigmas[0])
        det1 = np.linalg.det(sigmas[1])
        det2 = np.linalg.det(sg)

        val = (1/8) * np.dot(k, np.dot(sg_inv, k1)) + (1/2)*math.log(det2/math.sqrt(det0*det1), math.e)

    print(math.sqrt(priors[0]*priors[1])*math.pow(math.e,-val))

findError(d)





#%%


#%%


  #%%


#%%


#%%
