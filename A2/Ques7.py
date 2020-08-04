#%%
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from scipy.stats import norm
from scipy import integrate

sns.set(style = "darkgrid")
#%%
 ######## Univariate Normal ##########

def UnivariateNormal(x, m, v):
    
    power_term = (-1/2)*math.pow((x - m),2)*(1/v)
    p = (1/math.sqrt(2*np.pi*v))
    
    return p * math.pow(np.e, power_term)


#%%

def generateSamples(u0, v0, u1, v1):
        
    sample_0 = np.random.normal(u0, math.sqrt(v0), size = N)
    sample_0 = np.sort(sample_0)
    sample_1 = np.random.normal(u1, math.sqrt(v1), size = N)
    sample_1 = np.sort(sample_1)

    sample_0_y = [ UnivariateNormal(i, u0, v0) for i in sample_0]
    sample_1_y = [ UnivariateNormal(i, u1, v1) for i in sample_1]

    plt.plot(sample_0, sample_0_y)
    plt.plot(sample_1, sample_1_y)
    return sample_0, sample_1

def errorBound(u0, v0, u1, v1):
    """Bhattacharya bound"""        
    std_avg = (math.sqrt(v0) + math.sqrt(v1))/2
    std_pr = math.sqrt(math.sqrt(v0) * math.sqrt(v1))

    val = (1/8) * (u1 - u0) * (1/std_avg) * (u1 - u0) + (1/2)*math.log((std_avg/std_pr), np.e)

    return math.sqrt(priors[0]*priors[1])*math.pow(np.e,-val)



#%%

def g(x, ui, sigma, prior):  

    gi_x = (-1/2)*(x - ui)*(1/sigma)*(x - ui) + (-1/2)*math.log(sigma, np.e) + math.log(prior, np.e)
  
    return gi_x
#%%

def dichotomizer(x, k):
    
    t1 = g(x, u0, v0, priors[0]) 
    t2 = g(x, u1, v1, priors[1]) 
    
    # print(t1,':',t2)
    if(t1 > t2):
        if(k == 0):
            cm[0,0]+=1
        else:
            cm[0,1]+=1
        return 0
    else:
        if(k == 0):
            cm[1,0]+=1
        else:
            cm[1,1]+=1
        return 1
        
#%%

def classify(s0, s1):

    predicted_0 = []
    predicted_1 = []

    for i in s0:
        predicted_0.append(dichotomizer(i, 0))
        
    for j in s1:
        predicted_1.append(dichotomizer(j, 1))


#%%

priors = [1/2, 1/2]

u0 = -0.5
v0 = 2 #variance

u1 = 0.5
v1 = 2 #variance

N = 100 # no. of samples needed
cm = np.zeros((2,2))


##### 7.b and 7.c ######


# Using decision boundary equation to find the  (v0 == v1)

x0 = (1/2) * (u0 + u1) - (v0/abs(u0 - u1)) * math.log((priors[0]/priors[1])*(abs(u0 - u1)), np.e)

print(x0)

# true_err = 1 - (1/2)*(math.erf((x0 - u0)/(math.sqrt(2)*v0))) + (1/2)*(math.erf((x0 - u1)/(math.sqrt(2)*v1)))
# true_err = 1- norm.cdf((x0 -u0)/math.sqrt(v0)) + norm.cdf((x0 - u1)/math.sqrt(v1))
def fin1(x):
    power_term = (-1/2)*math.pow((x - u0),2)*(1/v0)
    p = (1/math.sqrt(2*np.pi*v0))
    
    return p * math.pow(np.e, power_term)

def fin2(x):
    power_term = (-1/2)*math.pow((x - u1),2)*(1/v1)
    p = (1/math.sqrt(2*np.pi*v1))
    
    return p * math.pow(np.e, power_term)


true_err = integrate.quad(fin2, np.NINF, 0) + integrate.quad(fin1, 0, np.PINF)

print('True error rate: ', true_err)

##### 7.d ######

sp_0, sp_1 = generateSamples(u0, v0, u1, v1)
classify(sp_0, sp_1)

print('Error rate : ', (cm[0,1] + cm[1,0])/sum(sum(cm)))
print('Bhattacharya bound is : ', errorBound(u0, v0, u1, v1))
#%%


#%%


#%%
