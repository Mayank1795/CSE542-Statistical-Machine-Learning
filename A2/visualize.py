#%%
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

#%%
sns.set(style = "darkgrid")

data_points = [[-5.01, -8.12, -3.68, -0.91, -0.18, -0.05, 5.35, 2.26],
        [-5.43, -3.48, -3.54, 1.30, -2.06, -3.53, 5.12, 3.22],
        [1.08, -5.52, 1.66, -7.75, -4.54, -0.95, -1.34, -5.31],
        [0.86, -3.78, -4.11, -5.47, 0.50, 3.92, 4.48, 3.42],
        [-2.67, 0.63, 7.39, 6.14, 5.72, -4.85, 7.11, 2.39],
        [4.94, 3.29, 2.08, 3.60, 1.26, 4.36, 7.17, 4.33],
        [-2.51, 2.09, -2.59, 5.37, -4.63, -3.65, 5.75, 3.97],
        [-2.25, -2.13, -6.94, 7.18, 1.46, -6.66, 0.77, 0.27],
        [5.56, 2.86, -2.26, -7.39, 1.17, 6.30, 0.90, -0.43],
        [1.03, -3.33, 4.33, -7.50, -6.32,  -0.31, 3.52, -0.36]]

data = pd.DataFrame(data = data_points)

priors = [0.5, 0.5, 0]

d = 2
mean_all = [ np.zeros((d, 1)) for i in range(0, 2) ]
var_all = [ np.zeros((d, d)) for i in range(0, 2) ]

for i in range(0, 2):
    mean_all[i] = np.mean(data.iloc[:, 3*i:3*(i+1) - 1], axis = 0)
    mean_all[i] = mean_all[i].T
    var_all[i] = np.cov(data.iloc[:, 3*i:3*(i+1) - 1], rowvar = False)

#%%

######## Univariate Normal ##########

def UnivariateNormal(x, m, v):
    
    power_term = (-1/2)*math.pow((x - m),2)*(1/v)
    p = (1/math.sqrt(2*np.pi*v))
    
    return p * math.pow(np.e, power_term)


#%%
    ######## Multivariate Normal ##########

def MultivariateNormal(x, mean_v, var_v, d):
    
    det_cov = abs(np.linalg.det(var_v))
    c = (1/(math.pow(2*np.pi, d/2)*math.sqrt(det_cov)))
    diff =  x - mean_v
    diff_transpose = diff.T
    inv_cov = np.linalg.inv(var_v)
    power_term = (-1/2)*np.dot(diff_transpose, np.dot(inv_cov, diff))
    
    return c * math.pow(np.e, power_term)



#%%

def g(x, ui, sigma, prior):
    e = math.e    
    
    # print(x, ui)
    ui = ui.values.reshape(-1,1)

    v = x - ui
    v_tran = v.T    
 
    if(sigma.ndim == 0):
        det = sigma
        sigma_inv = 1/sigma
        v = sigma_inv * v

    else:
        det = np.linalg.det(sigma)
        sigma_inv = np.linalg.inv(sigma)
        v = np.dot(sigma_inv, v)

    gi_x = (-1/2)*np.dot(v_tran,v) + (-1/2)*math.log(det, e) + math.log(prior, e)
    # print(x.shape, ui.shape, v.shape, v_tran.shape, sigma.shape, prior)
    return gi_x

#%%

def dichotomizer(x):
    global y_ll
    t1 = g(x, mean_all[0], var_all[0], priors[0]) 
    t2 = g(x, mean_all[1], var_all[1], priors[1]) 
    
    if(abs(t1 - t2) < 0.0001):
        return y_ll.append(x)
    else:
        return 

#%%

######### x1 plot ###########
ar = np.ones((20,))
ar[0:10] = 0

w = pd.Series(ar, name = 'class')

z = pd.Series(np.zeros((20,)), name = 'y')

x = data.iloc[:,0]
x = pd.concat([x, data.iloc[:,3]], axis = 0, ignore_index = True) # axis = 0, by row
x.name = 'x'
new_data = pd.concat([x, z, w], axis = 1)

ax = sns.scatterplot(x = 'x', y = 'y', hue = 'class', data = new_data)

ax.set_xlim(left = -10, right = 10)
ax.set_title('Scatter Plot For x1 Feature')


#%%
######### Fit Univariate Normal x1 ######


ar = np.ones((200,))
ar[0:100] = 0

w = pd.Series(ar, name = 'class')

mean_x1 = data.iloc[:,0].mean()
var_x1 = data.iloc[:,0].var()
mean_x2 = data.iloc[:,3].mean()
var_x2 = data.iloc[:,3].var()

pt = np.linspace(-10, 10, num = 100)  # 100 samples

ll_1 = [ UnivariateNormal(x, mean_x1, var_x1) for x in pt]
ll_2 = [ UnivariateNormal(x, mean_x2, var_x2) for x in pt]

prob_1 = pd.Series(ll_1)
prob_2 = pd.Series(ll_2)
x = pd.Series(pt)

x = pd.concat([x, x], axis = 0, ignore_index = True)
x.name = 'x'
prob = pd.concat([prob_1, prob_2], axis = 0, ignore_index = True)
prob.name = 'p(x|wi)'

new_data = pd.concat([x, prob, w], axis = 1)
# print(new_data)
ax = sns.lineplot(x = 'x', y = 'p(x|wi)', hue = 'class', data = new_data)
ax.set_xlim(left = -10, right = 10)
ax.set_title('Density function using x1')

###### Decision boundary #######

decision = []
for i in range(0, len(pt)):
    if(abs(ll_1[i] - ll_2[i]) < 0.001):
        decision.append(pt[i])

# print(decision)
ax.axvline(x = decision[0], color = 'g', linestyle = '--')
ax.axvline(x = decision[1], color = 'g', linestyle = '--')


#%%
########## x1 and x2 ############
ar = np.ones((20,))
ar[0:10] = 0

w = pd.Series(ar, name = 'class')

x = data.iloc[:,0]
x = pd.concat([x, data.iloc[:,3]], axis = 0, ignore_index = True)
x.name = 'x1'

y = data.iloc[:,1]
y = pd.concat([y, data.iloc[:,4]], axis = 0, ignore_index = True)
y.name = 'x2'

new_data = pd.concat([x, y, w], axis = 1)
ax = sns.scatterplot(x = 'x1', y = 'x2', hue = 'class', data = new_data)
print(ax)
ax.set_xlim(left = -10, right = 10)
ax.set_title('Scatter Plot For x1 &  x2 Feature')

########## Decision boundary (x1,x2) ###########

# series transpose to_frame().T
x1_ll = np.linspace(-10, 10, num = 1000)
x2_ll = np.linspace(-10, 10, num = 1000)

data_pt = []

for i in range(0, len(x1_ll)):
    for j in range(0, len(x2_ll)):
        data_pt.append([x1_ll[i], x2_ll[j]])

y_ll = [ ]


for x in data_pt:
    # print(x)
    dichotomizer(np.reshape(np.array(x), (-1,1)))
    
ptx = []
pty = []

ptx = pd.Series([ x[0][0] for x in y_ll], name = 'x1')
pty = pd.Series([ x[1][0] for x in y_ll], name = 'x2')

new_data = pd.concat([ptx, pty], axis = 1)
new_dt = new_data.sort_values(['x1'], ascending = True)

diff_series = new_dt['x1'] - new_dt['x2']
index_loc = diff_series.values.argmax()

sns.lineplot(x = 'x1', y = 'x2', data = new_dt.iloc[0:index_loc,:], ax = ax, color='g')
sns.lineplot(x = 'x1', y = 'x2', data = new_dt.iloc[index_loc+1:,:], ax = ax, color='g')


#%%

####### Fit Bivariate Normal Dist #####

# nd = data.iloc[:, 0:2]
# nd.columns = ['x1', 'x2']
# ax = sns.kdeplot(nd.x1, nd.x2)
# ax.set_title('Bivariate Density plot for class 1')
# sns.jointplot(x="x1", y="x2", data = nd, kind = "kde") # class 0



#%%



#%%
