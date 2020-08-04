#%%
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import json
import scipy.integrate as integrate

sns.set(style = "darkgrid")


#%%

""" Load the dataset """

training_data = pd.read_csv('train.txt', header = None)
test_data = pd.read_csv('test_all.txt', header = None)
test_missing = pd.read_csv('test_missing.txt', header = None)

""" Read risk.json """

with open('risk.json') as jd:
    risk_mat = json.load(jd)

risk_data = pd.DataFrame.from_dict(risk_mat)


training_data.columns = ['x1', 'x2', 'class']
test_data.columns = ['x1', 'x2', 'class']
test_missing.columns = ['x1', 'x2', 'class']

# print(np.isnan(test_missing))

rmv = []

for i in range(0, risk_data.shape[1]):
    risk_m = risk_data.iloc[:,i]
    z12 = risk_m.iloc[0][1]
    z21 = risk_m.iloc[1][0]
    z11 = risk_m.iloc[0][0]
    z22 = risk_m.iloc[1][1]
    
    rmv.append([z21 - z11, z12- z22])

#set priors

prior = []

prior.append(training_data.loc[training_data['class'] == 0].shape[0])
prior.append(training_data.loc[training_data['class'] == 1].shape[0])

prior = [x/sum(prior) for x in prior]

#%%

mean_all = []
var_all = []

mean_org = 0
var_org = 0

""" Set parameters of multivariate dist """
def setParam(data):

    global mean_all, var_all, mean_org, var_org

    mean_all.clear()
    var_all.clear()
    
    mean_org = data.mean(axis = 0)[:-1]
    var_org = np.cov(data, rowvar = False)[:-1,:-1]

    data_0, data_1 = separateData(data)
    mean_all.append( data_0.mean(axis = 0)[:-1])
    mean_all.append( data_1.mean(axis = 0)[:-1])
    
    var_all.append(np.cov(data_0, rowvar = False)[:-1,:-1])
    var_all.append(np.cov(data_1, rowvar = False)[:-1,:-1])

    return 



#%%

""" Visualize the data """

def visualizeData(data, i,j):
    
    if(len(data) > 1):    
        ax = sns.scatterplot(x = 'x1', y = 'x2', hue = 'class', data =  data[0])
        sns.lineplot(x = 'x1', y = 'x2', hue = 'class', ax = ax, data =  data[1])
        
    else:
        sns.scatterplot(x = 'x1', y = 'x2', ax=axes[i,j], hue = 'class', data =  data[0])
    return


#%%
""" Separate the data by class"""

def separateData(data, marg= False):

    if(marg):
        data_0 = data.loc[data['x2'] == np.nan]
        data_1 = data.loc[data['x2'] != np.nan]

        data_0.reset_index(drop = True, inplace = True) # reset index 1,2,3...
        data_1.reset_index(drop = True, inplace = True)
        
    else:
        data_0 = data.loc[data['class'] == 0]
        data_1 = data.loc[data['class'] == 1]

        data_0.reset_index(drop = True, inplace = True) # reset index 1,2,3...
        data_1.reset_index(drop = True, inplace = True)

    return data_0, data_1



#%%

def DecorrelationMatrix(data):

    cov_mat = np.cov(data, rowvar = False) # covar matrix of exisitng data

    eig_vals, eig_vecs = np.linalg.eig(cov_mat) 
    eig_vals = np.sqrt(1/eig_vals) # 1/sqrt(eig_val)
    
    eig_vals_mat = np.diag(eig_vals)
    A = np.dot(eig_vecs, eig_vals_mat)
    A = A.T

    return A


#%%
""" Projection """

def projectData(Aw0, Aw1, data):
    data_0, data_1 = separateData(data)

    data_0 = data_0.T
    data_1 = data_1.T

    for i in range(0, data_0.shape[1]):
        data_0.iloc[:-1, i] = np.dot(Aw0, data_0.iloc[:-1, i])

    for i in range(0, data_1.shape[1]):
        data_1.iloc[:-1, i] = np.dot(Aw1, data_1.iloc[:-1, i])

    data_0 = data_0.T
    data_1 = data_1.T

    new_set = pd.concat([data_0, data_1], ignore_index = True, axis = 0)

    return new_set


#%%
""" Decorrelation function """

def decorrelateData(tr_data, te_data):
    
    data_0, data_1 = separateData(tr_data) # data separted by class

    Aw0 = DecorrelationMatrix(data_0.iloc[:, 0:2])
    Aw1 = DecorrelationMatrix(data_1.iloc[:, 0:2])

    print(Aw0, Aw1)
    new_train_set = projectData(Aw0, Aw1, tr_data)
    new_test_set = projectData(Aw0, Aw1, te_data)

    return new_train_set, new_test_set 

new_train_set, new_test_set = decorrelateData(training_data, test_data)


#%%

f, axes = plt.subplots(2, 2)
f.set_size_inches(9,9)
visualizeData([training_data],0,0)
visualizeData([new_train_set],0,1)
visualizeData([test_data],1,0)
visualizeData([new_test_set],1,1)

plt.tight_layout()
# plt.subplots_adjust(wsp)
axes[0,0].set_title("Training data before decorrelation")
axes[0,1].set_title("Training data after decorrelation")
axes[1,0].set_title("Test data before decorrelation")
axes[1,1].set_title("Test data after decorrelation")
# plt.savefig('/home/mayank/Desktop/SEM2/SML/Assignments/A2/Code/test.png', bbox_inches='tight')

#%%

""" Checking the dist. of training data """
# sns.distplot(training_data.iloc[:,0], kde=False)
# sns.distplot(training_data.iloc[:,1], kde=False)



#%%

def gix(x, ui, sigma, prior):
    
    # print(x.shape, ui.shape)
    v = x - ui
    v_tran = v.T    
    det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    v = np.dot(sigma_inv, v)
    gi_x = (-1/2)*np.dot(v_tran,v) + (-1/2)*math.log(det, np.e) + math.log(prior, np.e)
    
    return gi_x



#%%
def boundaryPointsRisk(data_pt):
    
    all_rk = []

    like_0 = MultivariateNormal(data_pt, mean_all[0], var_all[0])
    like_1 = MultivariateNormal(data_pt, mean_all[1], var_all[1])
    evidence = like_0*prior[0] + like_1*prior[1]

    k0 = like_0/evidence
    k1 = 1 - k0

    for i in range(0, risk_data.shape[1]):    
        if(rmv[i][0]*k0 > rmv[i][1]*k1):
            all_rk.append(0)
        else:
            all_rk.append(1)

    return all_rk    

#%%

def boundaryPoints(x, classify = False):
    # print(x.shape)
    x = x[np.newaxis].T
    m0 = np.array(mean_all[0])[np.newaxis].T
    m1 = np.array(mean_all[1])[np.newaxis].T

    t1 = gix(x, m0 , var_all[0], prior[0]) 
    t2 = gix(x, m1, var_all[1], prior[1]) 
    
    if(classify):
        if(t1 > t2):
            return 0
        else:
            return 1
    else:
        if(abs(t1 - t2) < 0.01):
            return True
        else:
            return False

#%%

""" Multivariate Normal density """

def MultivariateNormal(x, mean_v, var_v):
    
    det_cov = np.linalg.det(var_v)
    c = (1/(2*np.pi*math.sqrt(det_cov)))
    diff =  x - mean_v
    diff_transpose = diff.T
    inv_cov = np.linalg.inv(var_v)
    power_term = (-1/2)*np.dot(diff_transpose, np.dot(inv_cov, diff))
    
    return c * math.pow(np.e, power_term)


#%%

""" Finding decision boundary """ 

def drawDecisionBoundary(data, st, n, pi, pj, risk = False):

    if(st == "train"):
        x1 = np.linspace(-2, 8, n)
        x2 = np.linspace(-3, 13, n)

    elif(st == "new_train"):
        x1 = np.linspace(-5, 3, n)
        x2 = np.linspace(-7, 1, n)

    elif(st == "test"):
        x1 = np.linspace(-2, 8, n)
        x2 = np.linspace(-3, 12, n)

    elif(st == "new_test"):
        x1 = np.linspace(-4, 2, n)
        x2 = np.linspace(-7, 2, n)

    x,y = np.meshgrid(x1, x2)

    pos = np.empty(x.shape + (2,))
    # print(pos.shape)
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    if(risk):
        z = [ np.zeros((n,n)) for i in range(0, risk_data.shape[1])]
        
        for i in range(0, pos.shape[0]):
            for j in range(0, pos.shape[1]):
                data_point = np.array([pos[i,j,0], pos[i,j,1]])            
                all_label = boundaryPointsRisk(data_point)

                for r in range(0, len(all_label)):
                    z[r][i,j] = all_label[r]

        sns.scatterplot(x = 'x1', y = 'x2', ax = axes[0,0], hue = 'class', data =  data)
        axes[0,0].contourf(x, y, z[0], alpha=0.4, linestyles='dashed')

        sns.scatterplot(x = 'x1', y = 'x2', ax = axes[0,1], hue = 'class', data =  data)
        axes[0,1].contourf(x, y, z[1], alpha=0.4, linestyles='dashed')
    
        sns.scatterplot(x = 'x1', y = 'x2', ax = axes[1,0], hue = 'class', data =  data)
        axes[1,0].contourf(x, y, z[2], alpha=0.4, linestyles='dashed')
    
        sns.scatterplot(x = 'x1', y = 'x2', ax = axes[1,1], hue = 'class', data =  data)
        axes[1,1].contourf(x, y, z[3], alpha=0.4, linestyles='dashed')
    
        sns.scatterplot(x = 'x1', y = 'x2', ax = axes[2,0], hue = 'class', data =  data)
        axes[2,0].contourf(x, y, z[4], alpha=0.4, linestyles='dashed')
    
        sns.scatterplot(x = 'x1', y = 'x2', ax = axes[2,1], hue = 'class', data =  data)
        axes[2,1].contourf(x, y, z[5], alpha=0.4, linestyles='dashed')  

    else:
        z = np.zeros((n, n))

        for i in range(0, pos.shape[0]):
            for j in range(0, pos.shape[1]):
                data_point = np.array([pos[i,j,0], pos[i,j,1]])            
                predict_c = boundaryPoints(data_point, True)
                if(predict_c):
                    z[i,j] = 1
                else:
                    z[i,j] = 0
        
        sns.scatterplot(x = 'x1', y = 'x2', ax = axes[pi], hue = 'class', data =  data)
        axes[pi].contourf(x, y, z, alpha=0.4, linestyles='dashed') 
        
    return




#%%

""" Decision boudary """

n = 150

plt.figure()
f, axes = plt.subplots(3, 2)
f.set_size_inches(8,12)
f.set_dpi(100)


setParam(training_data)
# drawDecisionBoundary(test_data,"test", n,0,0)
# drawDecisionBoundary(training_data,"train", n,0,0)
drawDecisionBoundary(test_data, "test", n,0,0, True)  # With all risk
# drawDecisionBoundary(new_test_set, "new_test", n,0,0, True)  # With all risk

# setParam(new_train_set)
# drawDecisionBoundary(new_test_set,"new_test", n,1,0)
# drawDecisionBoundary(new_train_set,"new_train", n,1,0)
# drawDecisionBoundary(new_test_set, "new_test", n,1,0, True) # With all risk

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace = 0.3)

axes[0,0].set_title("Risk 1")
axes[0,1].set_title("Risk 2")

axes[1,0].set_title("Risk 3")
axes[1,1].set_title("Risk 4")

axes[2,0].set_title("Risk 5")
axes[2,1].set_title("Risk 6")


# drawDecisionBoundary(new_train_set, "new_train", n, True) # With all risk

# drawDecisionBoundary(training_data, "train", n, True) # With all risk



#%%

""" Make confusion matrix and find accuracy"""


def confusion_mat(predicted_label, data, ax_no):
    
    actual_label = data.iloc[:,-1]
    
    li = [[0,0],[0,1],[1,0],[1,1],[2,0],[2,1]]

    if(len(predicted_label.columns)>1):

        for k in range(0, risk_data.shape[1]):    
            cm = np.zeros((2,2))
            single_col = predicted_label.iloc[:,k]

            for i in range(0, single_col.shape[0]):
                if(single_col[i] == actual_label[i]):
                    if(single_col[i] == 0):
                        cm[0,0]+=1
                    else:
                        cm[1,1]+=1
                else:
                    if(single_col[i] == 0):
                        cm[0,1]+=1
                    else:
                        cm[1,0]+=1
            
            print('Accuracy : ', np.trace(cm)/sum(sum(cm)))

            # plt.figure()
            hm = pd.DataFrame(cm, index = ['Predict 0', 'Predict 1'], columns = ['Actual 0', 'Actual 1'])
            sns.heatmap(hm, annot = True, ax=axes[li[k][0], li[k][1]], fmt='g', cmap = "YlGnBu")
        
    
    else:
        cm = np.zeros((2,2))
        # print(predicted_label.shape)
        # print(data.shape)
        for i in range(0, predicted_label.shape[0]):
            if(predicted_label[i] == actual_label[i]):
                if(predicted_label[i] == 0):
                    cm[0,0]+=1
                else:
                    cm[1,1]+=1
            else:
                if(predicted_label[i] == 0):
                    cm[0,1]+=1
                else:
                    cm[1,0]+=1
        
        print('Accuracy : ', np.trace(cm)/sum(sum(cm)))
        print('Precision: ', cm[0,0]/(cm[0,0]+cm[0,1]))

        # plt.figure()
        hm = pd.DataFrame(cm, index = ['Predict 0', 'Predict 1'], columns = ['Actual 0', 'Actual 1'])
        sns.heatmap(hm, annot = True, ax=axes[ax_no], fmt='g', cmap = "YlGnBu")
        
        
    return  


#%%
def fpr_tpr(data, check, thres):
    tcm = np.zeros((2, 2))
    
    # print(check, thres)
    for i in range(0, data.shape[0]):
        if(check.iloc[i] > thres):
            if(data.iloc[i, 2] == 0):
                tcm[0, 0]+=1
            else:
                tcm[0, 1]+=1
        else:
            if(data.iloc[i, 2] == 0):
                tcm[1, 0]+=1
            else:
                tcm[1, 1]+=1           

    fpr = tcm[0,1]/(tcm[0,1] + tcm[1,1])
    tpr = tcm[0,0]/(tcm[0,0] + tcm[1,0])

    return np.array([fpr, tpr])

def classify(data, thres = 0, roc = False, risk = False, marg = False):
    
    predict_class = pd.Series(name = 'predict_class')

    if(roc):
        if(risk):
            return fpr_tpr(data, learned_risk, thres)
        else:     
            return fpr_tpr(data, learned_pr, thres)
        
    else:    
        if(risk):
            predict_class = pd.DataFrame(columns=['r1', 'r2', 'r3', 'r4', 'r5', 'r6'])
            for i in range(0, data.shape[0]):
                predict_class_label = boundaryPointsRisk(np.array(data.iloc[i,0:2]))
                predict_class = predict_class.append(pd.Series(predict_class_label, index = ['r1','r2', 'r3', 'r4', 'r5', 'r6']), ignore_index = True)

            return predict_class
        else:
            predict_class = pd.Series(name = 'predict_class')    
            # print(predict_class)
            for i in range(0, data.shape[0]):
                predict_class = predict_class.set_value(i, boundaryPointsRisk(np.array(data.iloc[i,0:2])))
                
    return predict_class

#%%

# 2. Train bayes 

f, axes = plt.subplots(3, 2)
f.set_size_inches(6,9)
f.set_dpi(100)

setParam(new_train_set)
# confusion_mat(classify(new_test_set, risk=True), new_test_set, 0) 

# setParam(new_train_set)
# confusion_mat(classify(new_test_set), new_test_set, 1)

# setParam(training_data)
# confusion_mat(classify(training_data), training_data, 0)

# setParam(new_train_set) 
# confusion_mat(classify(new_train_set), new_train_set, 1)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace = 0.3)

axes[0,0].set_title("Risk 1")
axes[0,1].set_title("Risk 2")

axes[1,0].set_title("Risk 3")
axes[1,1].set_title("Risk 4")

axes[2,0].set_title("Risk 5")
axes[2,1].set_title("Risk 6")

# plt.tight_layout()
# plt.subplots_adjust(wspace=0.4, top=0.7)
# axes[0].set_title("Train data (Correlated)")
# axes[1].set_title("Train data (Decorrelated)")

#%%

#### ROC on training set #####

def learn_pr_roc(data):

    global learned_pr

    learned_pr = pd.DataFrame(columns = learned_pr.columns)

    """ P(wi|x) """
    for i in range(0, data.shape[0]):
        likeli_0 = MultivariateNormal(data.iloc[i,0:2].values, mean_all[0], var_all[0])
        likeli_1 = MultivariateNormal(data.iloc[i,0:2].values, mean_all[1], var_all[1])

        evidence = likeli_0*prior[0] + likeli_1*prior[1]

        p0 = (likeli_0*prior[0])/evidence
        p1 = 1 - p0

        learned_pr = learned_pr.append(pd.Series(np.array([p0, p1]), index = ['class0', 'class1']), ignore_index = True)
    
    return


#%%

def learn_pr_risk(data, risk_m):
    global learned_risk

    learned_risk = pd.Series(name = learned_risk.name)

    z12 = risk_m.iloc[0][1]
    # z21 = risk_m.iloc[1][0]
    z11 = risk_m.iloc[0][0]
    # z22 = risk_m.iloc[1][1]
    
    """ Calculate R(lamda|x) """

    for i in range(0, data.shape[0]):
        likeli_0 = MultivariateNormal(data.iloc[i,0:2].values, mean_all[0], var_all[0])
        likeli_1 = MultivariateNormal(data.iloc[i,0:2].values, mean_all[1], var_all[1])

        evidence = likeli_0*prior[0] + likeli_1*prior[1]
    
        pw0 = (likeli_0*prior[0])/evidence
        pw1 = 1 - pw0

        r0 = z11 * pw0  +  z12 * pw1
        # r1 = z21 * pw0  +  z22 * pw1   

        learned_risk = learned_risk.set_value(i, r0)
        
    return


#%%

def normalizeRisk():
    """ Normalized risk fun values"""
    
    global learned_risk

    # print(learned_risk)  
    rk = learned_risk.values
    Max = np.max(rk)
    Min = np.min(rk)
    # print(Max - Min)

    for i in range(0, rk.shape[0]):
        rk[i] = (rk[i] - Min)/(Max - Min)
    
    learned_risk = pd.Series(rk)
    learned_risk.name = 'R(r0 | x)'

    # print(learned_risk)
    return


def drawROC(data, st, pi, risk = False):

    thresholds = np.linspace(0,1, num = 100)

    if(not risk):
        learn_pr_roc(data)

        roc_points = pd.DataFrame(columns = ['FPR', 'TPR'])

        for threshold in thresholds:
            roc_pt = classify(data, threshold, True)     
            roc_points = roc_points.append(pd.Series(roc_pt, index = ['FPR', 'TPR']), ignore_index = True)

        roc_points = roc_points.sort_values(['FPR'], ascending = True)
        ax = sns.lineplot(x='FPR', y='TPR', data = roc_points)
        ax.set_title('ROC of '+st)
    else:

        roc_points = pd.DataFrame(columns = ['FPR', 'TPR', 'Risk'])

        for i in range(0, risk_data.shape[1]):
                
            rm = risk_data.iloc[:, i]
                
            learn_pr_risk(data, rm)
            normalizeRisk()

            for threshold in thresholds:
                roc_pt = classify(data, threshold, True, True)    
                roc_pt = np.append(roc_pt, (i+1)) 
                roc_points = roc_points.append(pd.Series(roc_pt, index = ['FPR', 'TPR', 'Risk']), ignore_index = True)

        roc_points = roc_points.sort_values(['FPR'], ascending = True)
        sns.lineplot(x='FPR', y='TPR', hue='Risk', ax = axes[pi], data = roc_points, legend='full')
        roc_points = pd.DataFrame(columns = roc_points.columns)

    return
        
#%%

learned_pr = pd.DataFrame(columns = ['class0', 'class1'])
learned_risk = pd.Series(name = 'R(r0 | x)')

#%%
### ROC without decorrelation ###

f, axes = plt.subplots(1,2)
f.set_size_inches(6,9)
f.set_dpi(100)

plt.figure()

setParam(training_data)
drawROC(test_data, "test data",0, True) # With all risk
# drawROC(test_data, "test data", 0)

# setParam(new_train_set)
# drawROC(new_test_set, "test(new) data",1, True) # With all risk

plt.tight_layout()
plt.subplots_adjust(wspace=0.6)
axes[0].set_title("ROC on test data (Correlated) ")
axes[1].set_title("ROC on test data (Decorrelated)")


# drawROC(test_data, "test data")  
# drawROC(test_data, "test data", True)                                  


### ROC with decorrelation ###
# plt.figure()
# drawROC(new_train_set, "new test data")
# drawROC(new_test_set, "new test data", True) 
#                                           # With all risk


#%%

def assignLabel(x):
     # print(x.shape)
    x = x[np.newaxis].T
    m0 = np.array(mean_all[0])[np.newaxis].T
    m1 = np.array(mean_all[1])[np.newaxis].T

    t1 = gix(x, m0 , var_all[0], prior[0]) 
    t2 = gix(x, m1, var_all[1], prior[1]) 
    
    if(t1 > t2):
        return 0
    else:
        return 1

""" Marginalisation Question"""

setParam(training_data)

def marginalClassifier(data):
    pass
    predict_class = pd.Series(name = 'predict_class')
    
    for i in range(0, data.shape[0]):
        predict_class = predict_class.set_value(i, classify( data,0, False, False, True))
        
    return predict_class



marginalClassifier(test_missing)



#%%
