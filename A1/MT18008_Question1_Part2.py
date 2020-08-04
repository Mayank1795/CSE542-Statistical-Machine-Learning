
from fashion_mnist.utils.mnist_reader import load_mnist
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt

image, label = load_mnist('./fashion_mnist/data/fashion/')
images = image.copy()
labels = label.copy()

t_images, t_labels = load_mnist('./fashion_mnist/data/fashion/',kind='t10k')
test_images = t_images.copy()
test_labels = t_labels.copy()

images[images < 127] = 0
images[images ==127] = np.random.randint(0,2)
images[images > 127] = 1

test_images[test_images < 127] = 0
test_images[test_images ==127] = np.random.randint(0,2)
test_images[test_images > 127] = 1


classes = list(range(10))
unq, cnt = np.unique(labels, return_counts = True)
class_priors = cnt/len(labels)
print(cnt[0])
    

cloth_type = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
N = 10 # no. of classes
cm_10 = np.zeros((N,N))

dist = [[] for c in range(0, N)]
llh = np.ones((len(test_images), N))





# Write to dist.pickle
# Do not run more than once!

# class_index = [[] for c in range(0,N)] # 10 x 6k

# for i in range(0, len(images)):
#     class_index[labels[i]].append(i)

# for j in range(0, len(images[0])):
#     for c in range(0,N):
#         s = 0
#         for i in class_index[c]:
#             if(images[i, j] == 1):
#                 s+=1
#         dist[c].append((float)(s/cnt[0]))

# # test index needed for finding roc points
# test_index = [[]]*N  

# for i in range(0, len(test_images)):
#     test_index[test_labels[i]].append(i)

# pk = open('dist.pickle', 'wb')
# pickle.dump(dist, pk)
# pk.close()



# Get dist from pickle 
pk = open('dist.pickle', 'rb')
dist= pickle.load(pk)


def get_roc_points(ans, sol):

    tp = 0 # trouser +ve 0 and pullover -ve 1
    tn = 0
    fp = 0
    fn = 0
    for i in range(0, len(sol)): 
        if(ans[i] == sol[i]):
            if(ans[i] == 0):
                tp+=1
            else:
                tn+=1
        else:
            if(ans[i] == 0) and (sol[i] == 1): #fp
                fp+=1
            else:
                fn+=1
    
#     print("CM:",tp,fp,fn,tn)
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    return tpr, fpr


def imgToLabel(img):
    # Get true label for any image (0,1)
    pos = np.where(np.all(test_images==img,axis=1))
    return test_labels[int(pos[0][0])]


def  NaiveBayes(test_images):
    #return predicted_labels
    global cm_10, llh
    
    predicted_labels = np.zeros((test_images.shape[0]))
    
    counter = 0    
    for img in test_images: # every image
        likelihood = [1]*N
        for c in range(0, N):  # every class
            for j in range(0, len(img)): # every feature
                if(img[j] == 1):
                    likelihood[c]*=dist[c][j]
                else:
                    likelihood[c]*=1-dist[c][j]
        
        llh[counter,:] = likelihood 
        
        my_label = likelihood.index(max(likelihood))
        predicted_labels[counter] = my_label
        true_label = imgToLabel(img)
        cm_10[my_label, true_label]+=1
        counter+=1
        likelihood.clear()
    # Write to llh.pickle
    
#     pk = open('llh.pickle', 'wb')
#     pickle.dump(llh, pk)
#     pk.close()
    
    return predicted_labels
    


def  NaiveBayesLLH(test_images, llh):
    #return predicted_labels
    global cm_10
    cm_10 = np.zeros((N,N))
    
    predicted_labels = np.zeros((test_images.shape[0]))
    
    counter = 0    
    for img in test_images: # every image 
        my_label = llh[counter,:].argmax(axis=0)
        predicted_labels[counter] = my_label
        true_label = imgToLabel(img)
        cm_10[my_label, true_label]+=1
        counter+=1
        
    # Write to llh.pickle
    
    pk = open('llh.pickle', 'wb')
    pickle.dump(llh, pk)
    pk.close()
    
    return predicted_labels
    


# Write to llh.pickle
# Do not run more than once!

# predicted_ls = NaiveBayes(test_images)
# print(cm_10)


# Read llh.pickle
pk = open('llh.pickle', 'rb')
llh = pickle.load(pk)
predicted_ls = NaiveBayesLLH(test_images, llh)
print(cm_10)



# Accuracy 10 class
print("Accuracy : ",sum(np.diag(cm_10))/(sum(sum(cm_10))))
# Precision and Recall for all classes 
precision_i = []
recall_i = []

for i in range(0, len(cm_10)):
    true_p = cm_10[i,i]
    false_p = sum(cm_10[i,:]) - true_p
    false_n = sum(cm_10[:,i]) - true_p
    precision_i.append(true_p/(true_p + false_p))
    recall_i.append(true_p/(true_p + false_n))

# Avereage P & R
print("Precision of 10 classes: ", precision_i)
print("Recall of 10 classes: ", recall_i)
    


def findLabel(img, i):
    # Get true label for any image (0,1)
    pos = np.where(np.all(test_images==img,axis=1))
    k = test_labels[int(pos[0][0])]
    if(k == i):
        return 0
    else:
        return 1



def  NaiveBayesROC(test_images, i):
    
    ps = []
    org_label = []  
    
    for img in test_images:
        likeli = 1

        for j in range(0, len(img)):
            if(img[j] == 1):
                likeli*=dist[i][j]
            else:
                likeli*=1-dist[i][j]

        ps.append(likeli)
        org_label.append(findLabel(img, i))  
    return ps, org_label
    

all_thresholds = np.empty((N, test_images.shape[0])) #N binary classifier
actual_label = np.empty((N, test_images.shape[0]))
predicted_label = np.empty((N, test_images.shape[0]))


# # Write to pickle file
                         
# all_ll = np.zeros((N, test_images.shape[0]))

# for i in range(0, N):
#     all_ll[i,:], actual_label[i,:] = NaiveBayesROC(test_images, i)
#     all_thresholds[i,:] = sorted(all_ll[i,:])


# pk = open('all_thresholds.pickle', 'wb')
# pickle.dump(all_thresholds, pk)
# pk.close()

# pk = open('actual_label.pickle', 'wb')
# pickle.dump(actual_label, pk)
# pk.close()

# pk = open('predicted_label.pickle', 'wb')
# pickle.dump(predicted_label, pk)
# pk.close()


# Reading from pickles

pk = open('all_thresholds.pickle', 'rb')
all_thresholds = pickle.load(pk)

pk = open('actual_label.pickle', 'rb')
actual_label = pickle.load(pk)

pk = open('predicted_label.pickle', 'rb')
predicted_label = pickle.load(pk)



def classify(c, t, all_ps, sol):
    global predicted_label
    
    for i in range(0, len(all_ps)):
        if(all_ps[i] > t):
            predicted_label[c,i] = 0
        elif(all_ps[i] < t):
            predicted_label[c,i] = 1
        else:
            predicted_label[c,i] = np.random.randint(0,2)
    return get_roc_points(predicted_label[c,:], sol)
    
    


x_fpr = np.empty((N, test_images.shape[0]))
y_tpr = np.empty((N, test_images.shape[0]))


# # Write ROC pts pickle file

# for i in range(0, len(all_ll)):
#     for j in range(0, len(all_ll[0])):
#         # ith class is pos.
#         t = all_thresholds[i,j]
        
#         y,x = classify(i, t, all_ll[i,:], actual_label[i,:]) # returns  tpr, fpr
#         x_fpr[i,j] = x
#         y_tpr[i,j] = y

# pk = open('tpr.pickle', 'wb')
# pickle.dump(y_tpr, pk)
# pk.close()

# pk = open('fpr.pickle', 'wb')
# pickle.dump(x_fpr, pk)
# pk.close()



# Reading ROC pts


pk = open('fpr.pickle', 'rb')
x_fpr = pickle.load(pk)

pk = open('tpr.pickle', 'rb')
y_tpr = pickle.load(pk)



for i in range(0, 10):
    plt.plot(x_fpr[i,:], y_tpr[i,:], label=cloth_type[i])
    
plt.title('Receiver Operating Characteristic Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# CMC Curve
rank = [0]*N # sum(rank) == total no. of test points

def findRankAccuracy(test_images):
    global rank
        
      
    for img in test_images: # every image
        likelihood = np.ones((N,2))   # col.1 = class 0 to 9, col.2 = ll
        likelihood[:,0] = list(range(10))
        likelihood[:,1] = [1]*N
        
        for c in range(0, N):  # every class
            for j in range(0, len(img)): # every feature
                if(img[j] == 1):
                    likelihood[c,1]*=dist[c][j]
                else:
                    likelihood[c,1]*=1-dist[c][j]
        
        llh = likelihood[likelihood[:,1].argsort()[::-1]]
        true_label = imgToLabel(img)
        
        
        for i in range(0, len(llh)):
            #check with true label
            if(true_label == llh[i,0]):
                rank[i]+=1
    
    for i in range(1,len(rank)):
        rank[i] = rank[i] + rank[i-1]
    y = [ i/(test_images.shape[0]) for i in rank]
    return y
   


# y_cmc = findRankAccuracy(test_images)
x_cmc = list(range(10))

# pk = open('y_cmc.pickle', 'wb')
# pickle.dump(y_cmc, pk)
# pk.close()

pk = open('y_cmc.pickle', 'rb')
y_cmc = pickle.load(pk)

plt.xlabel('Rank')
plt.ylabel('Rank Recognition Rate')
plt.title('Cumulative Match Curve')
plt.plot(x_cmc, y_cmc)


# get_roc_points(predicted_labels, test_labels)
