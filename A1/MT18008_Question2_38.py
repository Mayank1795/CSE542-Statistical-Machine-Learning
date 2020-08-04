from fashion_mnist.utils.mnist_reader import load_mnist
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
import pickle


image, label = load_mnist('./fashion_mnist/data/fashion/')
all_img = image.copy()
all_lbl = label.copy()

t_images, t_labels = load_mnist('./fashion_mnist/data/fashion/',kind='t10k')
all_test_images = t_images.copy()
all_test_labels = t_labels.copy()

cloth_type = ['Trouser', 'Pullover']
N = 2 # no. of classes
cm_2 = np.zeros((N,N))

classes = list(range(N))
unq, cnt = np.unique(all_lbl, return_counts = True) # no. of data points of a class in training set
unq1, cnt1 = np.unique(all_test_labels, return_counts = True) # no. of data points in test set of the same class

class_priors = cnt/len(all_lbl)
# print(cnt1[0])
# print(len(all_img[0]))

images = np.zeros((cnt[0]*N, len(all_img[0])), dtype='int')
labels = np.zeros((cnt[0]*N),dtype='int')
test_images = np.zeros((cnt1[0]*N, len(all_img[0])),dtype='int')
test_labels = np.zeros((cnt1[0]*N),dtype='int')

flag = 0
for i in range(0, len(all_img)):
    if(all_lbl[i] == 1) or (all_lbl[i] == 2):
        images[flag,:] = all_img[i,:]
        labels[flag] = all_lbl[i]-1
        flag+=1
        

fg = 0
for i in range(0, len(all_test_images)):
    if(all_test_labels[i] == 1) or (all_test_labels[i] == 2):
        test_images[fg,:] = all_test_images[i,:]
        test_labels[fg] = all_test_labels[i]-1
        fg+=1

images[images < 127] = 0
images[images ==127] = np.random.randint(0,2)
images[images > 127] = 1

test_images[test_images < 127] = 0
test_images[test_images ==127] = np.random.randint(0,2)
test_images[test_images > 127] = 1


dist = [[] for c in range(0,N)]

class_index = [[] for c in range(0,N)] # 10 x 6k

for i in range(0, len(images)):
    class_index[labels[i]].append(i)

# print(class_index)

for j in range(0, len(images[0])):
    for c in range(0,N):
        s = 0
        for i in class_index[c]:
            if(images[i, j] == 1):
                s+=1
        dist[c].append((float)(s/cnt[0]))

# test index needed for finding roc points
test_index = [[]]*N  

for i in range(0, len(test_images)):
    test_index[test_labels[i]].append(i)


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
    global cm_2
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
        
        my_label = likelihood.index(max(likelihood))
        predicted_labels[counter] = my_label
        true_label = imgToLabel(img)   # my labels 0(trouser) +ve, 1 -ve
        cm_2[my_label, true_label]+=1
        counter+=1
        likelihood.clear()
        
    return predicted_labels
    


predicted_ls = NaiveBayes(test_images)
print(cm_2)
print('Precision: ',cm_2[0,0]/(cm_2[0,0]+cm_2[0,1]) )
print('Recall: ', cm_2[0,0]/(cm_2[0,0] + cm_2[1,0]) )


# Accuracy 2 class
print(sum(np.diag(cm_2))/(sum(sum(cm_2))))

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

                         
all_ll = np.zeros((N, test_images.shape[0]))

for i in range(0, N):
    all_ll[i,:], actual_label[i,:] = NaiveBayesROC(test_images, i)
    all_thresholds[i,:] = sorted(all_ll[i,:])


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

for i in range(0, len(all_ll)):
    for j in range(0, len(all_ll[0])):
        # ith class is pos.
        t = all_thresholds[i,j]
        
        y,x = classify(i, t, all_ll[i,:], actual_label[i,:]) # returns  tpr, fpr
        x_fpr[i,j] = x
        y_tpr[i,j] = y


for i in range(0, N):
    plt.plot(x_fpr[i,:], y_tpr[i,:], label=cloth_type[i])

plt.plot([0,1],[0,1],'--c')
plt.title('Receiver Operating Characteristic Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()





