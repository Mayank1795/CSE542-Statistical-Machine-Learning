from fashion_mnist.utils.mnist_reader import load_mnist
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
import math
import pickle

image, label =  load_mnist('./fashion_mnist/data/mnist/')
all_img = image.copy()
all_lbl = label.copy()


t_images, t_labels = load_mnist('./fashion_mnist/data/mnist/',kind='t10k')
all_test_images = t_images.copy()
all_test_labels = t_labels.copy()

digit_type = [1, 8] # pos 1, negt 8

N = 2 # no. of classes
cm_2 = np.zeros((N,N))

classes = list(range(N))
unq, cnt = np.unique(all_lbl, return_counts = True) # no. of data points of a class in training set
unq1, cnt1 = np.unique(all_test_labels, return_counts = True) # no. of data points in test set of the same class
print(unq, cnt)
print(unq1, cnt1)
class_priors = cnt/len(all_lbl)  # cnt : No. of data points of each digit
total_train_points = cnt[digit_type[0]] + cnt[digit_type[1]]
total_test_points = cnt1[digit_type[0]] + cnt1[digit_type[1]]

print('Total no. of digits(',digit_type[0], 'or',digit_type[1],' ) in training data: ', total_train_points)
print('Total no. of digits(',digit_type[0], 'or',digit_type[1],' ) test data: ', total_test_points)

def digitTo(a):
    if a == digit_type[0]:
        return 0
    if a == digit_type[1]:
        return 1

images = np.zeros((total_train_points, len(all_img[0])), dtype='int')
labels = np.zeros((total_train_points),dtype='int')
test_images = np.zeros((total_test_points, len(all_img[0])),dtype='int')
test_labels = np.zeros((total_test_points),dtype='int')

flag = 0
for i in range(0, len(all_img)):
    if(all_lbl[i] == digit_type[0]) or (all_lbl[i] == digit_type[1]):
        images[flag,:] = all_img[i,:]
        labels[flag] = digitTo(all_lbl[i])
        flag+=1
        

fg = 0
for i in range(0, len(all_test_images)):
    if(all_test_labels[i] == digit_type[0]) or (all_test_labels[i] == digit_type[1]):
        test_images[fg,:] = all_test_images[i,:]
        test_labels[fg] = digitTo(all_test_labels[i])
        fg+=1

images[images < 127] = 0
images[images ==127] = np.random.randint(0,2)
images[images > 127] = 1

test_images[test_images < 127] = 0
test_images[test_images ==127] = np.random.randint(0,2)
test_images[test_images > 127] = 1


new_labels = labels.reshape((-1,1))
new_test_labels = test_labels.reshape((-1,1))

train_data = np.append(images, new_labels, axis=1)
test_data = np.append(test_images, new_test_labels, axis=1)


def buildDist(t_set, t_label):
    dist = [[] for i in range(N)]
    
    class_index = [[] for i in range(N)] 
        
    for i in range(0, len(t_label)):
        class_index[int(t_label[i])].append(i) 
  
    uq, ct = np.unique(t_label, return_counts = True) # no. of data points of a class in training set
    
    
    print(uq, ct)
    uq = uq.tolist()
      
    pr1 = ct[0]/len(t_set)

    for j in range(0, len(t_set[0])):
        for c in range(0,N):
            s = 0
            for i in class_index[c]:
                if(t_set[i, j] == 1):
                    s+=1
            dist[c].append((float)(s/ct[c]))
  
    return dist, pr1     

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

def NBClassifier(v_set, v_label, current_dist, prior_1):
    global cm_2
    predicted_labels = np.zeros((v_set.shape[0]))
    
    
    counter = 0    
    for img in v_set: # every image
        likelihood = [1]*N
        for c in range(0, N):  # every class
            for j in range(0, len(img)): # every feature
                if(img[j] == 1):
                    likelihood[c]*=current_dist[c][j]
                else:
                    likelihood[c]*=1-current_dist[c][j]
          
            if(c == 0):
                likelihood[c]*= prior_1
            else:
                likelihood[c]*= 1-prior_1
      
#         print(likelihood)
        my_label = likelihood.index(max(likelihood))
        predicted_labels[counter] = my_label
        true_label = v_label[counter]
#         print(my_label, int(true_label))
        cm_2[my_label, int(true_label)]+=1
        counter+=1
        likelihood.clear()
        
    return predicted_labels

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

def findLabel(final_test, img, final_label, i):
    # Get true label for any image (0,1)
    pos = np.where(np.all(final_test==img,axis=1))
    k = final_label[int(pos[0][0])]
    if(k == i):
        return 0
    else:
        return 1

def  NaiveBayesROC(final_dist, final_prior, final_test, final_label, i):
    
    ps = []
    org_label = []  
    
    for img in final_test:
        likeli = 1

        for j in range(0, len(img)):
            if(img[j] == 1):
                likeli*=final_dist[i][j]
            else:
                likeli*=1-final_dist[i][j]

        ps.append(likeli*final_prior) #note
        org_label.append(findLabel(final_test, img, final_label, i))  
    return ps, org_label
    
def classify(c, t, all_ps, sol, predicted_label):
    
    
    for i in range(0, len(all_ps)):
        if(all_ps[i] > t):
            predicted_label[c,i] = 0
        elif(all_ps[i] < t):
            predicted_label[c,i] = 1
        else:
            predicted_label[c,i] = np.random.randint(0,2)
    return get_roc_points(predicted_label[c,:], sol)
    

def  NaiveBayes(final_dist, final_prior, final_test, final_label,):
    #return predicted_labels
    global cm_2
    cm_2 = np.zeros((N,N))
    predicted_labels = np.zeros((final_test.shape[0]))
    
    counter = 0    
    for img in final_test: # every image
        likelihood = [1]*N
        for c in range(0, N):  # every class
            for j in range(0, len(img)): # every feature
                if(img[j] == 1):
                    likelihood[c]*=final_dist[c][j]
                else:
                    likelihood[c]*=1-final_dist[c][j]
            if(c==0):
                likelihood[c]*=final_prior
            else:
                likelihood[c]*=1-final_prior
                
        my_label = likelihood.index(max(likelihood))
        predicted_labels[counter] = my_label
        true_label = final_label[counter]
        cm_2[my_label, true_label]+=1
        counter+=1
        likelihood.clear()
        
    return predicted_labels
    
K = 5

Acc = []

def start(td, K):
    global cm_2,Acc

    all_sets = np.split(td, K, axis=0)

    all_predicted_labels = []
    
    all_validation_train = []
    all_validation_label = []
    all_validation_dist = []
    all_validation_prior = []
    
    for i in range(0, K):  
        Acc = []
#         print(train_data.shape)
#         print(all_sets[0].shape)

        v_label = all_sets[i][:,-1]
        v_set = all_sets[i][:,:-1]     
        
        train_si = []

        train_si = [ j for j in range(0, K) if(j!=i)]
#         print(train_si)
        
        train_set = all_sets[train_si[0]]
        
        
        for j in range(1, len(train_si)):
            train_set = np.concatenate((train_set, all_sets[train_si[j]]), axis=0)
        
        
        t_label = train_set[:,-1]

        t_set = train_set[:, :-1]
        
        all_validation_train.append(t_set)
        all_validation_label.append(t_label)
        
        current_dist, prior_1 = buildDist(t_set, t_label)  # 
#         print(current_dist[0])
        all_validation_dist.append(current_dist)
        all_validation_prior.append(prior_1)
        
        all_predicted_labels.append(NBClassifier(v_set, v_label, current_dist, prior_1))
    
        print('CM: ',i+1)
        print(cm_2)
        acc = sum(np.diag(cm_2))/(sum(sum(cm_2)))
        print('Accuracy of at fold ',i+1,' ',acc)
        Acc.append(acc)
        cm_2 = np.zeros((N,N))
        train_si.clear()
    
    # ROC of the best model : Stored in Pickle files
    
    print('Mean: ', stat.mean(Acc), 'Standard deviation: ',stat.pstdev(Acc))
    
    best_model_index = Acc.index(max(Acc))
    final_test = all_validation_train[best_model_index]
    final_label = all_validation_label[best_model_index]
    final_dist = all_validation_dist[best_model_index]
    final_prior = all_validation_prior[best_model_index]
    
    all_thresholds = np.empty((N, final_test.shape[0])) #N binary classifier
    actual_label = np.empty((N, final_test.shape[0]))
    predicted_label = np.empty((N, final_test.shape[0]))
    
    #Confusion Matrix
    predicted_ls = NaiveBayes(final_dist, final_prior, final_test, final_label)
    print(cm_2)
    print('Precision: ',cm_2[0,0]/(cm_2[0,0]+cm_2[0,1]) )
    print('Recall: ', cm_2[0,0]/(cm_2[0,0] + cm_2[1,0]) )
    # Accuracy 2 class
    print(sum(np.diag(cm_2))/(sum(sum(cm_2))))
    
                         
    all_ll = np.zeros((N, final_test.shape[0]))

    for i in range(0, N):
        all_ll[i,:], actual_label[i,:] = NaiveBayesROC(final_dist, final_prior, final_test, final_label, i)
        all_thresholds[i,:] = sorted(all_ll[i,:])
    
    x_fpr = np.empty((N, final_test.shape[0]))
    y_tpr = np.empty((N, final_test.shape[0]))

    for i in range(0, len(all_ll)):
        for j in range(0, len(all_ll[0])):
            # ith class is pos.
            t = all_thresholds[i,j]
        
            y,x = classify(i, t, all_ll[i,:], actual_label[i,:], predicted_label) # returns  tpr, fpr
            x_fpr[i,j] = x
            y_tpr[i,j] = y
        
    
    


    for i in range(0, N):
        plt.plot(x_fpr[i,:], y_tpr[i,:], label=digit_type[i])
    
    plt.plot([0,1],[0,1],'--c')
    plt.title('Receiver Operating Characteristic Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    #DET of the Best Model

    
    for i in range(0, N):
        y_tpr[i,:] = 1 - y_tpr[i,:]
        plt.plot(x_fpr[i,:], y_tpr[i,:], label=digit_type[i])
        
    plt.title('Detection Error Tradeoff Curve')
    plt.xlabel('False Accept Rate') # FAR = FPR
    plt.ylabel('False Reject Rate') # FRR = 1-TPR
    plt.plot([0,1],[0,1],'--c')
    plt.legend()
    
    plt.show()
    return None

np.random.shuffle(train_data)
if(digit_type[0] == 1):
    td = train_data[:-3,:]
else:
    td = train_data[:-2,:]

start(td, K)
print(stat.mean(Acc))

