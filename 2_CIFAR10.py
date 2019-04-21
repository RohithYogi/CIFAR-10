#!/usr/bin/env python
# coding: utf-8

# In[17]:


import cv2
import numpy as np 
from glob import glob 
import argparse
import matplotlib as mpl

from __future__ import print_function
from six.moves import cPickle as pickle
import os
from scipy.misc import imread
import platform
import pickle

from matplotlib import pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


# In[18]:


def load_CIFAR10_batch(cifar10_dataset_folder_path, batch_id):

    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1).astype("float")
    labels = batch['labels']

    return features,labels


# In[19]:


def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# In[4]:


# path = './cifar-10-batches-py/' 
# features,labels = load_cfar10_batch(path,1)
# features


# In[20]:


def load_CIFAR10(ROOT):
    labels = load_label_names()
    for i in range(0,len(labels)):
        os.mkdir('./CIFAR10_data/Train/'+labels[i])
        os.mkdir('./CIFAR10_data/Test/'+labels[i])
    for b in range(1,6):
        Xtrain, Ytrain = load_CIFAR10_batch(ROOT,b)
        for i in range (0,len(Ytrain)):
            path = './CIFAR10_data/Train/'+labels[Ytrain[i]]+'/train_'+'batch_'+str(b)+'_'+str(i)+'.png'
            cv2.imwrite(path,Xtrain[i])    
    
    with open(ROOT +'/test_batch', mode='rb') as file:
        testdata = pickle.load(file, encoding='latin1')
    
    Xtest = testdata['data'].reshape((len(testdata['data']), 3, 32, 32)).transpose(0, 2, 3, 1).astype("float")
    Ytest = testdata['labels']
    
    for i in range (0,len(Ytest)):
        path = './CIFAR10_data/Test/'+labels[Ytest[i]]+'/test_'+str(i)+'.png'
        cv2.imwrite(path,Xtest[i])
    return


# In[21]:


path = './cifar-10-batches-py'


# In[22]:


labels = load_label_names()


# In[23]:


# load_CIFAR10(path)


# In[24]:


# labels


# In[25]:


# x,y = load_CIFAR10_batch(path,1)


# In[26]:


# for i in range (0,len(y)):
#     path = './CIFAR10_data/Train/'+labels[y[i]]+'/train_'+'batch_1_'+str(i)+'.png'
#     cv2.imwrite(path,x[i])


# In[27]:


# for i in range(0,len(y)):
#     print(labels[y[i]])


# In[28]:


# for i in range(0,len(y)):
#     print(x[i])


# In[29]:


# cv2.imwrite('image.png',x[0])


# In[30]:


def load_images(images,path,labels):
    total=0
    for i in labels:
        count = 0
        cat_path = path+i+'/*'
        glob_path = glob(cat_path) 
        images[i] = []
        for imagefile in glob_path: 
            img = cv2.imread(imagefile, 0)
            images[i].append(img)
            count +=1
        total+=count
    return total


# In[31]:


images = {}
path = './CIFAR10_data/Train/'
count_images = load_images(images,path,labels)


# In[34]:


print(count_images)


# In[35]:


def features(image):
    keypoints, descriptors = cv2.xfeatures2d.SIFT_create().detectAndCompute(image, None)
    return [keypoints, descriptors]


# In[36]:


count = count_images


# In[37]:


label_count = 0 
train_labels = np.array([])
labels_dict = {}
descriptors = []

for category, imgs in images.items():
    labels_dict[str(label_count)] = category
    for img in imgs:
        kp, des = features(img)
        if des is not None:
            train_labels = np.append(train_labels, label_count)
            descriptors.append(des)

    label_count += 1


# In[38]:


def clustering(n_clusters,descriptors):
    #format
    l=descriptors
    vStack = np.array(l[0])
    for remaining in l[1:]:
        vStack = np.vstack((vStack, remaining))
    descriptor_stack = vStack
    print(descriptor_stack)
    #clustering
    clusters = KMeans(n_clusters = n_clusters)
    fit = clusters.fit_predict(descriptor_stack)
    return [clusters,fit]


# In[39]:


# # descriptors
# l=descriptors
# vStack = np.array(l[0])
# # np.vstack((vStack,l[84]))
# if(l[84]==None):
#     print('ok')
descriptors


# In[40]:


import time
start = time.time()
n_clusters=200
kms_clusters,kms_fit = clustering(n_clusters,descriptors)
end = time.time()
print("time run : " + str(end - start)+" secs")


# In[45]:


def Histogram(count,descriptors,n_clusters,kms_fit):
    histogram = np.array([np.zeros(n_clusters) for i in range(count)])
    length = 0
    for i in range(count):
        l = len(descriptors[i])
        for j in range(l):
            index = kms_fit[length+j]
            histogram[i][index] += 1
        length += l 
    return histogram


# In[47]:


count=len(descriptors)
histogram = Histogram(count,descriptors,n_clusters,kms_fit)


# In[48]:


def model():
    model=SVC()
    return model


# In[49]:


print(histogram)


# In[50]:


scale = StandardScaler().fit(histogram)
histogram = scale.transform(histogram)


# In[51]:


print(histogram)


# In[52]:


start = time.time()
print( "labels", train_labels) 
model=model()
model.fit(histogram, train_labels)
#start 10:31 - end 10-51
end = time.time()
print("time run : " + str(end - start)+" secs")


# In[53]:


test_images = {}
path = './CIFAR10_data/Test/'
count_test_images = load_images(test_images,path,labels)


# In[54]:


labels_dict


# In[55]:


label_count = 0 
test_labels = np.array([])
test_label_dict ={}

for category, imgs in test_images.items():
    test_label_dict[str(label_count)] = category
#     print( "Generating test label for : "+ category+"--"+ str(label_count))
    for img in imgs:
        kp, des = features(img)
        if des is not None:
            test_labels = np.append(test_labels, label_count)

    label_count += 1


# In[56]:


len(test_labels)


# In[57]:


test_label_dict


# In[58]:


def recognize_image(test_img,no_clusters,scale,kms,model):

#   extract features for test image
    kp, des = features(test_img)
#   generate vocabulary for test image by kmeans predict
    vocab = np.array( [[ 0 for i in range(no_clusters)]])
#   predict cluster for that descriptor    
    test_ret = kms.predict(des)
#   generating histogram   
    for item in test_ret:
        vocab[0][item] += 1
#   scale transformation
    vocab = scale.transform(vocab) 
#   predicted label 
    lb = model.predict(vocab)
    return lb


# In[59]:


def predict(test_images,labels_dict,n_clusters,scale,kms_clusters,model):

    predictions = []
    for category, imgs in test_images.items():
        print( "Processing : " ,category) 
        for img in imgs:
            # print( imgs[0].shape, imgs[1].shape) 
            print( img.shape) 
            kp, des = features(img)
            if des is not None:
                cls = recognize_image(img,n_clusters,scale,kms_clusters,model)
                print(cls) 
                predictions.append(cls)

    return predictions


# In[60]:


test_labels


# In[61]:


start = time.time()
predictions=predict(test_images,labels_dict,n_clusters,scale,kms_clusters,model)
end = time.time()
print("time run : " + str(end - start)+" secs")
# time run : 775.2853643894196 secs


# In[62]:


len(predictions)


# In[63]:


# for item in predictions:
#     plt.imshow(cv2.cvtColor(item['image'], cv2.COLOR_GRAY2RGB))
#     plt.title(item['object_name'])
#     plt.show()


# In[64]:


predicted_labels=[]
for item in predictions:
    for i in range(0,10):
        if(item == i):
            predicted_labels.append([i])
            break


# In[65]:


len(predicted_labels)


# In[66]:


predicted_labels


# In[67]:


test_labels


# In[68]:


ground_truth=[]
for i in test_labels:
    ground_truth.append([int(i)])


# In[69]:


ground_truth


# In[70]:


from sklearn.metrics import accuracy_score
accuracy_score(ground_truth, predicted_labels)


# In[71]:


from sklearn.externals import joblib
filename = 'svm_model_c200.pkl'
joblib.dump(model, filename)


# In[72]:


loaded_model = joblib.load(filename)
result = accuracy_score(ground_truth, predicted_labels)
print(result)


# In[ ]:




