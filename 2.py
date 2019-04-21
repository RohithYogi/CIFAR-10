#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np 
from glob import glob 
import argparse
import matplotlib as mpl
from matplotlib import pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression


# In[2]:


import sys, random, os
import shutil
def splitdata(files, dir1, dir2, ratio):
    shuffled_files = files[:]
    random.shuffle(shuffled_files)
    num = round(len(shuffled_files) * ratio)
    print(num)
    to_dir1, to_dir2 = shuffled_files[num:],shuffled_files[:num]
    for file in to_dir1:
        shutil.copy2(file, os.path.join(dir1, os.path.basename(file)))
    for file in to_dir2:
        shutil.copy2(file, os.path.join(dir2, os.path.basename(file)))


# In[3]:


files = glob('./Data/Horses/*') 
dir1 = './Data/Train/Horses/'
dir2 = './Data/Test/Horses/'
percentage_test_data = 0.25
splitdata(files,dir1,dir2,percentage_test_data)


# In[4]:


files = glob('./Data/Bikes/*') 
dir1 = './Data/Train/Bikes/'
dir2 = './Data/Test/Bikes/'
percentage_test_data = 0.25
splitdata(files,dir1,dir2,percentage_test_data)


# In[4]:


def load_images(images,path,category):
    count = 0
    images[category] = []
    for imagefile in path:
        img = cv2.imread(imagefile, 0)
        images[category].append(img)
        count +=1
    return count


# In[5]:


images = {}
path = glob('./Data/Train/Bikes/*')
category = 'Bikes'
count_bikes = load_images(images,path,category)
print(count_bikes)


# In[6]:


path = glob('./Data/Train/Horses/*')
category = 'Horses'
count_horses = load_images(images,path,category)
print(count_horses)
total = count_horses+count_bikes
print(total)


# In[8]:


print(images)


# In[7]:


def features(image):
    keypoints, descriptors = cv2.xfeatures2d.SIFT_create().detectAndCompute(image, None)
    return [keypoints, descriptors]


# In[8]:


count = total


# In[9]:


label_count = 0 
train_labels = np.array([])
labels_dict = {}
descriptors = []

for category, imgs in images.items():
    labels_dict[str(label_count)] = category
    for img in imgs:
        train_labels = np.append(train_labels, label_count)
        kp, des = features(img)
        descriptors.append(des)

    label_count += 1


# In[10]:


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


# In[11]:


import time
start = time.time()
n_clusters=100
kms_clusters,kms_fit = clustering(n_clusters,descriptors)
end = time.time()
print("time run : " + str(end - start)+" secs")


# In[ ]:





# In[12]:


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


# In[13]:


histogram = Histogram(count,descriptors,n_clusters,kms_fit)


# In[14]:


print(histogram)


# In[15]:


def model():
#     model =  LogisticRegression()
#     model =  KNeighborsClassifier(n_neighbors=100) 
#     model =  AdaBoostClassifier(n_estimators=100)
    model=SVC()

    return model


# In[16]:


# /home/rohith/anaconda3/lib/python3.6/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
#   "avoid this warning.", FutureWarning)


# In[17]:


scale = StandardScaler().fit(histogram)
histogram = scale.transform(histogram)


# In[18]:


print(histogram)


# In[19]:


print( "labels", train_labels) 
model=model()
model.fit(histogram, train_labels)


# In[20]:


test_images = {}
path = glob('./Data/Test/Bikes/*')
category = 'Bikes'
count_bikes = load_images(test_images,path,category)
print(count_bikes)


# In[21]:


path = glob('./Data/Test/Horses/*')
category = 'Horses'
count_horses = load_images(test_images,path,category)
print(count_horses)
test_images_count = count_horses+count_bikes
print(total)


# In[22]:


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


# In[23]:


def predict(test_images,labels_dict,n_clusters,scale,kms_clusters,model):
    predictions = []
    for category, imgs in test_images.items():
        for img in imgs:
            cls = recognize_image(img,n_clusters,scale,kms_clusters,model)
            predictions.append(cls)

    return predictions


# In[24]:


predictions=predict(test_images,labels_dict,n_clusters,scale,kms_clusters,model)


# In[ ]:





# In[25]:


predicted_labels=[]
for i in predictions:
    if(i==[0.]):
        predicted_labels.append([0])
    else:
        predicted_labels.append([1])


# In[26]:


print(predicted_labels)


# In[27]:


test_labels=[]
for i in range(count_bikes):
    test_labels.append([0])
for i in range(count_horses):
    test_labels.append([1])


# In[28]:


print(test_labels)


# In[29]:


from sklearn.metrics import accuracy_score
accuracy_score(test_labels, predicted_labels)


# In[ ]:





# In[ ]:




