#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import scipy.io
import numpy as np


# In[ ]:


#Load data

X_train = scipy.io.loadmat('Data for Problem 1/X_train.mat')
X_train=pd.DataFrame(X_train["X_train"])

X_test = scipy.io.loadmat('Data for Problem 1/X_test.mat')
X_test=pd.DataFrame(X_test["X_test"])

Y_train = scipy.io.loadmat('Data for Problem 1/y_train.mat')
Y_train=pd.DataFrame(Y_train["y_train"])

Y_test = scipy.io.loadmat('Data for Problem 1/y_test.mat')
Y_test=pd.DataFrame(Y_test["y_test"])


# # Polynomial kernel , parameter 2

# In[ ]:


#Import svm model
from sklearn import svm

X_test_predict=pd.DataFrame()
for i in range(6):
    #Create a svm Classifier
    clf = svm.SVC(kernel='poly',degree=2) # Linear Kernel
    #Train the model using the training sets
    clf.fit(X_train, Y_train[i])
    #Predict the response for test dataset
    y_pred= clf.predict(X_test)
    X_test_predict[i]=y_pred


# In[ ]:


X_test_predict=X_test_predict.values.tolist()
Y_test=Y_test.values.tolist()


# In[ ]:


accuracy=[]
length=len(X_test)
for row in range(length):
    numerator=0
    denominator=0
    for class_index in range(6):
        if X_test_predict[row][class_index]==1 and Y_test[row][class_index]==1:
            numerator=numerator+1
        if X_test_predict[row][class_index]==1 or Y_test[row][class_index]==1:
            denominator=denominator+1
    acc=numerator/denominator
    accuracy.append(acc)

accuracy_percen_poly=(sum(accuracy)/len(accuracy))*100
accuracy_percen_poly


# # Gaussian kernel , parameter 2

# In[ ]:


Y_test = scipy.io.loadmat('Data for Problem 1/y_test.mat')
Y_test=pd.DataFrame(Y_test["y_test"])


# In[ ]:


#Import svm model
from sklearn import svm

X_test_predict=pd.DataFrame()
for i in range(6):
    #Create a svm Classifier
    clf = svm.SVC(kernel='rbf',degree=2) # gaussian Kernel
    #Train the model using the training sets
    clf.fit(X_train, Y_train[i])
    #Predict the response for test dataset
    y_pred= clf.predict(X_test)
    X_test_predict[i]=y_pred


# In[ ]:


X_test_predict=X_test_predict.values.tolist()
Y_test=Y_test.values.tolist()


# In[ ]:


accuracy=[]
length=len(X_test)
for row in range(length):
    numerator=0
    denominator=0
    for class_index in range(6):
        if X_test_predict[row][class_index]==1 and Y_test[row][class_index]==1:
            numerator=numerator+1
        if X_test_predict[row][class_index]==1 or Y_test[row][class_index]==1:
            denominator=denominator+1
    acc=numerator/denominator
    accuracy.append(acc)

accuracy_percen_gaus=(sum(accuracy)/len(accuracy))*100
accuracy_percen_gaus


# In[ ]:


print('------accuracy percentage-------------','\n',
      'Polynomial kernel : ',accuracy_percen_poly,'\n',
      'Gausian kernel : ', accuracy_percen_gaus)

