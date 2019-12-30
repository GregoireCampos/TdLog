#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 12:45:02 2019

@author: badr
"""

from dataprocess import *

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 

df =pd.read_csv('training_file.csv', sep=';')

df_dict = df.to_dict()
keys_to_keep = ["FTR","B365H","B365D","B365A","HTGDBG","ATGDBG","HTPBG","ATPBG"]
#I only keep "before-game data" except FTR which I will use to train my classification algorithm
dataset = {}
for key in keys_to_keep : 
    dataset[key] = df_dict[key]
dataset_df = pd.DataFrame.from_dict(dataset)
df_dict = dataset_df.T.to_dict()

X=[list(df_dict[i].values())[1:] for i in df_dict.keys()]
Y = [list(df_dict[i].values())[0] for i in df_dict.keys()]
for i in range(len(Y)) : 
    if Y[i]=="H" : 
        Y[i]=0
    elif Y[i] =="D" : 
        Y[i]=1
    else :
        Y[i]=2
        
training_X = X[:int(len(X)-len(X)/5)]
testing_X = X[int(len(X)-len(X)/5):]

training_Y = Y[:int(len(Y)-len(Y)/5)]
testing_Y = Y[int(len(Y)-len(Y)/5):]

from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 10).fit(training_X, training_Y) 
dtree_predictions = dtree_model.predict(testing_X) 
  
# creating a confusion matrix 
cm = confusion_matrix(testing_Y, dtree_predictions) 
true_class = cm[0][0]+cm[1][1]+cm[2][2]
print("Decision tree correct answers (%): ")
print(true_class/len(testing_Y))


# training a linear SVM classifier 
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(training_X, training_Y) 
svm_predictions = svm_model_linear.predict(testing_X) 
  
# model accuracy for X_test   
accuracy = svm_model_linear.score(testing_X, testing_Y) 
  
# creating a confusion matrix 
cm = confusion_matrix(testing_Y, svm_predictions)
true_class = cm[0][0]+cm[1][1]+cm[2][2]
print(" SVM correct answers (%): ")
print(true_class/len(testing_Y)) 

# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 3).fit(training_X, training_Y) 
  
  
# creating a confusion matrix 
knn_predictions = knn.predict(testing_X)  
cm = confusion_matrix(testing_Y, knn_predictions) 
true_class = cm[0][0]+cm[1][1]+cm[2][2]
print(" KNN correct answers (%): ")
print(true_class/len(testing_Y)) 


# training a Naive Bayes classifier 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(training_X, training_Y) 
gnb_predictions = gnb.predict(testing_X) 
  
  
# creating a confusion matrix 
cm = confusion_matrix(testing_Y, gnb_predictions) 
true_class = cm[0][0]+cm[1][1]+cm[2][2]
print(" Bayes correct answers (%): ")
print(true_class/len(testing_Y)) 

