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
#sep = "," fot F1_processed or sep = ";" for training_file
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
dtree_predictions = dtree_model.predict_proba(testing_X) 

threshold = 0.5
accepted_games = []
accepted_odd =[]
accepted_Y = []
for k in range (len(dtree_predictions)) : 
    for i in range(len(dtree_predictions[k])) : 
        if dtree_predictions[k][i]>threshold : 
            accepted_games+=[i]
            accepted_Y+=[testing_Y[k]]
            accepted_odd+=[testing_X[k][i]]
#creating a confusion matrix 
cm = confusion_matrix(accepted_Y, accepted_games) 
true_class = cm[0][0]+cm[1][1]+cm[2][2]
print("Decision tree correct answers (%): ")
print(true_class/len(accepted_Y))
print("cote moyenne")
print(sum(accepted_odd)/len(accepted_odd))
print("Gain moyen par match pour 20E par mise  : ")
print((true_class/len(accepted_Y))*(sum(accepted_odd)/len(accepted_odd))*20-20)
print("-----------------")


#training a linear SVM classifier 
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1, probability = True).fit(training_X, training_Y) 
svm_predictions = svm_model_linear.predict_proba(testing_X) 

accepted_games = []
accepted_Y = []
accepted_odd =[]

for k in range (len(svm_predictions)) : 
    for i in range(len(svm_predictions[k])) : 
        if svm_predictions[k][i]>threshold : 
            accepted_games+=[i]
            accepted_Y+=[testing_Y[k]]
            accepted_odd+=[testing_X[k][i]]
#model accuracy for X_test   
accuracy = svm_model_linear.score(testing_X, testing_Y) 

# creating a confusion matrix 
cm = confusion_matrix(accepted_Y, accepted_games)
true_class = cm[0][0]+cm[1][1]+cm[2][2]
print(" SVM correct answers (%): ")
print(true_class/len(accepted_Y))
print("cote moyenne")
print(sum(accepted_odd)/len(accepted_odd))
print("Gain moyenpar match pour 20E par mise  : ")
print((true_class/len(accepted_Y))*(sum(accepted_odd)/len(accepted_odd))*20-20)
print("-----------------")

# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 3).fit(training_X, training_Y) 
  
  
# creating a confusion matrix 
knn_predictions = knn.predict_proba(testing_X) 

accepted_games = []
accepted_Y = []
accepted_odd =[]

for k in range (len(knn_predictions)) : 
    for i in range(len(knn_predictions[k])) : 
        if knn_predictions[k][i]>threshold : 
            accepted_games+=[i]
            accepted_Y+=[testing_Y[k]]
            accepted_odd+=[testing_X[k][i]] 
cm = confusion_matrix(accepted_Y, accepted_games) 
true_class = cm[0][0]+cm[1][1]+cm[2][2]
true_class = cm[0][0]+cm[1][1]+cm[2][2]
print("KNN correct answers (%): ")
print(true_class/len(accepted_Y))
print("cote moyenne")
print(sum(accepted_odd)/len(accepted_odd))
print("Gain moyenpar match pour 20E par mise  : ")
print((true_class/len(accepted_Y))*(sum(accepted_odd)/len(accepted_odd))*20-20)
print("-----------------")


# training a Naive Bayes classifier 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(training_X, training_Y) 
gnb_predictions = gnb.predict_proba(testing_X) 
  
accepted_games = []
accepted_Y = []
accepted_odd =[]

for k in range (len(gnb_predictions)) : 
    for i in range(len(gnb_predictions[k])) : 
        if gnb_predictions[k][i]>threshold : 
            accepted_games+=[i]
            accepted_Y+=[testing_Y[k]]
            accepted_odd+=[testing_X[k][i]] 
  
# creating a confusion matrix 
cm = confusion_matrix(accepted_Y, accepted_games) 
true_class = cm[0][0]+cm[1][1]+cm[2][2]
true_class = cm[0][0]+cm[1][1]+cm[2][2]
print("Bayes correct answers (%): ")
print(true_class/len(accepted_Y))
print("cote moyenne")
print(sum(accepted_odd)/len(accepted_odd))
print("Gain moyenpar match pour 20E par mise  : ")
print((true_class/len(accepted_Y))*(sum(accepted_odd)/len(accepted_odd))*20-20)
print("-----------------")

