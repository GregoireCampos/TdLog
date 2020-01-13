#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 12:45:02 2019

@author: badr
"""

from dataprocess import *

from PyQt5.QtWidgets import (QWidget, QGridLayout, QPushButton, QApplication, QLabel, QMainWindow, QMessageBox)
from matplotlib.image import *
from PyQt5 import QtGui
from PyQt5 import QtCore
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
import math
import sys

df =pd.read_csv('training_file.csv', sep=';')

## Creating a training files from seasons 07-08 to 18-19
keys_to_keep = ["FTR","B365H","B365D","B365A","HTGDBG","ATGDBG","HTPBG","ATPBG"]
X =[]
Y = []
for k in range(7,19) :
    file_name = str(k)+"-"+str(k+1)+"_processed.csv"
    df=pd.read_csv('Training_Files/France/'+file_name, sep=',')
    #sep = "," fot F1_processed or sep = ";" for training_file
    #I only keep "before-game data" except FTR which I will use to train my classification algorithm
    dataset = {}
    df_dict = df.to_dict()
    print(file_name)
    for key in keys_to_keep : 
        dataset[key] = df_dict[key]
    dataset_df = pd.DataFrame.from_dict(dataset)
    df_dict = dataset_df.T.to_dict()
    X+=[list(df_dict[i].values())[1:] for i in df_dict.keys()]
    Y+=[list(df_dict[i].values())[0] for i in df_dict.keys()]

#check for NaN values
flag = 0
X_copy=[]
Y_copy=[]
for i in range(len(X)) :
    for k in range(len(X[i])):
        if math.isnan(X[i][k]):
            flag = 1
    if flag ==0 :
        X_copy+=[X[i]]
        Y_copy+=[Y[i]]
    else :
        print("Incorrect data : " + str(i))
    flag = 0
X = X_copy
Y=Y_copy

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
accepted_games_decision_tree = []
accepted_odd_decision_tree =[]
accepted_Y_decision_tree = [] # résultat du match 0 1 ou 2
for k in range (len(dtree_predictions)) : 
    for i in range(len(dtree_predictions[k])) : 
        if dtree_predictions[k][i]>threshold : 
            accepted_games_decision_tree+=[i]
            accepted_Y_decision_tree+=[testing_Y[k]]
            accepted_odd_decision_tree+=[testing_X[k][i]]
#creating a confusion matrix 
cm = confusion_matrix(accepted_Y_decision_tree, accepted_games_decision_tree) 
true_class_decision_tree = cm[0][0]+cm[1][1]+cm[2][2]
"""
print("Decision tree correct answers (%): ")
print(true_class/len(accepted_Y))
print("cote moyenne")
print(sum(accepted_odd)/len(accepted_odd))
print("Gain moyen par match pour 20E par mise  : ")
print((true_class/len(accepted_Y))*(sum(accepted_odd)/len(accepted_odd))*20-20)
print("-----------------")
"""

#training a linear SVM classifier 
#from sklearn.svm import SVC 
#svm_model_linear = SVC(kernel = 'linear', C = 1, probability = True).fit(training_X, training_Y) 
#svm_predictions = svm_model_linear.predict_proba(testing_X) 

#accepted_games = []
#accepted_Y = []
#accepted_odd =[]

#for k in range (len(svm_predictions)) : 
#    for i in range(len(svm_predictions[k])) : 
#        if svm_predictions[k][i]>threshold : 
#            accepted_games+=[i]
#            accepted_Y+=[testing_Y[k]]
#            accepted_odd+=[testing_X[k][i]]
#model accuracy for X_test   
#accuracy = svm_model_linear.score(testing_X, testing_Y) 

# creating a confusion matrix 
#cm = confusion_matrix(accepted_Y, accepted_games)
#true_class = cm[0][0]+cm[1][1]+cm[2][2]
#print(" SVM correct answers (%): ")
#print(true_class/len(accepted_Y))
#print("cote moyenne")
#print(sum(accepted_odd)/len(accepted_odd))
#print("Gain moyenpar match pour 20E par mise  : ")
#print((true_class/len(accepted_Y))*(sum(accepted_odd)/len(accepted_odd))*20-20)
#print("-----------------")

# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 3).fit(training_X, training_Y) 
  
  
# creating a confusion matrix 
knn_predictions = knn.predict_proba(testing_X) 

accepted_games_knn = []
accepted_Y_knn = []
accepted_odd_knn =[]

for k in range (len(knn_predictions)) : 
    for i in range(len(knn_predictions[k])) : 
        if knn_predictions[k][i]>threshold : 
            accepted_games_knn+=[i]
            accepted_Y_knn+=[testing_Y[k]]
            accepted_odd_knn+=[testing_X[k][i]] 
cm = confusion_matrix(accepted_Y_knn, accepted_games_knn) 
true_class_knn = cm[0][0]+cm[1][1]+cm[2][2]
"""
print("KNN correct answers (%): ")
print(true_class/len(accepted_Y))
print("cote moyenne")
print(sum(accepted_odd)/len(accepted_odd))
print("Gain moyenpar match pour 20E par mise  : ")
print((true_class/len(accepted_Y))*(sum(accepted_odd)/len(accepted_odd))*20-20)
print("-----------------")
"""

# training a Naive Bayes classifier 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(training_X, training_Y) 
gnb_predictions = gnb.predict_proba(testing_X) 
  
accepted_games_bayes = []
accepted_Y_bayes = []
accepted_odd_bayes =[]

for k in range (len(gnb_predictions)) : 
    for i in range(len(gnb_predictions[k])) : 
        if gnb_predictions[k][i]>threshold : 
            accepted_games_bayes+=[i]
            accepted_Y_bayes+=[testing_Y[k]]
            accepted_odd_bayes+=[testing_X[k][i]] 
  
# creating a confusion matrix 
cm = confusion_matrix(accepted_Y_bayes, accepted_games_bayes) 
true_class_bayes = cm[0][0]+cm[1][1]+cm[2][2]
"""
print("Bayes correct answers (%): ")
print(true_class/len(accepted_Y))
print("cote moyenne")
print(sum(accepted_odd)/len(accepted_odd))
print("Gain moyenpar match pour 20E par mise  : ")
print((true_class/len(accepted_Y))*(sum(accepted_odd)/len(accepted_odd))*20-20)
print("-----------------")
"""


#button =QPushButton(" ", self)
#button.setIcon(labyrinthe.B[i][j].representation_graphique)
#button.setIconSize(QtCore.QSize(30,30))

class Window(QMainWindow):
    
    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(50,50,500,300)
        self.setWindowTitle("TdLog")
        self.home()

    def home(self):
        btn = QPushButton("KNN method", self)
        btn.clicked.connect(self.knnmethod)
        btn.resize(100,100)
        btn.move(100,100)
        self.show()
        
    def knnmethod(self):
        print("KNN correct answers (%): ")
        print(true_class_knn/len(accepted_Y_knn))
        print("cote moyenne")
        print(sum(accepted_odd_knn)/len(accepted_odd_knn))
        print("Gain moyenpar match pour 20E par mise  : ")
        print((true_class_knn/len(accepted_Y_knn))*(sum(accepted_odd_knn)/len(accepted_odd_knn))*20-20)
        print("-----------------")
        sys.exit()

def run():        
    app = QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())
    
run()