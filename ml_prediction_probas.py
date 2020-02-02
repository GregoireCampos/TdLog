#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 12:45:02 2019

@author: badr
"""

from dataprocess import *
import random
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGridLayout, QPushButton, QApplication, QLabel, QMainWindow, QMessageBox)
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
file_name_total = ""

for k in range(7,19) :
    file_name = str(k)+"-"+str(k+1)+"_processed.csv"
    file_name_total = file_name_total+"""
    """+file_name
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
        
##à changer pour s'entraîner sur des valeurs prises aléatoirement et tester sur le reste des valeurs

r = random.random()
#print(r)
#print( "X =")
#print(X)
#print("Y=")
#print(Y)
random.shuffle(X, lambda:r)
random.shuffle(Y, lambda:r)
#print( "X shuffle =")
#print(X)
#print("Y shuffle =")
#print(Y)
training_X = X[:int(len(X)-len(X)/5)]
testing_X = X[int(len(X)-len(X)/5):]

training_Y = Y[:int(len(Y)-len(Y)/5)]
testing_Y = Y[int(len(Y)-len(Y)/5):] 

from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 10).fit(training_X, training_Y) 
dtree_predictions = dtree_model.predict_proba(testing_X) 
# print(testing_X[0])

threshold = 0.5
accepted_games_decision_tree = []
accepted_odd_decision_tree =[]
accepted_Y_decision_tree = [] # résultat du match 0 1 ou 2
sum_true_odd_decision_tree = 0 #somme des cotes des matchs gagnés
number_of_won_games_decision_tree = 0
for k in range (len(dtree_predictions)) : 
    for i in range(len(dtree_predictions[k])) : 
        if dtree_predictions[k][i]>threshold : 
            accepted_games_decision_tree+=[i] #prédiction
            accepted_Y_decision_tree+=[testing_Y[k]] #vrai résultat
            accepted_odd_decision_tree+=[testing_X[k][i]]
            if testing_Y[k] == i:
                sum_true_odd_decision_tree += testing_X[k][i]
                number_of_won_games_decision_tree += 1
#creating a confusion matrix 
cm_decision_tree = confusion_matrix(accepted_Y_decision_tree, accepted_games_decision_tree) 
true_class_decision_tree = cm_decision_tree[0][0]+cm_decision_tree[1][1]+cm_decision_tree[2][2]
#les paris où on avait raison
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
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'poly', C = 1, probability = True).fit(training_X, training_Y) 
svm_predictions = svm_model_linear.predict_proba(testing_X) 

accepted_games = []
accepted_Y = []
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
sum_true_odd_knn = 0
number_of_won_games_knn = 0
for k in range (len(knn_predictions)) : 
    for i in range(len(knn_predictions[k])) : 
        if knn_predictions[k][i]>threshold : 
            accepted_games_knn+=[i]
            accepted_Y_knn+=[testing_Y[k]]
            accepted_odd_knn+=[testing_X[k][i]] 
            if testing_Y[k] == i:
                sum_true_odd_knn += testing_X[k][i]
                number_of_won_games_knn += 1
cm_knn = confusion_matrix(accepted_Y_knn, accepted_games_knn) 
true_class_knn = cm_knn[0][0]+cm_knn[1][1]+cm_knn[2][2]
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
sum_true_odd_bayes = 0
number_of_won_games_bayes = 0
for k in range (len(gnb_predictions)) : 
    for i in range(len(gnb_predictions[k])) : 
        if gnb_predictions[k][i]>threshold : 
            accepted_games_bayes+=[i]
            accepted_Y_bayes+=[testing_Y[k]]
            accepted_odd_bayes+=[testing_X[k][i]] 
            if testing_Y[k] == i:
                sum_true_odd_bayes += testing_X[k][i]
                number_of_won_games_bayes += 1
# creating a confusion matrix 
cm_bayes = confusion_matrix(accepted_Y_bayes, accepted_games_bayes) 
true_class_bayes = cm_bayes[0][0]+cm_bayes[1][1]+cm_bayes[2][2]
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




def predict_games_dtree() : 
    df = pd.read_csv("Next_games.csv", sep=';')
    dataset = {}
    df_dict_origin = df.to_dict()
    for key in keys_to_keep : 
        dataset[key] = df_dict_origin[key]
    dataset_df = pd.DataFrame.from_dict(dataset)
    df_dict = dataset_df.T.to_dict()
    X =[]
    X+=[list(df_dict[i].values())[1:] for i in df_dict.keys()]
    predictions = dtree_model.predict_proba(X)
    margin = 0
    Games_you_need_to_bet_on = []
    number_of_bets = 0
    for i in range(len(predictions)) :
        # print("%s gagne avec proba %f, Match nul avec %f, et %s gagne avec proba %f" %(df_dict_origin["HomeTeam"][i],predictions[i][0],predictions[i][1], df_dict_origin["AwayTeam"][i],predictions[i][2]))
        if (predictions[i][0] > 1/df_dict_origin["B365H"][i] + margin) :
            #print("proba calculée victoire dom = %f" %(predictions[i][0]))
            #print("cote victoire dom = %f" %(df_dict_origin["B365H"][i]))
            Games_you_need_to_bet_on.append("\nParier sur %s lors du match %s contre %s du %s" %(df_dict_origin["HomeTeam"][i],df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
            number_of_bets += 1
        if (predictions[i][1] > 1/df_dict_origin["B365D"][i] + margin) :
            #print("proba calculée match nul = %f" %(predictions[i][1]))
            #print("cote match nul = %f" %(df_dict_origin["B365D"][i]))
            Games_you_need_to_bet_on.append("\nParier sur match nul lors du match %s contre %s du %s"%(df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
            number_of_bets += 1
        if (predictions[i][2] > 1/df_dict_origin["B365A"][i] + margin) :
            #print("proba calculée victoire ext = %f" %(predictions[i][2]))
            #print("cote victoire ext = %f" %(df_dict_origin["B365A"][i]))
            Games_you_need_to_bet_on.append("\nParier sur %s lors du match %s contre %s du %s"%(df_dict_origin["AwayTeam"][i],df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
            number_of_bets += 1
    return (Games_you_need_to_bet_on)


def predict_games_knn() : 
    df = pd.read_csv("Next_games.csv", sep=';')
    dataset = {}
    df_dict_origin = df.to_dict()
    for key in keys_to_keep : 
        dataset[key] = df_dict_origin[key]
    dataset_df = pd.DataFrame.from_dict(dataset)
    df_dict = dataset_df.T.to_dict()
    X =[]
    X+=[list(df_dict[i].values())[1:] for i in df_dict.keys()]
    predictions = knn.predict_proba(X)
    Games_you_need_to_bet_on = []
    number_of_bets = 0
    margin = 0
    for i in range(len(predictions)) :
        print("%s gagne avec proba %f, Match nul avec %f, et %s gagne avec proba %f" %(df_dict_origin["HomeTeam"][i],predictions[i][0],predictions[i][1], df_dict_origin["AwayTeam"][i],predictions[i][2])) 
        if (predictions[i][0] > 1/df_dict_origin["B365H"][i] + margin) :
            #print("proba calculée victoire dom = %f" %(predictions[i][0]))
            #print("cote victoire dom = %f" %(df_dict_origin["B365H"][i]))
            Games_you_need_to_bet_on.append("\nParier sur %s lors du match %s contre %s du %s" %(df_dict_origin["HomeTeam"][i],df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
            number_of_bets += 1
        if (predictions[i][1] > 1/df_dict_origin["B365D"][i] + margin) :
            #print("proba calculée match nul = %f" %(predictions[i][1]))
            #print("cote match nul = %f" %(df_dict_origin["B365D"][i]))
            Games_you_need_to_bet_on.append("\nParier sur match nul lors du match %s contre %s du %s"%(df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
            number_of_bets += 1
        if (predictions[i][2] > 1/df_dict_origin["B365A"][i] + margin) :
            #print("proba calculée victoire ext = %f" %(predictions[i][2]))
            #print("cote victoire ext = %f" %(df_dict_origin["B365A"][i]))
            Games_you_need_to_bet_on.append("\nParier sur %s lors du match %s contre %s du %s"%(df_dict_origin["AwayTeam"][i],df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
            number_of_bets += 1
    return (Games_you_need_to_bet_on)
        
def predict_games_gnb() : 
    df = pd.read_csv("Next_games.csv", sep=';')
    dataset = {}
    df_dict_origin = df.to_dict()
    for key in keys_to_keep : 
        dataset[key] = df_dict_origin[key]
    dataset_df = pd.DataFrame.from_dict(dataset)
    df_dict = dataset_df.T.to_dict()
    X =[]
    X+=[list(df_dict[i].values())[1:] for i in df_dict.keys()]
    predictions = gnb.predict_proba(X)
    Games_you_need_to_bet_on = []
    number_of_bets = 0
    # avec une marge de 0 on parie 12 fois la même journée ... si on rajoutait une marge ? recalculer le gain ? 
    # après les partiels ?
    margin = 0
    for i in range(len(predictions)) :
        # print("%s gagne avec proba %f, Match nul avec %f, et %s gagne avec proba %f" %(df_dict_origin["HomeTeam"][i],predictions[i][0],predictions[i][1], df_dict_origin["AwayTeam"][i],predictions[i][2]))
        if (predictions[i][0] > 1/df_dict_origin["B365H"][i] + margin) :
            #print("proba calculée victoire dom = %f" %(predictions[i][0]))
            #print("cote victoire dom = %f" %(df_dict_origin["B365H"][i]))
            Games_you_need_to_bet_on.append("\nParier sur %s lors du match %s contre %s du %s" %(df_dict_origin["HomeTeam"][i],df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
            number_of_bets += 1
        if (predictions[i][1] > 1/df_dict_origin["B365D"][i] + margin) :
            #print("proba calculée match nul = %f" %(predictions[i][1]))
            #print("cote match nul = %f" %(df_dict_origin["B365D"][i]))
            Games_you_need_to_bet_on.append("\nParier sur match nul lors du match %s contre %s du %s"%(df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
            number_of_bets += 1
        if (predictions[i][2] > 1/df_dict_origin["B365A"][i] + margin) :
            #print("proba calculée victoire ext = %f" %(predictions[i][2]))
            #print("cote victoire ext = %f" %(df_dict_origin["B365A"][i]))
            Games_you_need_to_bet_on.append("\nParier sur %s lors du match %s contre %s du %s"%(df_dict_origin["AwayTeam"][i],df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
            number_of_bets += 1
    return (Games_you_need_to_bet_on)


class Window(QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(50,50,1500,800)
        self.setWindowTitle("TdLog")
        self.home()
        self.flag = 0

    def home(self):
        btn_Quit = QPushButton("Quit", self)
        btn_Quit.clicked.connect(self.quitt)
        btn_Quit.move(1500,0)
        btn_0 = QPushButton("Process data", self)
        btn_0.clicked.connect(self.processed_data)
        btn_0.move(0,10)
        btn_0.adjustSize()
        self.label_0 = QLabel(self)
        self.label_0.setText("please process the data")
        self.label_0.move(0,50)
        self.label_0.adjustSize()
        btn_1 = QPushButton("KNN method", self)
        btn_1.clicked.connect(self.knnmethod)
        btn_1.move(250,350)
        btn_1.adjustSize()
        self.label_1 = QLabel(self)
        self.label_1.setText("")
        self.label_1.move(250,400)
        self.label_1.adjustSize()
        btn_2 = QPushButton("Bayes", self)
        btn_2.clicked.connect(self.bayes)
        btn_2.move(650,350)
        btn_2.adjustSize()
        self.label_2 = QLabel(self)
        self.label_2.setText("")
        self.label_2.move(650,400)
        btn_3 = QPushButton("Decision_tree", self)
        btn_3.clicked.connect(self.decision_tree)
        btn_3.move(1000,350)
        btn_3.adjustSize()
        self.label_3 = QLabel(self)
        self.label_3.setText("")
        self.label_3.move(1000,400)
        btn_6 = QPushButton("Next games you will bet on according to Decision Tree", self)
        btn_6.clicked.connect(self.knn_next_games)
        btn_6.move(1350,600)
        btn_6.adjustSize()
        self.label_6 = QLabel(self)
        self.label_6.setText("")
        self.label_6.move(1350,620)
        btn_5 = QPushButton("Next games you will bet on according to Bayes",self)
        btn_5.clicked.connect(self.bayes_next_games)
        btn_5.resize(100,100)
        btn_5.move(770,600)
        btn_5.adjustSize()
        self.label_5 = QLabel(self)
        self.label_5.setText("")
        self.label_5.move(770,620)
        btn_4 = QPushButton("Next games you will bet on according to KNN", self)
        btn_4.clicked.connect(self.decision_tree_next_games)
        btn_4.resize(100,100)
        btn_4.move(120,600)
        btn_4.adjustSize()
        self.label_4 = QLabel(self)
        self.label_4.setText("")
        self.label_4.move(120,620)
        self.show()        
        
    def quitt(self):
        self.close()
    
    
    def knnmethod(self):
        if self.flag == 1:
            correct_answers = true_class_knn/len(accepted_Y_knn)
            correct_answers = round(correct_answers,4)
            correct_answers = str(correct_answers)
            average_bet = sum_true_odd_knn/number_of_won_games_decision_tree
            average_bet = round(average_bet,4)
            average_bet = str(average_bet)
            #cote moyenne des acceptés * proportion de victoire
            average_gain = (true_class_knn/len(accepted_Y_knn))*(sum_true_odd_knn/(true_class_knn))*20-20
            average_gain = round(average_gain,4)
            average_gain = str(average_gain)
            self.label_1.setText("KNN correct answers (%):"+correct_answers+" \nAverage won odd :"+average_bet+" \nWhat you got betting 20 £ :"+average_gain)
            self.label_1.adjustSize()
        else :
            self.label_1.setText("""Please process the data before  
            starting analyse them ...""")
            self.label_1.adjustSize()
            
    def knn_next_games(self):
        if self.flag == 1:
            self.label_6.setText(' '.join(predict_games_knn()))
            self.label_6.adjustSize()
        else :
            print (0)

        
    def bayes(self):
        if self.flag == 1:
            r = random.random()
            random.shuffle(X, lambda:r)
            random.shuffle(Y, lambda:r)
            gnb = GaussianNB().fit(training_X, training_Y) 
            gnb_predictions = gnb.predict_proba(testing_X) 
            accepted_games_bayes = []
            accepted_Y_bayes = []
            accepted_odd_bayes =[]
            sum_true_odd_bayes = 0
            number_of_won_games_bayes = 0
            for k in range (len(gnb_predictions)) : 
                for i in range(len(gnb_predictions[k])) : 
                    if gnb_predictions[k][i]>threshold : 
                        accepted_games_bayes+=[i]
                        accepted_Y_bayes+=[testing_Y[k]]
                        accepted_odd_bayes+=[testing_X[k][i]] 
                        if testing_Y[k] == i:
                            sum_true_odd_bayes += testing_X[k][i]
                            number_of_won_games_bayes += 1
            cm_bayes = confusion_matrix(accepted_Y_bayes, accepted_games_bayes) 
            true_class_bayes = cm_bayes[0][0]+cm_bayes[1][1]+cm_bayes[2][2]
            correct_answers = true_class_bayes/len(accepted_Y_bayes)
            correct_answers = round(correct_answers,4)
            correct_answers = str(correct_answers)
            average_bet = sum_true_odd_bayes/number_of_won_games_bayes
            average_bet = round(average_bet,4)
            average_bet = str(average_bet)
            average_gain = (true_class_bayes/len(accepted_Y_bayes))*(sum_true_odd_bayes/(true_class_bayes))*20-20
            average_gain = round(average_gain,4)
            average_gain = str(average_gain)
            self.label_2.setText("Bayes correct answers (%):"+correct_answers+" \nAverage won odd :"+average_bet+" \nWhat you got betting 20 £ :"+average_gain)
            self.label_2.adjustSize()
        else :
            self.label_2.setText("""Please process the data before 
            starting analyse them ...""")
            self.label_2.adjustSize()
        
    def bayes_next_games(self):
        if self.flag == 1:
            self.label_5.setText(' '.join(predict_games_gnb()))
            self.label_5.adjustSize()            
        else :
            print (0)
            
    def decision_tree(self):
        if self.flag == 1:
            correct_answers = true_class_decision_tree/len(accepted_Y_decision_tree)
            correct_answers = round(correct_answers,4)
            correct_answers = str(correct_answers)
            average_bet = (sum_true_odd_decision_tree)/number_of_won_games_decision_tree
            average_bet = round(average_bet,4)
            average_bet = str(average_bet)
            average_gain = (true_class_decision_tree/len(accepted_Y_decision_tree))*(sum_true_odd_decision_tree/(true_class_decision_tree))*20-20
            average_gain = round(average_gain,4)
            average_gain = str(average_gain)
            self.label_3.setText("Decision tree correct answers (%):"+correct_answers+" \nAverage won odd :"+average_bet+" \nWhat you got betting 20 £ :"+average_gain)
            self.label_3.adjustSize()
        else :
            self.label_3.setText("""Please process the data before 
            starting analyse them ...""")
            self.label_3.adjustSize()

    def decision_tree_next_games(self):
        if self.flag == 1:
            self.label_4.setText(' '.join(predict_games_dtree()))
            self.label_4.adjustSize()
        else :
            print (0)
            
    def processed_data(self):
        self.label_0.setText(file_name_total +"\n data successfully processed")
        self.label_0.adjustSize()
        self.flag = 1
        

def run():        
    app = QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())

   
predict_games_gnb()    
run()
