# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 22:05:04 2020

@author: campo
"""
import datetime
import time
import pandas as pd
from dataprocess import *
import random
from PyQt5.QtWidgets import (QWidget, QLineEdit, QVBoxLayout, QGridLayout, QCheckBox, QPushButton, 
                             QApplication, QLabel, QMainWindow, QMessageBox, QHBoxLayout)
import sys
import matplotlib.pyplot as plt
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC 
from PyQt5.QtCore import Qt



class Window(QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(50,50, 2000,1200)
        self.setWindowTitle("TdLog : betting sports")
        self.home()
        self.flag = 0
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.yellow)
        self.setPalette(p)
        # mettre un bouton détail
        
    def home(self):
        self.knn_method_triggered = 0
        self.dtree_method_triggered = 0
        self.bayes_method_triggered = 0
        self.svm_method_triggered = 0
        self.btn_Quit = QPushButton("Quit", self)
        self.btn_Quit.clicked.connect(self.quitt)
        self.btn_svm = QPushButton("svm method", self)
        self.btn_svm.clicked.connect(self.svmmethod)
        self.label_svm = QLabel(self)
        self.label_svm.setText("")
        self.threshold = 0.5
        self.btn_Thresh = QPushButton("Threshold value", self)
        self.btn_Thresh.clicked.connect(self.threshold_value)
        self.label_Thresh = QLineEdit(self)
        self.label_Thresh.setText("0.5")
        self.label_Thresh_text = QLabel(self)
        self.label_Thresh_text.setText("")
        self.btn_0 = QPushButton("Process data", self)
        self.btn_0.clicked.connect(self.processed_data)
        self.label_0 = QLabel(self)
        self.label_0.setText("Please process data")        
        self.btn_1 = QPushButton("KNN method", self)
        self.btn_1.clicked.connect(self.knnmethod)
        self.label_1 = QLabel(self)
        self.label_1.setText("")
        self.details_knn = QPushButton("details..", self)
        self.details_knn.clicked.connect(self.details_over_knn)
        self.details_bayes = QPushButton("details..", self)
        self.details_bayes.clicked.connect(self.details_over_bayes)
        self.details_dtree = QPushButton("details..", self)
        self.details_dtree.clicked.connect(self.details_over_dtree)
        self.btn_2 = QPushButton("Bayes", self)
        self.btn_2.clicked.connect(self.bayes)
        self.label_2 = QLabel(self)
        self.label_2.setText("")
        self.btn_3 = QPushButton("Decision_tree", self)
        self.btn_3.clicked.connect(self.decision_tree)
        self.label_3 = QLabel(self)
        self.label_3.setText("")
        self.btn_6 = QCheckBox("Next games you will bet on according to Decision Tree", self)
        self.btn_6.clicked.connect(self.decision_tree_next_games)
        self.label_6 = QLabel(self)
        self.label_6.setText("Please process data and try knn method \nbefore checking the box")
        self.btn_5 = QCheckBox("Next games you will bet on according to Bayes",self)
        self.btn_5.clicked.connect(self.bayes_next_games)
        self.label_5 = QLabel(self)
        self.label_5.setText("Please process data and try Bayes method \nbefore checking the box")
        self.btn_4 = QCheckBox("Next games you will bet on according to KNN", self)
        self.btn_4.clicked.connect(self.knn_next_games)
        self.label_4 = QLabel(self)
        self.label_4.setText("Please process data and try decision tree \nmethod before checking the box")
        self.position()
        self.show()        
        
    def position(self):
        self.btn_Quit.move(1500,0)
        self.btn_0.move(0,10)
        self.btn_0.adjustSize()
        self.label_0.move(0,50)
        self.label_0.adjustSize()
        self.btn_Thresh.move(600,0)
        self.btn_Thresh.adjustSize()
        self.label_Thresh.move(650,50)
        self.label_Thresh_text.move(650,100)
        self.label_Thresh_text.adjustSize()
        self.btn_1.move(250,350)
        self.btn_1.adjustSize()
        self.details_knn.move(120,470)
        self.details_knn.adjustSize()
        self.label_1.move(250,400)
        self.btn_2.move(650,350)
        self.btn_2.adjustSize()
        self.details_bayes.move(520,470)
        self.details_bayes.adjustSize()
        self.label_2.move(650,400)
        self.btn_3.move(1000,350)
        self.btn_3.adjustSize()
        self.details_dtree.move(870,470)
        self.details_dtree.adjustSize()
        self.label_3.move(1000,400)
        self.btn_6.move(1350,600)
        self.btn_6.adjustSize()
        self.label_4.move(1350,620)
        self.label_4.adjustSize()
        self.btn_5.move(770,600)
        self.btn_5.adjustSize()
        self.label_5.move(770,620)
        self.btn_4.move(120,600)
        self.btn_4.adjustSize()
        self.label_6.move(120,620)
        self.label_5.adjustSize()
        self.btn_svm.move(1550,350)
        self.btn_svm.adjustSize()
        self.label_svm.move(1550,620)
        self.label_svm.adjustSize()
        self.label_6.adjustSize()
        

        
    def quitt(self):
        self.close()
    
    def threshold_value(self):
        print(self.label_Thresh.text())
        if (float(self.label_Thresh.text()) > 1):
            self.label_Thresh_text.setText("please enter a value between 0 and 1")
            self.label_Thresh_text.adjustSize()
        else :
            self.threshold = float(self.label_Thresh.text())
            if self.flag == 1  and self.knn_method_triggered == 1 :
                self.knnmethod()
            if self.btn_4.checkState()==2 and self.flag == 1 :
                    self.print_knn_next_games() 
            if self.flag == 1  and self.bayes_method_triggered == 1 :
                self.bayes()
            if self.btn_5.checkState() == 2 and self.flag == 1 :
                    self.print_bayes_next_games() 
            if self.flag == 1  and self.dtree_method_triggered == 1 :
                self.decision_tree()
            if self.btn_6.checkState() == 2 and self.flag == 1 :
                    self.print_decision_tree_next_games()
    
    
    def knnmethod(self):
        if self.flag == 1:
            self.knn_method_triggered = 1
            r = random.random()
            random.shuffle(self.X, lambda:r)
            random.shuffle(self.Y, lambda:r)
            training_X = self.X[:int(len(self.X)-len(self.X)/5)]
            testing_X = self.X[int(len(self.X)-len(self.X)/5):]       
            training_Y = self.Y[:int(len(self.Y)-len(self.Y)/5)]
            testing_Y = self.Y[int(len(self.Y)-len(self.Y)/5):]        
            self.knn = KNeighborsClassifier(n_neighbors = 3).fit(training_X, training_Y) 
            # creating a confusion matrix 
            knn_predictions = self.knn.predict_proba(testing_X) 
            accepted_games_knn = []
            accepted_Y_knn = []
            accepted_odd_knn =[]
            sum_true_odd_knn = 0
            number_of_won_games_knn = 0
            X_bet_knn = [20]
            for k in range (len(knn_predictions)) : 
                for i in range(len(knn_predictions[k])) : 
                    if knn_predictions[k][i]>self.threshold : 
                        accepted_games_knn+=[i]
                        accepted_Y_knn+=[testing_Y[k]]
                        accepted_odd_knn+=[testing_X[k][i]]
                        l = len(X_bet_knn)
                        if testing_Y[k] == i:
                            sum_true_odd_knn += testing_X[k][i]
                            number_of_won_games_knn += 1
                            X_bet_knn.append(X_bet_knn[l-1]+20*((sum_true_odd_knn/number_of_won_games_knn) - 1))
                        else :
                            X_bet_knn.append(X_bet_knn[l-1]-20)
            self.X_bet_knn = X_bet_knn
            cm_knn = confusion_matrix(accepted_Y_knn, accepted_games_knn) 
            true_class_knn = cm_knn[0][0]+cm_knn[1][1]+cm_knn[2][2]
            correct_answers = true_class_knn/len(accepted_Y_knn)
            correct_answers = round(correct_answers,4)
            correct_answers = str(correct_answers)
            average_bet = sum_true_odd_knn/number_of_won_games_knn
            average_bet = round(average_bet,4)
            average_bet = str(average_bet)
            #cote moyenne des acceptés * proportion de victoire
            average_gain = (true_class_knn/len(accepted_Y_knn))*(sum_true_odd_knn/(true_class_knn))*20-20
            average_gain = round(average_gain,4)
            average_gain = str(average_gain)
            self.label_1.setText("KNN correct answers (%):"+correct_answers+" \nAverage won odd :"+average_bet+" \nWhat you got betting 20 £ :"+average_gain)
            self.label_1.adjustSize()
            if self.btn_4.checkState()==2:
                self.print_knn_next_games() 
        else :
            self.label_1.setText("""Please process the data before  
            starting analyse them ...""")
            self.label_1.adjustSize()
            
    def print_knn_next_games(self):
        df = pd.read_csv("Next_games.csv", sep=';')
        dataset = {}
        df_dict_origin = df.to_dict()
        for key in self.keys_to_keep : 
            dataset[key] = df_dict_origin[key]
        dataset_df = pd.DataFrame.from_dict(dataset)
        df_dict = dataset_df.T.to_dict()
        X_knn =[]
        X_knn+=[list(df_dict[i].values())[1:] for i in df_dict.keys()]
        predictions = self.knn.predict_proba(X_knn)
        Games_you_need_to_bet_on = []
        number_of_bets = 0
        for i in range(len(predictions)) :
            # print("%s gagne avec proba %f, Match nul avec %f, et %s gagne avec proba %f" %(df_dict_origin["HomeTeam"][i],predictions[i][0],predictions[i][1], df_dict_origin["AwayTeam"][i],predictions[i][2])) 
            if (predictions[i][0] > self.threshold) :
                #print("proba calculée victoire dom = %f" %(predictions[i][0]))
                #print("cote victoire dom = %f" %(df_dict_origin["B365H"][i]))
                Games_you_need_to_bet_on.append("\nBet on %s during %s against %s on %s" %(df_dict_origin["HomeTeam"][i],df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
                number_of_bets += 1
            if (predictions[i][1] > self.threshold) :
                #print("proba calculée match nul = %f" %(predictions[i][1]))
                #print("cote match nul = %f" %(df_dict_origin["B365D"][i]))
                Games_you_need_to_bet_on.append("\nBet on a draw during %s against %s on %s"%(df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
                number_of_bets += 1
            if (predictions[i][2] > self.threshold) :
                #print("proba calculée victoire ext = %f" %(predictions[i][2]))
                #print("cote victoire ext = %f" %(df_dict_origin["B365A"][i]))
                Games_you_need_to_bet_on.append("\nBet on %s during %s against %s on %s"%(df_dict_origin["AwayTeam"][i],df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
                number_of_bets += 1
        self.label_6.setText(' '.join(Games_you_need_to_bet_on))
        self.label_6.adjustSize()
        
    def knn_next_games(self):
        if self.flag == 1 and self.btn_4.checkState() == 2 :
            self.print_knn_next_games()
        else :
            self.label_6.setText("Please process data, check the box and \ntry knn method")
            self.label_6.adjustSize()
            
    def details_over_knn(self):
        if self.flag == 1 and self.knn_method_triggered == 1:
            # tracer le gain en fonction de la date
            # fonction de gain
            l = len(self.X_bet_knn)
            Y = [1]*(l)
            for i in range(1,l):
                Y[i] = Y[i-1]+1               
            plt.plot(Y, self.X_bet_knn)
            plt.title("Evolution of you current cash using that instance of KNN method")
            plt.ylabel("Current_cash")
            plt.xlabel("Time")
            plt.show()
        
                
            
    
    def bayes(self):
        if self.flag == 1:
            self.bayes_method_triggered = 1
            r = random.random()
            random.shuffle(self.X, lambda:r)
            random.shuffle(self.Y, lambda:r)
            training_X = self.X[:int(len(self.X)-len(self.X)/5)]
            testing_X = self.X[int(len(self.X)-len(self.X)/5):]          
            training_Y = self.Y[:int(len(self.Y)-len(self.Y)/5)]
            testing_Y = self.Y[int(len(self.Y)-len(self.Y)/5):]
            self.gnb = GaussianNB().fit(training_X, training_Y) 
            gnb_predictions = self.gnb.predict_proba(testing_X) 
            accepted_games_bayes = []
            accepted_Y_bayes = []
            accepted_odd_bayes =[]
            sum_true_odd_bayes = 0
            number_of_won_games_bayes = 0
            self.X_bet_bayes = [20]
            for k in range (len(gnb_predictions)) : 
                for i in range(len(gnb_predictions[k])) : 
                    if gnb_predictions[k][i]>self.threshold : #si la proba du joueur i de gagner le match k est au dessus du threshold
                        accepted_games_bayes+=[i]
                        accepted_Y_bayes+=[testing_Y[k]]
                        accepted_odd_bayes+=[testing_X[k][i]] 
                        l = len(self.X_bet_bayes)
                        if testing_Y[k] == i:
                            sum_true_odd_bayes += testing_X[k][i]
                            number_of_won_games_bayes += 1
                            self.X_bet_bayes.append(self.X_bet_bayes[l-1]+20*((sum_true_odd_bayes/number_of_won_games_bayes) - 1))
                        else :
                            self.X_bet_bayes.append(self.X_bet_bayes[l-1]-20)
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
            if self.btn_5.checkState() == 2:
                self.print_bayes_next_games()   
        else :
            self.label_2.setText("""Please process the data before 
            starting analyse them ...""")
            self.label_2.adjustSize()
    
    def print_bayes_next_games(self):
        df = pd.read_csv("Next_games.csv", sep=';')
        dataset = {}
        df_dict_origin = df.to_dict()
        for key in self.keys_to_keep : 
            dataset[key] = df_dict_origin[key]
        dataset_df = pd.DataFrame.from_dict(dataset)
        df_dict = dataset_df.T.to_dict()
        X_bayes =[]
        X_bayes+=[list(df_dict[i].values())[1:] for i in df_dict.keys()]
        predictions = self.gnb.predict_proba(X_bayes)
        Games_you_need_to_bet_on = []
        number_of_bets = 0
        for i in range(len(predictions)) :
            # print("%s gagne avec proba %f, Match nul avec %f, et %s gagne avec proba %f" %(df_dict_origin["HomeTeam"][i],predictions[i][0],predictions[i][1], df_dict_origin["AwayTeam"][i],predictions[i][2]))
            if (predictions[i][0] > self.threshold) :
                #print("proba calculée victoire dom = %f" %(predictions[i][0]))
                #print("cote victoire dom = %f" %(df_dict_origin["B365H"][i]))
                Games_you_need_to_bet_on.append("\nBet on %s during %s against %s on %s" %(df_dict_origin["HomeTeam"][i],df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
                number_of_bets += 1
            if (predictions[i][1] > self.threshold) :
                #print("proba calculée match nul = %f" %(predictions[i][1]))
                #print("cote match nul = %f" %(df_dict_origin["B365D"][i]))
                Games_you_need_to_bet_on.append("\nBet on a draw during %s against %s on %s"%(df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
                number_of_bets += 1
            if (predictions[i][2] > self.threshold) :
                #print("proba calculée victoire ext = %f" %(predictions[i][2]))
                #print("cote victoire ext = %f" %(df_dict_origin["B365A"][i]))
                Games_you_need_to_bet_on.append("\nBet on %s during %s against %s on %s"%(df_dict_origin["AwayTeam"][i],df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
                number_of_bets += 1
        self.label_5.setText(' '.join(Games_you_need_to_bet_on))
        self.label_5.adjustSize()             
        
    def bayes_next_games(self):
        if self.flag == 1 and self.btn_5.checkState() == 2 :
            self.print_bayes_next_games()
        else :
            self.label_5.setText("Please process data, check the box and /ntry Bayes method")
            
    def details_over_bayes(self):
        if self.flag == 1 and self.bayes_method_triggered == 1:
            # tracer le gain en fonction de la date
            # fonction de gain
            l = len(self.X_bet_bayes)
            Y = [1]*(l)
            for i in range(1,l):
                Y[i] = Y[i-1]+1               
            plt.plot(Y, self.X_bet_bayes)
            plt.title("Evolution of you current cash using that instance of Bayes method")
            plt.ylabel("Current_cash")
            plt.xlabel("Time")
            plt.show()

    def decision_tree(self):
        if self.flag == 1:
            self.dtree_method_triggered = 1
            r = random.random()
            random.shuffle(self.X, lambda:r)
            random.shuffle(self.Y, lambda:r)
            training_X = self.X[:int(len(self.X)-len(self.X)/5)]
            testing_X = self.X[int(len(self.X)-len(self.X)/5):]
            training_Y = self.Y[:int(len(self.Y)-len(self.Y)/5)]
            testing_Y = self.Y[int(len(self.Y)-len(self.Y)/5):] 
            self.dtree_model = DecisionTreeClassifier(max_depth = 10).fit(training_X, training_Y) 
            dtree_predictions = self.dtree_model.predict_proba(testing_X) 
            accepted_games_decision_tree = []
            accepted_odd_decision_tree =[]
            accepted_Y_decision_tree = [] # résultat du match 0 1 ou 2
            sum_true_odd_decision_tree = 0 #somme des cotes des matchs gagnés
            number_of_won_games_decision_tree = 0
            X_bet_dtree = [20]
            for k in range (len(dtree_predictions)) : 
                for i in range(len(dtree_predictions[k])) : 
                    if dtree_predictions[k][i]>self.threshold :
                        accepted_games_decision_tree+=[i] #prédiction
                        accepted_Y_decision_tree+=[testing_Y[k]] #vrai résultat
                        accepted_odd_decision_tree+=[testing_X[k][i]]
                        l = len(X_bet_dtree)       
                        if testing_Y[k] == i:
                            sum_true_odd_decision_tree += testing_X[k][i]
                            number_of_won_games_decision_tree += 1                     
                            X_bet_dtree.append(X_bet_dtree[l-1]+20*((sum_true_odd_decision_tree/number_of_won_games_decision_tree) - 1))
                        else :
                            X_bet_dtree.append(X_bet_dtree[l-1]-20) 
            self.X_bet_dtree = X_bet_dtree
            #creating a confusion matrix 
            cm_decision_tree = confusion_matrix(accepted_Y_decision_tree, accepted_games_decision_tree) 
            true_class_decision_tree = cm_decision_tree[0][0]+cm_decision_tree[1][1]+cm_decision_tree[2][2]
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
            if self.btn_6.checkState() == 2:
                self.print_decision_tree_next_games()
        else :
            self.label_3.setText("""Please process the data before 
            starting analyse them ...""")
            self.label_3.adjustSize()
            
    def print_decision_tree_next_games(self):
        df = pd.read_csv("Next_games.csv", sep=';')            
        dataset = {}
        df_dict_origin = df.to_dict()
        for key in self.keys_to_keep : 
            dataset[key] = df_dict_origin[key]
        dataset_df = pd.DataFrame.from_dict(dataset)
        df_dict = dataset_df.T.to_dict()           
        X_dtree =[]
        X_dtree+=[list(df_dict[i].values())[1:] for i in df_dict.keys()]
        predictions = self.dtree_model.predict_proba(X_dtree)
        Games_you_need_to_bet_on = []
        number_of_bets = 0
        # avec une marge de 0 on parie 12 fois la même journée ... si on rajoutait une marge ? recalculer le gain ? 
        # après les partiels ?
        for i in range(len(predictions)) :
            # print("%s gagne avec proba %f, Match nul avec %f, et %s gagne avec proba %f" %(df_dict_origin["HomeTeam"][i],predictions[i][0],predictions[i][1], df_dict_origin["AwayTeam"][i],predictions[i][2]))
            if (predictions[i][0] > self.threshold) :
                #print("proba calculée victoire dom = %f" %(predictions[i][0]))
                #print("cote victoire dom = %f" %(df_dict_origin["B365H"][i]))
                Games_you_need_to_bet_on.append("\nBet on %s during %s against %s on %s" %(df_dict_origin["HomeTeam"][i],df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
                number_of_bets += 1
            if (predictions[i][1] > self.threshold) :
                #print("proba calculée match nul = %f" %(predictions[i][1]))
                #print("cote match nul = %f" %(df_dict_origin["B365D"][i]))
                Games_you_need_to_bet_on.append("\nBet on a draw during %s against %s on %s"%(df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
                number_of_bets += 1
            if (predictions[i][2] > self.threshold) :
                #print("proba calculée victoire ext = %f" %(predictions[i][2]))
                #print("cote victoire ext = %f" %(df_dict_origin["B365A"][i]))
                Games_you_need_to_bet_on.append("\nBet on %s during %s against %s on %s"%(df_dict_origin["AwayTeam"][i],df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
                number_of_bets += 1
        self.label_4.setText(' '.join(Games_you_need_to_bet_on))
        self.label_4.adjustSize()

    def decision_tree_next_games(self):
        if self.flag == 1 and self.btn_6.checkState() == 2:
            self.print_decision_tree_next_games() 
        else :
            self.label_4.setText("Please process data, check the box and \ntry decision tree method")
            
    def details_over_dtree(self):
        if self.flag == 1 and self.dtree_method_triggered == 1:
            # tracer le gain en fonction de la date
            # fonction de gain
            l = len(self.X_bet_dtree)
            Y = [1]*(l)
            for i in range(1,l):
                Y[i] = Y[i-1]+1               
            plt.plot(Y, self.X_bet_dtree)
            plt.title("Evolution of you current cash using that instance of Decision Tree method")
            plt.ylabel("Current_cash")
            plt.xlabel("Time")
            plt.show()

            
    def processed_data(self):
        df =pd.read_csv('training_file.csv', sep=';')         
        self.threshold = 0.5
        ## Creating a training files from seasons 07-08 to 18-19
        self.keys_to_keep = ["FTR","B365H","B365D","B365A","HTGDBG","ATGDBG","HTPBG","ATPBG"]
        self.X =[]
        self.Y = []
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
            for key in self.keys_to_keep : 
                dataset[key] = df_dict[key]
            dataset_df = pd.DataFrame.from_dict(dataset)
            df_dict = dataset_df.T.to_dict()
            self.X+=[list(df_dict[i].values())[1:] for i in df_dict.keys()]
            self.Y+=[list(df_dict[i].values())[0] for i in df_dict.keys()]
        
        #check for NaN values
        flag = 0
        X_copy=[]
        Y_copy=[]
        for i in range(len(self.X)) :
            for k in range(len(self.X[i])):
                if math.isnan(self.X[i][k]):
                    flag = 1
            if flag ==0 :
                X_copy+=[self.X[i]]
                Y_copy+=[self.Y[i]]
            else :
                print("Incorrect data : " + str(i))
            flag = 0
        self.X = X_copy
        self.Y=Y_copy
        
        for i in range(len(self.Y)) : 
            if self.Y[i]=="H" : 
                self.Y[i]=0
            elif self.Y[i] =="D" : 
                self.Y[i]=1
            else :
                self.Y[i]=2
        self.label_0.setText(file_name_total +"\n data successfully processed")
        self.label_0.adjustSize()
        self.flag = 1

    def svmmethod(self):
        if self.flag == 1:
            self.svm_method_triggered = 1
            r = random.random()
            random.shuffle(self.X, lambda:r)
            random.shuffle(self.Y, lambda:r)
            training_X = self.X[:int(len(self.X)-len(self.X)/5)]
            testing_X = self.X[int(len(self.X)-len(self.X)/5):]       
            training_Y = self.Y[:int(len(self.Y)-len(self.Y)/5)]
            testing_Y = self.Y[int(len(self.Y)-len(self.Y)/5):] 
            self.svm_model_linear = SVC(kernel = 'linear', C = 1, probability = True).fit(training_X, training_Y) 
            svm_predictions = self.svm_model_linear.predict_proba(testing_X) 
            accepted_games_svm = []
            accepted_Y_svm = []
            accepted_odd_svm =[]
            sum_true_odd_svm = 0
            number_of_won_games_svm = 0
            for k in range (len(svm_predictions)) : 
                for i in range(len(svm_predictions[k])) : 
                    if svm_predictions[k][i]>self.threshold : 
                        accepted_games_svm+=[i]            
                        accepted_Y_svm+=[testing_Y[k]]
                        accepted_odd_svm+=[testing_X[k][i]]
                        if testing_Y[k] == i:
                            sum_true_odd_svm += testing_X[k][i]
                            number_of_won_games_svm += 1
            cm_svm = confusion_matrix(accepted_Y_svm, accepted_games_svm)
            true_class_svm = cm_svm[0][0]+cm_svm[1][1]+cm_svm[2][2]
            correct_answers = true_class_svm/len(accepted_Y_svm)
            correct_answers = round(correct_answers,4)
            correct_answers = str(correct_answers)
            average_bet = sum_true_odd_svm/number_of_won_games_svm
            average_bet = round(average_bet,4)
            average_bet = str(average_bet)
            #cote moyenne des acceptés * proportion de victoire
            average_gain = (true_class_svm/len(accepted_Y_svm))*(sum_true_odd_svm/(true_class_svm))*20-20
            average_gain = round(average_gain,4)
            average_gain = str(average_gain)
            self.label_1.setText("SVM correct answers (%):"+correct_answers+" \nAverage won odd :"+average_bet+" \nWhat you got betting 20 £ :"+average_gain)
            self.label_1.adjustSize()
            if self.btn_svm.checkState()==2:
                self.print_svm_next_games() 
        else :
            self.label_svm.setText("Please process the data before  starting analyse them ...")
            self.label_svm.adjustSize()
            
    def print_svm_next_games(self):
        df = pd.read_csv("Next_games.csv", sep=';')
        dataset = {}
        df_dict_origin = df.to_dict()
        for key in self.keys_to_keep : 
            dataset[key] = df_dict_origin[key]
        dataset_df = pd.DataFrame.from_dict(dataset)
        df_dict = dataset_df.T.to_dict()
        X_svm =[]
        X_svm+=[list(df_dict[i].values())[1:] for i in df_dict.keys()]
        predictions = self.svm.predict_proba(X_svm)
        Games_you_need_to_bet_on = []
        number_of_bets = 0
        margin = 0
        for i in range(len(predictions)) :
            # print("%s gagne avec proba %f, Match nul avec %f, et %s gagne avec proba %f" %(df_dict_origin["HomeTeam"][i],predictions[i][0],predictions[i][1], df_dict_origin["AwayTeam"][i],predictions[i][2])) 
            if (predictions[i][0] > self.threshold) :
                #print("proba calculée victoire dom = %f" %(predictions[i][0]))
                #print("cote victoire dom = %f" %(df_dict_origin["B365H"][i]))
                Games_you_need_to_bet_on.append("\nParier sur %s lors du match %s contre %s du %s" %(df_dict_origin["HomeTeam"][i],df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
                number_of_bets += 1
            if (predictions[i][1] > self.threshold) :
                #print("proba calculée match nul = %f" %(predictions[i][1]))
                #print("cote match nul = %f" %(df_dict_origin["B365D"][i]))
                Games_you_need_to_bet_on.append("\nParier sur match nul lors du match %s contre %s du %s"%(df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
                number_of_bets += 1
            if (predictions[i][2] > self.threshold) :
                #print("proba calculée victoire ext = %f" %(predictions[i][2]))
                #print("cote victoire ext = %f" %(df_dict_origin["B365A"][i]))
                Games_you_need_to_bet_on.append("\nParier sur %s lors du match %s contre %s du %s"%(df_dict_origin["AwayTeam"][i],df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
                number_of_bets += 1
        self.label_svm.setText(' '.join(Games_you_need_to_bet_on))
        self.label_svm.adjustSize()
        
    def svm_next_games(self):
        if self.flag == 1 and self.btn_svm.checkState() == 2 :
            self.print_svm_next_games()
            
        else :
            self.label_svm.setText("Please process data, check the box and \ntry svm method")
            self.label_svm.adjustSize()

    
        

def run():        
    app = QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())

run()
