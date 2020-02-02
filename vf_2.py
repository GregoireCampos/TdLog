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
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix 
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
        p.setColor(self.backgroundRole(), Qt.lightGray)
        self.setPalette(p)
        self.details_over_dtree_triggered = 0
        self.details_over_bayes_triggered = 0
        self.details_over_knn_triggered = 0
        
    def home(self):
        self.knn_method_triggered = 0
        self.decision_tree_method_triggered = 0
        self.bayes_method_triggered = 0
        self.svm_method_triggered = 0
        self.flag = 0
        self.label_1 = QLabel(self)
        self.label_1.setText("")
        self.label_2 = QLabel(self)
        self.label_2.setText("")
        self.label_3 = QLabel(self)
        self.label_3.setText("")
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
        self.btn_1.clicked.connect(self.launch_knn)
        self.details_knn = QPushButton("details..", self)
        self.details_knn.clicked.connect(self.details)
        self.details_bayes = QPushButton("details..", self)
        self.details_bayes.clicked.connect(self.details)
        self.details_dtree = QPushButton("details..", self)
        self.details_dtree.clicked.connect(self.details)
        self.btn_2 = QPushButton("Bayes", self)
        self.btn_2.clicked.connect(self.launch_bayes)
        self.btn_3 = QPushButton("Decision_tree", self)
        self.btn_3.clicked.connect(self.launch_decision_tree)
        self.btn_6 = QCheckBox("Next games you will bet on according to Decision Tree", self)
        self.btn_6.clicked.connect(self.next_games)
        self.label_6 = QLabel(self)
        self.label_6.setText("Please process data and try knn method \nbefore checking the box")
        self.btn_5 = QCheckBox("Next games you will bet on according to Bayes",self)
        self.btn_5.clicked.connect(self.next_games)
        self.label_5 = QLabel(self)
        self.label_5.setText("Please process data and try Bayes method \nbefore checking the box")
        self.btn_4 = QCheckBox("Next games you will bet on according to KNN", self)
        self.btn_4.clicked.connect(self.next_games)
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
            # sep = "," fot F1_processed or sep = ";" for training_file
            # We only keep "before-game data" except FTR which we will use to train our classification algorithm
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
        # allow user to use the other buttons
        self.flag = 1
        
    
    
    def quitt(self):
        self.close()
    
    def threshold_value(self):
        # we compare the threshold with a probability so it has to stand between 0 and 1
        if (float(self.label_Thresh.text()) > 1 or float(self.label_Thresh.text()) < 0):
            self.label_Thresh_text.setText("please enter a value between 0 and 1")
            self.label_Thresh_text.adjustSize()
        else :
            self.threshold = float(self.label_Thresh.text())
            if self.flag == 1  and self.knn_method_triggered == 1 :
                self.launch_knn()
            if self.btn_4.checkState()==2 and self.flag == 1 :
                self.used_method = 2
                self.print_next_games(self.used_method)
            if self.flag == 1  and self.bayes_method_triggered == 1 :
                self.launch_bayes()
            if self.btn_5.checkState() == 2 and self.flag == 1 :
                self.used_method = 1
                self.print_next_games(self.used_method)
            if self.flag == 1  and self.decision_tree_method_triggered == 1 :
               self.launch_decision_tree()
            if self.btn_6.checkState() == 2 and self.flag == 1 :
                self.used_method = 0
                self.print_next_games(self.used_method)
            if self.details_over_knn_triggered == 1 :
                self.details()
            if self.details_over_bayes_triggered == 1 :
                self.details()
            if self.details_over_dtree_triggered == 1 :
                self.details()
    
    
    def launch_knn(self):
        self.method_number = 2
        self.method()
    
    def launch_decision_tree(self):
        self.method_number = 0
        self.method()
    
    def launch_bayes(self):
        self.method_number = 1
        self.method()
        
    
    def method(self):
        if self.flag == 1:
            r = random.random()
            random.shuffle(self.X, lambda:r)
            random.shuffle(self.Y, lambda:r)
            training_X = self.X[:int(len(self.X)-len(self.X)/5)]
            testing_X = self.X[int(len(self.X)-len(self.X)/5):]       
            training_Y = self.Y[:int(len(self.Y)-len(self.Y)/5)]
            testing_Y = self.Y[int(len(self.Y)-len(self.Y)/5):]    
            if self.method_number == 0:
                self.decision_tree_method_triggered = 1
                self.dtree_model = DecisionTreeClassifier(max_depth = 10).fit(training_X, training_Y) 
                predictions = self.dtree_model.predict_proba(testing_X)
            if self.method_number == 1:
                self.bayes_method_triggered = 1
                self.gnb = GaussianNB().fit(training_X, training_Y) 
                predictions = self.gnb.predict_proba(testing_X)
            if self.method_number == 2:
                self.knn_method_triggered = 1    
                self.knn = KNeighborsClassifier(n_neighbors = 3).fit(training_X, training_Y) 
                # creating a confusion matrix 
                predictions = self.knn.predict_proba(testing_X) 
            accepted_games = []
            accepted_Y = []
            accepted_odd =[]
            sum_true_odd = 0
            number_of_won_games = 0
            X_bet = [20]
            for k in range (len(predictions)) : 
                for i in range(len(predictions[k])) : 
                    if predictions[k][i]>self.threshold : 
                        accepted_games+=[i]
                        accepted_Y+=[testing_Y[k]]
                        accepted_odd+=[testing_X[k][i]]
                        l = len(X_bet)
                        if testing_Y[k] == i:
                            sum_true_odd += testing_X[k][i]
                            number_of_won_games += 1
                            X_bet.append(X_bet[l-1]+20*((sum_true_odd/number_of_won_games) - 1))
                        else :
                            X_bet.append(X_bet[l-1]-20)
            if self.method_number == 0:
                self.X_bet_dtree = X_bet
            if self.method_number == 1:
                self.X_bet_bayes = X_bet
            if self.method_number == 2:
                self.X_bet_knn = X_bet
            cm = confusion_matrix(accepted_Y, accepted_games) 
            true_class = cm[0][0]+cm[1][1]+cm[2][2]
            correct_answers = true_class/len(accepted_Y)
            correct_answers = round(correct_answers,4)
            correct_answers = str(correct_answers)
            average_bet = sum_true_odd/number_of_won_games
            average_bet = round(average_bet,4)
            average_bet = str(average_bet)
            average_gain = (true_class/len(accepted_Y))*(sum_true_odd/(true_class))*20-20
            average_gain = round(average_gain,4)
            average_gain = str(average_gain)
            if self.method_number == 2:
                self.label_1.setText("KNN correct answers (%):"+correct_answers+" \nAverage won odd :"+average_bet+" \nWhat you got betting 20 £ :"+average_gain)
                self.label_1.adjustSize()
            if self.method_number ==1:
                self.label_2.setText("Bayes correct answers (%):"+correct_answers+" \nAverage won odd :"+average_bet+" \nWhat you got betting 20 £ :"+average_gain)
                self.label_2.adjustSize()
            if self.method_number == 0:
                self.label_3.setText("Decision tree correct answers (%):"+correct_answers+" \nAverage won odd :"+average_bet+" \nWhat you got betting 20 £ :"+average_gain)
                self.label_3.adjustSize()
            if self.method_number == 2 and self.btn_4.checkState()==2:
                self.used_method = 2
                self.print_next_games(self.used_method)
                if self.details_over_knn_triggered == 1:
                    self.details()
            if self.method_number == 1 and self.btn_5.checkState()==2:
                self.used_method = 1
                self.print_next_games(self.used_method)  
                if self.details_over_bayes_triggered == 1:
                    self.details()
            if self.method_number == 0 and self.btn_6.checkState()==2:
                self.used_method = 0
                self.print_next_games(self.used_method) 
                if self.details_over_dtree_triggered == 1:    
                    self.details()
                    
            # you have to push "details" if you have pushed again a method, this is one of the problems 
        else :
            self.label_1.setText("""Please process the data before  
            starting analyse them ...""")
            self.label_1.adjustSize()
            self.label_2.setText("""Please process the data before 
            starting analyse them ...""")
            self.label_2.adjustSize()  
            self.label_3.setText("""Please process the data before 
            starting analyse them ...""")
            self.label_3.adjustSize()
            

### Those  functions print on the user interface the next games and teams he has to bet on
    
    def next_games(self):
        if self.flag == 1 :
            if self.btn_4.checkState() == 2 :
                self.used_method = 2
                self.print_next_games(self.used_method)
            if self.btn_5.checkState() == 2 :
                self.used_method = 1
                self.print_next_games(self.used_method)
            if self.btn_6.checkState() == 2:
                self.used_method = 0
                self.print_next_games(self.used_method)            
            
    def print_next_games(self, method_name):
        df = pd.read_csv("Next_games.csv", sep=';')            
        dataset = {}
        df_dict_origin = df.to_dict()
        for key in self.keys_to_keep : 
            dataset[key] = df_dict_origin[key]
        dataset_df = pd.DataFrame.from_dict(dataset)
        df_dict = dataset_df.T.to_dict()           
        X =[]
        X +=[list(df_dict[i].values())[1:] for i in df_dict.keys()]
        if method_name == 0 :
            predictions = self.dtree_model.predict_proba(X)
        if method_name == 1 :
            predictions = self.gnb.predict_proba(X)
        if method_name == 2 :
            predictions = self.knn.predict_proba(X)
        Games_you_need_to_bet_on = []
        number_of_bets = 0
        for i in range(len(predictions)) :
            # we browse predictions and print the team and the game if our prediction are good enough to bet on it
            if (predictions[i][0] > self.threshold) :
                # if probability of home team victory is higher than the threshold, we bet on home team
                Games_you_need_to_bet_on.append("\nBet on %s during %s against %s on %s" %(df_dict_origin["HomeTeam"][i],df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
                number_of_bets += 1
            if (predictions[i][1] > self.threshold) :
                Games_you_need_to_bet_on.append("\nBet on a draw during %s against %s on %s"%(df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
                number_of_bets += 1
            if (predictions[i][2] > self.threshold) :
                Games_you_need_to_bet_on.append("\nBet on %s during %s against %s on %s"%(df_dict_origin["AwayTeam"][i],df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
                number_of_bets += 1
        if method_name == 0:
            self.label_4.setText(' '.join(Games_you_need_to_bet_on))
            self.label_4.adjustSize()
        if method_name == 1:
            self.label_5.setText(' '.join(Games_you_need_to_bet_on))
            self.label_5.adjustSize()
        if method_name == 2:
            self.label_6.setText(' '.join(Games_you_need_to_bet_on))
            self.label_6.adjustSize()
            

### Those functions plot the details of current cash over time according relating to the differents kinds of method we consider

    
    def details(self):
        if self.flag == 1:  
            if self.decision_tree_method_triggered == 1:
                self.details_over_dtree_triggered = 1
            if self.bayes_method_triggered == 1:
                self.details_over_bayes_triggered = 1
            if self.knn_method_triggered == 1:
                self.details_over_knn_triggered = 1
            self.draw()

    def draw(self):
        plt.figure(1)
        plt.clf()
        plt.subplot(221)
        if self.details_over_dtree_triggered == 1:  
            l = len(self.X_bet_dtree)
            Y = [1]*(l)
            for i in range(1,l):
                Y[i] = Y[i-1]+1               
            plt.figure(1)
            plt.plot(Y, self.X_bet_dtree, 'b')
            plt.title("Decision Tree method")
            plt.ylabel("Current_cash")
            plt.xlabel("Time")
        plt.subplot(222)
        if self.details_over_bayes_triggered == 1:  
            l = len(self.X_bet_bayes)
            Y = [1]*(l)
            for i in range(1,l):
                Y[i] = Y[i-1]+1               
            plt.plot(Y, self.X_bet_bayes, 'm')
            plt.title("Bayes method")
            plt.ylabel("Current_cash")
            plt.xlabel("Time")
        plt.subplot (223)
        if self.details_over_knn_triggered == 1:
            l = len(self.X_bet_knn)
            Y = [1]*(l)
            for i in range(1,l):
                Y[i] = Y[i-1]+1      
            plt.plot(Y, self.X_bet_knn, 'g')
            plt.title("KNN method")
            plt.ylabel("Current_cash")
            plt.xlabel("Time")
        plt.draw()

    def svmmethod(self):
        # check if he has processed data
        if self.flag == 1:
            self.svm_method_triggered = 1
            # remix the data to train the model each time on different values set
            r = random.random()
            random.shuffle(self.X, lambda:r)
            random.shuffle(self.Y, lambda:r)
            training_X = self.X[:int(len(self.X)-len(self.X)/5)]
            testing_X = self.X[int(len(self.X)-len(self.X)/5):]       
            training_Y = self.Y[:int(len(self.Y)-len(self.Y)/5)]
            testing_Y = self.Y[int(len(self.Y)-len(self.Y)/5):] 
            # use of svm_linear model
            self.svm_model_linear = SVC(kernel = 'linear', C = 1, probability = True).fit(training_X, training_Y) 
            svm_predictions = self.svm_model_linear.predict_proba(testing_X) 
            ### compute some data concerning the model to give an overview of it
            # pool of games we bet on
            accepted_games_svm = []
            accepted_Y_svm = []
            accepted_odd_svm =[]
            sum_true_odd_svm = 0
            number_of_won_games_svm = 0
            for k in range (len(svm_predictions)) : 
                for i in range(len(svm_predictions[k])) : 
                    # if the probability we computed thanks to svm model is higher than the threshold, we bet
                    if svm_predictions[k][i]>self.threshold : 
                        accepted_games_svm+=[i]            
                        accepted_Y_svm+=[testing_Y[k]]
                        accepted_odd_svm+=[testing_X[k][i]]
                        if testing_Y[k] == i:
                            sum_true_odd_svm += testing_X[k][i]
                            number_of_won_games_svm += 1
            # building a confusion matrix
            cm_svm = confusion_matrix(accepted_Y_svm, accepted_games_svm)
            # pool of games we bet on successfully 
            true_class_svm = cm_svm[0][0]+cm_svm[1][1]+cm_svm[2][2]
            correct_answers = true_class_svm/len(accepted_Y_svm)
            correct_answers = round(correct_answers,4)
            correct_answers = str(correct_answers)
            average_bet = sum_true_odd_svm/number_of_won_games_svm
            average_bet = round(average_bet,4)
            average_bet = str(average_bet)
            average_gain = (true_class_svm/len(accepted_Y_svm))*(sum_true_odd_svm/(true_class_svm))*20-20
            average_gain = round(average_gain,4)
            average_gain = str(average_gain)
            # it gives the overview of the model efficiency on a QLabel just under the QPushButton
            self.label_1.setText("SVM correct answers (%):"+correct_answers+" \nAverage won odd :"+average_bet+" \nWhat you got betting 20 £ :"+average_gain)
            self.label_1.adjustSize()
            # if the QCheckBox is checked, it prints the next games and teams we have to bet on
            if self.btn_svm.checkState()==2:
                self.print_svm_next_games() 
        # if data are not processed
        else :
            self.label_svm.setText("Please process the data before  starting analyse them ...")
            self.label_svm.adjustSize()
            
    """
    This function prints the next games the user need to bet on according to the method he has choosen
    """
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
        for i in range(len(predictions)) :
        # we browse predictions and print the team and the game if our prediction are good enough to bet on it
            if (predictions[i][0] > self.threshold) :
                # if probability of home team victory is higher than the threshold, we bet on home team
                Games_you_need_to_bet_on.append("\nBet on %s during %s against %s on %s" %
                                                (df_dict_origin["HomeTeam"][i],df_dict_origin["HomeTeam"][i],
                                                 df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
            if (predictions[i][1] > self.threshold) :
                Games_you_need_to_bet_on.append("\nBet on a draw game during %s againt %s on %s"%
                                                (df_dict_origin["HomeTeam"][i],df_dict_origin["AwayTeam"][i],
                                                 df_dict_origin["Date"][i]))
            if (predictions[i][2] > self.threshold) :
                Games_you_need_to_bet_on.append("\nBet on %s during %s against %s on %s"%
                                                (df_dict_origin["AwayTeam"][i],df_dict_origin["HomeTeam"][i],
                                                 df_dict_origin["AwayTeam"][i],df_dict_origin["Date"][i]))
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
