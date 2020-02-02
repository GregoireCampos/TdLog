#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 12:28:53 2019

@author: badr and gregoire
"""
import time
import pandas as pd

def add_data_goal_diff(dataset,team,file_name) :
    #Home team goal difference before and after the game = différence de buts de l'équipe à domicile avant et après le match
    nb_games = len(dataset['Date'].keys())
    last_game_index = None;
    for i in range (nb_games) :
        if dataset["HomeTeam"][i]!=team and dataset["AwayTeam"][i]!=team :
            #the game is doesn't concern the team
            pass;
        else : 
             #Need to take this game into account
            if dataset["HomeTeam"][i]==team : #team plays home
                if last_game_index == None: #team never played yet
                    dataset["HTGDBG"][i] = 0
                    dataset["HTGDAG"][i] = dataset["HTGDBG"][i] + dataset["FTHG"][i] - dataset["FTAG"][i] 
                else :
                    if dataset["HomeTeam"][last_game_index]==team : #last game was home
                        dataset["HTGDBG"][i] = dataset["HTGDAG"][last_game_index]
                        dataset["HTGDAG"][i] = dataset["HTGDAG"][last_game_index] + dataset["FTHG"][i] - dataset["FTAG"][i]
                    else : #last gamz was away
                        dataset["HTGDBG"][i] = dataset["ATGDAG"][last_game_index]
                        dataset["HTGDAG"][i] = dataset["ATGDAG"][last_game_index] + dataset["FTHG"][i] - dataset["FTAG"][i]
            else :
                #dataset["awayTeam"]==team  
                if last_game_index == None: #team never played yet
                    dataset["ATGDBG"][i] = 0
                    dataset["ATGDAG"][i] = dataset["ATGDBG"][i] - dataset["FTHG"][i] + dataset["FTAG"][i] 
                else :
                    if dataset["HomeTeam"][last_game_index]==team :
                        dataset["ATGDBG"][i] = dataset["HTGDAG"][last_game_index]
                        dataset["ATGDAG"][i] = dataset["HTGDAG"][last_game_index] + dataset["FTAG"][i] - dataset["FTHG"][i]

                    else :
                        dataset["ATGDBG"][i] = dataset["ATGDAG"][last_game_index]
                        dataset["ATGDAG"][i] = dataset["ATGDAG"][last_game_index] + dataset["FTAG"][i] - dataset["FTHG"][i]
            last_game_index = i
    dataset_df = pd.DataFrame.from_dict(dataset)
    dataset_df.to_csv("Training_Files/France/"+file_name+"_processed.csv")

    
#Same process to add the points 
    
def add_data_points(dataset,team,file_name) :
    #Home team goal difference before and after the game
    nb_games = len(dataset['Date'].keys())
    last_game_index = None;
    for i in range (nb_games) :
        if dataset["HomeTeam"][i]!=team and dataset["AwayTeam"][i]!=team :
            #the game is not about this team
            pass;
        else : 
             #we need to take this game into account
            if dataset["HomeTeam"][i]==team : #Team plays home
                if last_game_index == None: #Team never played yet
                    dataset["HTPBG"][i] = 0
                    if dataset["FTR"][i]=="H" : 
                        dataset["HTPAG"][i] = dataset["HTPBG"][i] + 3
                    elif dataset["FTR"][i]=="D" : 
                        dataset["HTPAG"][i] = dataset["HTPBG"][i] + 1
                    else :
                        dataset["HTPAG"][i] = dataset["HTPBG"][i]
                else :
                    if dataset["HomeTeam"][last_game_index]==team : #last game was home
                        dataset["HTPBG"][i] = dataset["HTPAG"][last_game_index]
                        if dataset["FTR"][i]=="H" : 
                            dataset["HTPAG"][i] = dataset["HTPBG"][i] + 3
                        elif dataset["FTR"][i]=="D" : 
                            dataset["HTPAG"][i] = dataset["HTPBG"][i] + 1
                        else :
                            dataset["HTPAG"][i] = dataset["HTPBG"][i]
                    else : #last game was away
                        dataset["HTPBG"][i] = dataset["ATPAG"][last_game_index]
                        if dataset["FTR"][i]=="H" : 
                            dataset["HTPAG"][i] = dataset["HTPBG"][i] + 3
                        elif dataset["FTR"][i]=="D" : 
                            dataset["HTPAG"][i] = dataset["HTPBG"][i] + 1
                        else :
                            dataset["HTPAG"][i] = dataset["HTPBG"][i]
            else :
                #dataset["awayTeam"]==team  
                if last_game_index == None: #team never played yet
                    dataset["ATPBG"][i] = 0
                    if dataset["FTR"][i]=="A" : 
                        dataset["ATPAG"][i] = dataset["ATPBG"][i] + 3
                    elif dataset["FTR"][i]=="D" : 
                        dataset["ATPAG"][i] = dataset["ATPBG"][i] + 1
                    else :
                        dataset["ATPAG"][i] = dataset["ATPBG"][i]
                else :
                    if dataset["HomeTeam"][last_game_index]==team :
                        dataset["ATPBG"][i] = dataset["HTPAG"][last_game_index]
                        if dataset["FTR"][i]=="A" : 
                            dataset["ATPAG"][i] = dataset["ATPBG"][i] + 3
                        elif dataset["FTR"][i]=="D" : 
                            dataset["ATPAG"][i] = dataset["ATPBG"][i] + 1
                        else :
                            dataset["ATPAG"][i] = dataset["ATPBG"][i]
                    else :
                        dataset["ATPBG"][i] = dataset["ATPAG"][last_game_index]
                        if dataset["FTR"][i]=="A" : 
                            dataset["ATPAG"][i] = dataset["ATPBG"][i] + 3
                        elif dataset["FTR"][i]=="D" : 
                            dataset["ATPAG"][i] = dataset["ATPBG"][i] + 1
                        else :
                            dataset["ATPAG"][i] = dataset["ATPBG"][i]
            last_game_index = i
    dataset_df = pd.DataFrame.from_dict(dataset)
    dataset_df.to_csv("Training_Files/France/"+file_name+"_processed.csv")


keys_to_keep = ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR","B365H","B365D","B365A"]
list_teams= ["Marseille", "Toulouse","Monaco","Lille","Rennes","Nimes","Reims","Lyon","Caen","Guingamp","Strasbourg","Paris SG"
             ,"Montpellier","Amiens","Dijon","Nantes","St Etienne","Angers","Nice","Bordeaux","Lens", "Nancy", "Sochaux", "Valenciennes", "Metz"
             ,"Lorient","Auxerre","Le Mans", "Le Havre", "Grenoble","Boulogne","Arles","Brest","Ajaccio","Evian Thonon Gaillard","Bastia", "Ajaccio GFCO"]
list_files = ["7-8","8-9","9-10","10-11","11-12","12-13",
              "13-14","14-15","15-16","16-17","17-18","18-19"]
def process_files() :   
    for file_name in list_files:
        df = pd.read_csv("Training_Files/France/"+file_name+".csv")
        df_dict = df.to_dict()
        dataset = {}
        for key in keys_to_keep : 
            dataset[key] = df_dict[key]
    
        #Initializing every new data
        #home team or Away team    goal average before and after the game 
        dataset["HTGDBG"]= {}
        dataset["HTGDAG"]= {}
        dataset["ATGDBG"]= {}
        dataset["ATGDAG"]= {}
        #points before and after the game
        dataset["HTPBG"]= {}
        dataset["HTPAG"]= {}
        dataset["ATPBG"]= {}
        dataset["ATPAG"]= {}
    

        #Add points and goal average, team by team 

        for team in list_teams : 
            add_data_points(dataset,team,file_name)
        for team in list_teams : 
            add_data_goal_diff(dataset,team,file_name)
            
     
        
            
        
        

    
    
    