#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 12:28:53 2019

@author: badr
"""

import pandas as pd
df = pd.read_csv("F1.csv")
df_dict = df.to_dict()
keys_to_keep = ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR","B365H","B365D","B365A"]
dataset = {}
for key in keys_to_keep : 
    dataset[key] = df_dict[key]
dataset["HTGDBG"]= {}
dataset["HTGDAG"]= {}
dataset["ATGDBG"]= {}
dataset["ATGDAG"]= {}
for i in dataset['Date'].keys() : 
    dataset["HTGDBG"][i] = 0
    dataset["HTGDAG"][i] = 0
    dataset["ATGDBG"][i] = 0
    dataset["ATGDAG"][i] = 0

dataset["HTPBG"]= {}
dataset["HTPAG"]= {}
dataset["ATPBG"]= {}
dataset["ATPAG"]= {}
for i in dataset['Date'].keys() : 
    dataset["HTPBG"][i] = 0
    dataset["HTPAG"][i] = 0
    dataset["ATPBG"][i] = 0
    dataset["ATPAG"][i] = 0
    
#Je me prépare a mettre les différences de but avant et après match, les points avant et après match que j'initialise à 0
dataset_df = pd.DataFrame.from_dict(dataset)
dataset_df.to_csv("F1_processed.csv")

#Fonction pour ajouter les différences de buts avant et apres match des équipes a domicile et a l'exterieur pour chaque match
#Prend en argument le dataset créé en haut de fichier et le nom de l'équipe à traiter (je traite les équipes une par une)
#Donc il faudra lancer la fonction sur la liste suivante pour avoir le tout

list_teams= ["Marseille", "Toulouse","Monaco","Lille","Rennes","Nimes","Reims","Lyon","Caen","Guingamp","Strasbourg","Paris SG"
             ,"Montpellier","Amiens","Dijon","Nantes","St Etienne","Angers","Nice","Bordeaux"]

def add_data_goal_diff(dataset,team) :
    #Home team goal difference before and after the game = différence de buts de l'équipe à domicile avant et après le match
    nb_games = len(dataset['Date'].keys())
    last_game_index = None;
    for i in range (nb_games) :
        if dataset["HomeTeam"][i]!=team and dataset["AwayTeam"][i]!=team :
            #le match ne concerne pas l'équipe en question
            pass;
        else : 
             #il faut prendre en compte ce match pour l'équipe team
            if dataset["HomeTeam"][i]==team : #homeTeam joue a domicile
                if last_game_index == None: #l'équipe n'a encore jamais joué
                    dataset["HTGDBG"][i] = 0
                    dataset["HTGDAG"][i] = dataset["HTGDBG"][i] + dataset["FTHG"][i] - dataset["FTAG"][i] 
                else :
                    if dataset["HomeTeam"][last_game_index]==team : #Son dernier match était à domicile
                        dataset["HTGDBG"][i] = dataset["HTGDAG"][last_game_index]
                        dataset["HTGDAG"][i] = dataset["HTGDAG"][last_game_index] + dataset["FTHG"][i] - dataset["FTAG"][i]
                    else : #Son dernier match était à l'exterieur
                        dataset["HTGDBG"][i] = dataset["ATGDAG"][last_game_index]
                        dataset["HTGDAG"][i] = dataset["ATGDAG"][last_game_index] + dataset["FTHG"][i] - dataset["FTAG"][i]
            else :
                #dataset["awayTeam"]==team  
                if last_game_index == None: #l'équipe n'a encore jamais joué
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
    dataset_df.to_csv("F1_processed.csv")

    
#la je lance cette fonction pour remplir toute les différences de buts
    
for team in list_teams : 
    add_data_goal_diff(dataset,team)
    
    
def add_data_points(dataset,team) :
    #Home team goal difference before and after the game = différence de buts de l'équipe à domicile avant et après le match
    nb_games = len(dataset['Date'].keys())
    last_game_index = None;
    for i in range (nb_games) :
        if dataset["HomeTeam"][i]!=team and dataset["AwayTeam"][i]!=team :
            #le match ne concerne pas l'équipe en question
            pass;
        else : 
             #il faut prendre en compte ce match pour l'équipe team
            if dataset["HomeTeam"][i]==team : #Team joue a domicile
                if last_game_index == None: #l'équipe n'a encore jamais joué
                    dataset["HTPBG"][i] = 0
                    if dataset["FTR"][i]=="H" : 
                        dataset["HTPAG"][i] = dataset["HTPBG"][i] + 3
                    elif dataset["FTR"][i]=="D" : 
                        dataset["HTPAG"][i] = dataset["HTPBG"][i] + 1
                    else :
                        dataset["HTPAG"][i] = dataset["HTPBG"][i]
                else :
                    if dataset["HomeTeam"][last_game_index]==team : #Son dernier match était à domicile
                        dataset["HTPBG"][i] = dataset["HTPAG"][last_game_index]
                        if dataset["FTR"][i]=="H" : 
                            dataset["HTPAG"][i] = dataset["HTPBG"][i] + 3
                        elif dataset["FTR"][i]=="D" : 
                            dataset["HTPAG"][i] = dataset["HTPBG"][i] + 1
                        else :
                            dataset["HTPAG"][i] = dataset["HTPBG"][i]
                    else : #Son dernier match était à l'exterieur
                        dataset["HTPBG"][i] = dataset["ATPAG"][last_game_index]
                        if dataset["FTR"][i]=="H" : 
                            dataset["HTPAG"][i] = dataset["HTPBG"][i] + 3
                        elif dataset["FTR"][i]=="D" : 
                            dataset["HTPAG"][i] = dataset["HTPBG"][i] + 1
                        else :
                            dataset["HTPAG"][i] = dataset["HTPBG"][i]
            else :
                #dataset["awayTeam"]==team  
                if last_game_index == None: #l'équipe n'a encore jamais joué
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
    dataset_df.to_csv("F1_processed.csv")

for team in list_teams : 
    add_data_points(dataset,team)