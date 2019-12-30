#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 12:28:53 2019

@author: badr and gregoire
"""
import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.read_csv("F2.csv")
df_dict = df.to_dict()
keys_to_keep = ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR","B365H","B365D","B365A"]
dataset = {}
for key in keys_to_keep : 
    dataset[key] = df_dict[key]
    
# Initialise toutes les différences de but à 0
    
dataset["HTGDBG"]= {}
#home team goal average before game 
dataset["HTGDAG"]= {}
dataset["ATGDBG"]= {}
dataset["ATGDAG"]= {}
for i in dataset['Date'].keys() : 
    dataset["HTGDBG"][i] = -200
    dataset["HTGDAG"][i] = -200
    dataset["ATGDBG"][i] = -200
    dataset["ATGDAG"][i] = -200
    

# Initialise tous les points à 0
dataset["HTPBG"]= {}
dataset["HTPAG"]= {}
dataset["ATPBG"]= {}
dataset["ATPAG"]= {}
dataset["RATIO"] = {}
dataset["CC"] = {}
dataset["GO"] = {}
for i in dataset['Date'].keys() : 
    dataset["HTPBG"][i] = -200
    dataset["HTPAG"][i] = -200
    dataset["ATPBG"][i] = -200
    dataset["ATPAG"][i] = -200
    dataset["RATIO"][i] = -200
    dataset["CC"][i] = 20
    dataset["GO"][i] = False
# on initialise à -200 pour qu'on puisse savoir si les données ont été complétées
    
#Je me prépare a mettre les différences de but avant et après match, les points avant et après match que j'initialise à 0
dataset_df = pd.DataFrame.from_dict(dataset) # à quoi ça sert ? : il remet sous forme de tableau
dataset_df.to_csv("F2_processed.csv")

#Fonction pour ajouter les différences de buts avant et apres match des équipes à domicile et à l'exterieur pour chaque match
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
            if dataset["HomeTeam"][i]==team : #l'équipe qu'on regarde joue a domicile
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
    dataset_df.to_csv("F2_processed.csv")

    
#la je lance cette fonction pour remplir toute les différences de buts
    
for team in list_teams : 
    add_data_goal_diff(dataset,team)
    
# Même process pour ajouter les points
    
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
    dataset_df.to_csv("F2_processed.csv")

for team in list_teams : 
    add_data_points(dataset,team)
    
mise = 20
# On considère que sur tous les matchs pariés, on a mis 20 euros
a = 0.45
b = 1 - a
#ce sont les coefficients du ratio, on cherche ceux qui marchent le mieux
# le ratio est de la forme : a * (diff_buts) + b * (diff_points)

""" Valeur du seuil """
seuil = 20
# je l'ai fixé de manière aléatoire, mais on peut regarder le CC final et voir pour quel seuil il est max

"""Maintenant, on calcule le ratio sur chaque match"""
def add_data_ratio(dataset):
    nb_games = len(dataset['Date'].keys())
    for i in range (nb_games):
        if dataset["HTGDBG"][i] != -200 and dataset["ATGDBG"][i] != -200 and dataset["HTPBG"][i] != -200 and dataset["ATPBG"][i] != -200 :
            dataset["RATIO"][i] = a * (dataset["HTGDBG"][i] - dataset["ATGDBG"][i]) + b * (dataset["HTPBG"][i] - dataset["ATPBG"][i])
    dataset_df = pd.DataFrame.from_dict(dataset)
    dataset_df.to_csv("F2_processed.csv")
add_data_ratio(dataset)


"""Maintenant, on aimerait remplir la colonne "CC" (current cash) qui bouge quand on prend un pari, et la colonne GO qui affiche True
si on a pris le pari et False sinon"""   
def add_data_cc(dataset, seuil_choisi):
    # seuil_choisi = seuil # grisé si on veut tracer le cash final en fonction du seuil choisi
    nb_games = len(dataset['Date'].keys())
    previous_cc = 0
    for i in range (nb_games):
        if dataset["RATIO"][i] + dataset["B365H"][i] >= seuil_choisi :
            dataset["GO"][i] = True
            if dataset["FTR"][i] == "H":
                # le pari a été gagnant
                dataset["CC"][i] = previous_cc + mise * dataset["B365H"][i] - mise
            else :
                dataset["CC"][i] = previous_cc - mise
            previous_cc = dataset["CC"][i]
        else :
            dataset["CC"][i] = previous_cc
    dataset_df = pd.DataFrame.from_dict(dataset)
    dataset_df.to_csv("F2_processed.csv")

add_data_cc(dataset, seuil)
nb_games = len(dataset['Date'].keys())
final_cash = dataset["CC"][nb_games-1]

date = "03/03/2019"

"""Maintenant, on demande au programme de nous donner les matchs à parier de la semaine"""

"""
def where_should_you_go_this_week(dataset):
    # pb quand les dates sont du type : xx/xx/18 qu'il faudrait changer en xx/xx/2018 automatiquement
    nb_games = len(dataset['Date'].keys())
    i = 0
    newdate_match = time.strptime(dataset["Date"][0], "%d/%m/%y")
    newdate_now = time.strptime(date, "%d/%m/%y")
    while(newdate_now > newdate_match) and (i<nb_games -1):
        i = i+1
        newdate_match = time.strptime(dataset["Date"][i], "%d/%m/%y")
    if i == nb_games -1 :
        print("pb de date")
    # On a récupéré le prochain match
    Week_matches = []
    next_week = i+7
    while (i < next_week):
        Week_matches.append([dataset["HomeTeam"][i], dataset["AwayTeam"][i], dataset["GO"][i]])
        i = i+1
    print(Week_matches)
    
where_should_you_go_this_week(dataset)
"""

""" Tracer l'évolution du cash final en fonction du seuil choisi """

""" Tracer l'évolution du current cash en fonction du temps """ 
plt.plot(list(dataset["CC"].keys()),list(dataset["CC"].values()), label = "20")

""" Tracer l'évolution du current cash en fonction du temps et du seuil choisi """

seuil = -100
add_data_cc(dataset, seuil)
plt.plot(list(dataset["CC"].keys()),list(dataset["CC"].values()), label = "-50")
seuil = 30
add_data_cc(dataset, seuil)
plt.plot(list(dataset["CC"].keys()),list(dataset["CC"].values()), label = "30")
seuil = 15
add_data_cc(dataset, seuil)
plt.plot(list(dataset["CC"].keys()),list(dataset["CC"].values()), label = "15")
seuil = 0
add_data_cc(dataset, seuil)
plt.plot(list(dataset["CC"].keys()), list(dataset["CC"].values()), label = "0")
plt.legend()
plt.xlabel("Nombres de matchs joués")
plt.ylabel("Current cash")
plt.show()

""" Calcul de la rentabilité : regarder combien de paris on a pris et regarder le gain """

def calcul_renta(dataset, seuil):
    nb_games = len(dataset['Date'].keys())
    mise_totale = 0
    for i in range (nb_games):
        if dataset["RATIO"][i] + dataset["B365H"][i] >= seuil :
            mise_totale = mise_totale + mise
    # On a récupéré la mise totale
    # Il faut récupérer le gain
    add_data_cc(dataset, seuil)
    return (dataset["CC"][nb_games - 1] / mise_totale)

print(calcul_renta(dataset, 15))
# Quand on pose un seuil de 15 avec a = 0.25 et b = 0.75, on a une rentabilité de 8.6 % : mise totale de 940 euros et gain de 80.4 euros
# Quand on pose un seuil de 15 avec a = 0.45 et b = 0.55, on a une rentabilité de 10.8 %
# 14 % de renta avec un seuil de -10 et a = 0.45 ???? BIZARRE, l'autre programme a l'air mieux il y a du avoir un pb ou alors cette saison est chelou
# sur un an tu peux aussi bien gagner +10% que faire -8% alors que si tu prends seuil = 15 tu fais 10% les deux fois
     