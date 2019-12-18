#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:33:27 2019

@author: badrbelkeziz
"""
from crawl_country import get_next_games, get_teams_stats
from crawl_team import get_previous_games, get_next_game
from Analyse import *
from driver_setup_match import launch_driver
def launch_process(country) : 
    driver=launch_driver()
    time.sleep(2)
    next_games = get_next_games(country,driver) #récupérer les matchs de la prochaine journée et remplir le sheet
    print("OK")
    next_games_country = []
    get_teams_stats(country,next_games,driver) #Récupérer les Excel par équipe, mais ça ne marche plus ....
    for game in next_games :
        #print(game)
        next_dom = get_next_game(game["Dom"],country)
        #print(next_dom)
        previous_dom = get_previous_games(next_dom["dom"],country)
        expected_goals_dom = set_expected_goals_dom(next_dom,previous_dom,next_dom["dom"], country)
        next_ext = get_next_game(game["Ext"],country)
        previous_ext = get_previous_games(game["Ext"],country)
        expected_goals_ext = set_expected_goals_ext(next_ext,previous_ext,next_ext["ext"], country)
        predicted_score_dom = (expected_goals_dom[0]+expected_goals_ext[1])/2
        predicted_score_ext = (expected_goals_dom[1]+expected_goals_ext[0])/2
        print(next_dom["dom"] +" " + str(predicted_score_dom)+ "   -   " + str(predicted_score_ext) +"   "+ next_ext["ext"])
        
        
        
        
        
launch_process('Espagne')
#le modèle est bizarre ...

# L.15 : pour les équipes en Champions et en Europa League, on récupère les données de leurs matchs de poule, pour les autres on a quelques
#
    