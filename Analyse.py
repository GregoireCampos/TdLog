#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:33:12 2019

@author: badrbelkeziz
"""

from driver_setup_match import launch_driver
from crawl_team import crawl_livescore
import pandas as pd
from datetime import date
import datetime
import math
from math import factorial, exp
def analyse_team(country,club) :
    data = json.loads(pd.read_csv(country+"/"+club+".csv").to_json(orient="records"))
    i=0
    last_game = data[i]
    status = last_game["Status"]
    while status ==None : 
        i+=1
        status = data[i]["Status"]
    last_game = data[i]
    return(last_game)
def analyse_game_with_shape (previous_game, next_game) : 
    date_to_game = get_date_to_game(previous_game, next_game)
    same_location = is_same_location(previous_game, next_game)
    goals_scored = goals_scored(previous_game, next_game)
    goals_in = goals_in(previous_game, next_game)
    next_opponent = get_opponent(next_game)
    next_opponent_shape = get_shape (next_game)
    previous_opponent = get_opponent(previous_game)
    previous_opponent_shape = get_shape (previous_game)    
    
def analyse_team_shape () :
    date_to_game = get_date_to_game(previous_game, next_game)
    same_location = is_same_location(previous_game, next_game)
    goals_scored = goals_scored(previous_game, next_game)
    goals_in = goals_in(previous_game, next_game)
    next_opponent = get_opponent(next_game)
    next_opponent_shape = get_shape (next_game)
    previous_opponent = get_opponent(previous_game)
    previous_opponent_shape = get_shape (previous_game) 
    
def get_date_to_game(previous_game, next_game) : 
    try : 
        return((datetime.datetime.strptime(next_game["date"],'%d.%m.%y')-datetime.datetime.strptime(previous_game["date"],'%d.%m.%y')).days)
    except : 
        try : 
            return((datetime.datetime.strptime(datetime.date.today().strftime("%d.%m.%y"),'%d.%m.%y')-datetime.datetime.strptime(previous_game["date"],'%d.%m.%y')).days)
        except : 
            return(0)
def is_same_location(previous_game, next_game) :
    return(previous_game["lieu"]==next_game["lieu"])
    
def goals_scores(previous_game) : 
    return(previous_game["But "+previous_game["lieu"]])
    
def goals_in(previous_game) : 
    if previous_game["lieu"] == "dom" : 
        return(previous_game["But ext"])
    else : 
        return(previous_game["But dom"])
        
def get_opponent(game) : 
    if game["lieu"]=="dom" : 
        return (game["ext"])
    else : 
        return(game["dom"])

def get_shape(game,country) : 
    opponent = get_opponent(game)
    game_date = datetime.datetime.strptime(game["date"],'%d.%m.%y')
    opponent_data = json.loads(pd.read_csv(country+"/"+opponent+".csv").to_json(orient="records"))
    for data in opponent_data : 
        if (datetime.datetime.strptime(game["date"],'%d.%m.%y')-datetime.datetime.strptime(data["date"],'%d.%m.%y')).days<0 : 
            continue;
        #else : 
            
def set_expected_goals_dom(next_game,previous_dom,club, country) :
    deltatime = []
    goals_scored =[]
    goals_received=[]
    locations=[]
    for game in previous_dom :
        #print(game)
        deltatime += [get_date_to_game(game,next_game)]
        goals_scored +=[goals_scores(game)]
        goals_received +=[goals_in(game)]
        locations+=[is_same_location(game, next_game)]
    expected_goals_scored = 0.0
    expected_goals_received = 0.0
    for i in range(len(deltatime)) : 
        if locations[i]==True : 
                expected_goals_scored += float(goals_scored[i])*(4.8992**(i+1))*exp(-(i+1))/factorial(i+1)
                expected_goals_received += float(goals_received[i])*(4.8992**(i+1))*exp(-(i+1))/factorial(i+1)
        else : 
                expected_goals_scored += float(goals_scored[i])*(4.8992**(i+1))*exp(-(i+1))*0.95/factorial(i+1)
                expected_goals_received += float(goals_received[i])*(4.8992**(i+1))*exp(-(i+1))*0.95/factorial(i+1)
    expected_goals_scored = expected_goals_scored/5
    expected_goals_received = expected_goals_received/5
    return(expected_goals_scored,expected_goals_received)
    
def set_expected_goals_ext(next_game,previous_ext,club, country) :
    deltatime = []
    goals_scored =[]
    goals_received=[]
    locations=[]
    for game in previous_ext :
        #print(game)
        deltatime += [get_date_to_game(game,next_game)]
        goals_scored +=[goals_scores(game)]
        goals_received +=[goals_in(game)]
        locations+=[is_same_location(game, next_game)]
    expected_goals_scored = 0.0
    expected_goals_received = 0.0
    for i in range(len(deltatime)) : 
        if locations[i]==True : 
                expected_goals_scored += float(goals_scored[i])*(4.8992**(i+1))*exp(-(i+1))/factorial(i+1)
                expected_goals_received += float(goals_received[i])*(4.8992**(i+1))*exp(-(i+1))/factorial(i+1)
        else : 
                expected_goals_scored += float(goals_scored[i])*(4.8992**(i+1))*exp(-(i+1))*0.95/factorial(i+1)
                expected_goals_received += float(goals_received[i])*(4.8992**(i+1))*exp(-(i+1))*0.95/factorial(i+1)
    expected_goals_scored = expected_goals_scored/5
    expected_goals_received = expected_goals_received/5
    return(expected_goals_scored,expected_goals_received)
         
         
        
    
    
    