#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:20:26 2019

@author: badrbelkeziz
"""
from driver_setup_match import launch_driver
from datetime import date
driver=launch_driver()
country = 'Espagne'


from crawl_team import crawl_livescore
import pandas as pd

def get_next_games(country,driver) : 
    link = "https://www.matchendirect.fr/"+country+"/"
    driver.get(link)
    division = driver.find_element_by_xpath('//table[1]')
    """
    Mettre la date
    """
    dates = division.find_elements_by_xpath('.//tbody')
    next_games = [day]
    #print(dates)
    for journee in dates : 
        number = len(journee.find_elements_by_xpath('.//tr'))
        #print(journee.text)
        #print(number)
        for i in range(number) : 
            next_games+= [{}]
            next_games[-1]["Time"]=journee.find_element_by_xpath('.//tr['+str(i+1)+']//td[1]').text
            next_games[-1]["Dom"]=journee.find_element_by_xpath('.//tr['+str(i+1)+']//td[3]//a//span[1]').text
            next_games[-1]["Ext"]=journee.find_element_by_xpath('.//tr['+str(i+1)+']/td[3]//a//span[3]').text  
        #print(next_games)

    jour = date.today().strftime("%d-%m-%Y")
    df = pd.DataFrame(next_games)
    df.to_csv(country+"/Next"+jour+".csv")
    return(next_games)   

def get_list_links(competition, competition_link) : 
    driver=launch_driver()
    driver.get(competition_link)
    teams = {}
    div = driver.find_element_by_xpath("//div[contains(@class, 'col-md-12 col-lg-10')]")
    for team in div.find_elements_by_xpath('.//a'):
        teams[team.text]= team.get_attribute("href")
    print(teams)
    df = pd.DataFrame.from_dict(teams, orient = "index")
    df.to_csv(competition+".csv")
    
def get_championnats() :
    df = pd.read_csv("championnats.csv")
    dico = df.to_dict(orient="index")
    for i in range(len(dico.keys())) :
        get_list_links(dico[i]["Name"],dico[i]["Link"])
    return(df)
        

"""
A changer pour avoir les bonnes dates
Il faut également le traduire avec différentes cases dans Excel pour pouvoir traiter le dossier correctement
"""
def get_teams_stats(country,next_games,driver) :
    links = pd.read_csv(country+"/"+country+".csv", header=None, index_col = 0).to_dict(orient="index")
    #print(links)
    for game in next_games : 
        print(game)
        crawl_livescore(country,game["Dom"],links[game["Dom"]][1], driver)
        crawl_livescore(country,game["Ext"],links[game["Ext"]][1], driver)