#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:54:50 2019

@author: badrbelkeziz
"""

import pandas as pd
import json 
def crawl_livescore(country,club,link,driver) :
    #donne les scores des matchs en direct
    driver.get(link)
    div_score = driver.find_element_by_xpath("//div[@id='livescore']")
    championnats = div_score.find_elements_by_xpath(".//div[contains(@class,'panel-info')]")[:5]
    games = []
    for championnat in championnats : 
        ligue = championnat.find_element_by_xpath(".//div[contains(@class,'panel-heading')]").text
        div_games = championnat.find_element_by_xpath(".//div[contains(@class,'panel-body')]")
        div_divisions = div_games.find_elements_by_xpath(".//table")
        for div_game in div_divisions[:-1] : 
            matches = div_game.find_elements_by_xpath(".//tr")
            lieu=""
            for match in matches : 
            #try : 
                #print(match.text)
                championship = ligue
                try : 
                    heure = match.find_element_by_xpath(".//td[contains(@class,'lm1')]").text
                    date =  match.find_element_by_xpath(".//td[contains(@class,'lm2')]").text
                    Jeu =  match.find_element_by_xpath(".//td[contains(@class,'lm3')]")
                    dom = Jeu.find_element_by_xpath(".//span[contains(@class,'lm3_eq1')]").text
                    #sprint(dom)
                    ext = Jeu.find_element_by_xpath(".//span[contains(@class,'lm3_eq2')]").text
                    score = Jeu.find_element_by_xpath(".//span[contains(@class,'lm3_score')]").text
                except : 
                    print(match.text)
                    continue;
            #print(But_dom>But_ext)
                if dom==club: 
                    lieu = "dom";
                elif ext ==club : 
                    lieu = "ext"
                else : 
                    continue;
                status = ""
                if score == " " or score =="" :
                    status = "NA"
                    score ="XX"
                else : 
                    But_dom = score[0]
                    But_ext = score[-1]
                    if dom==club : 
                        if But_dom>But_ext : 
                            status = "Victoire"
                        elif But_dom==But_ext : 
                            status = "Nul"
                        else : 
                            status = "Defaite"
                    else : 
                        if But_dom>But_ext : 
                            status = "Defaite"
                        elif But_dom==But_ext : 
                            status = "Nul"
                        else : 
                            status = "Victoire"                
                ext = Jeu.find_element_by_xpath(".//span[contains(@class,'lm3_eq2')]").text
                games+=[{"ligue": ligue, "heure" : heure, "date" : date, "dom": dom,  "lieu": lieu,  "But dom" : score[0], "ext": ext, "But ext" : score[-1], "Status" : status}]
            #except : 
            #    print(match.text)
                #break; 
    df = pd.DataFrame(games)
    df.to_csv(country+"/"+club+".csv")
    return(games)
            
def get_previous_games(club,country) :
    data = json.loads(pd.read_csv(country+"/"+club+".csv").to_json(orient="records"))
    previous_games = []
    i=0
    for game in data : 
        if i>=5 :
            break;
        if game["Status"]!= None : 
            previous_games+=[game]
            i+=1
    return(previous_games)
def get_next_game(club,country) : 
    data = json.loads(pd.read_csv(country+"/"+club+".csv").to_json(orient="records"))
    previous_games = []
    i=0
    for game in data : 
        if game["Status"]== None : 
            i+=1
        else :
            return(data[i-1])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    