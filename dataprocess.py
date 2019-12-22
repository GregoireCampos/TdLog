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
dataset_df = pd.DataFrame.from_dict(dataset)
dataset_df.to_csv("F1_processed.csv")



#print(df.keys)