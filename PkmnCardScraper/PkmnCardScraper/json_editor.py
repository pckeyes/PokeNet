#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 19:59:22 2018

@author: piperkeyes
"""

import pandas as pd
import json

df = pd.read_json("scraped_data.json")
data_dicts = df.to_dict(orient="records")

for i in range(len(data_dicts)):
    data_str = data_dicts[i]["type"]
    
    #ignore non-pokemon cards
    try:
        index = data_str.index('HP<br>')
    except ValueError:
        data_dicts[i] = None
        continue
    
    #get card type
    index = data_str.index('–')
    data_str = data_str[index+2:]
    index_end = data_str.index('–')
    pkmn_type = data_str[:index_end-1]
    data_str = data_str[index_end+2:]

    #set card type
    data_dicts[i]["type"] = pkmn_type
    
    #get card HP
    #index = data_str.index('HP')
    index = data_str.index('HP')
    pkmn_hp = data_str[:index-1]
    
    #set card HP
    data_dicts[i]["HP"] = pkmn_hp
    
    #remove $ from prices
    data_dicts[i]["low price"] = data_dicts[i]["low price"][1:]
    data_dicts[i]["mid price"] = data_dicts[i]["mid price"][1:]
    data_dicts[i]["high price"] = data_dicts[i]["high price"][1:]

#remove ignored non-pokemon cards from data
data_dicts = list(filter(None, data_dicts))

data_json = json.dumps(data_dicts)
fp = open('cards.json', 'a')

# write to json file
fp.write(data_json)

# close the connection
fp.close()
