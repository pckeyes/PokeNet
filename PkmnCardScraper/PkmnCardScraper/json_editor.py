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
    #remove $ from prices
    data_dicts[i]["low price"] = data_dicts[i]["low price"][1:]
    data_dicts[i]["mid price"] = data_dicts[i]["mid price"][1:]
    data_dicts[i]["high price"] = data_dicts[i]["high price"][1:]
    
    #split name into name and set
    name_and_set = data_dicts[i]["name"]
    this_name, this_set = name_and_set.split("(")
    data_dicts[i]["name"]  = this_name[:len(this_name)-1]
    data_dicts[i]["set"] = this_set[:len(this_set)-1]
    
    data_str = data_dicts[i]["type"]
    
    #set type and HP of non-pokemon cards to N/A
    try:
        index = data_str.index('HP<br>')
    except ValueError:
        data_dicts[i]["type"] = 'N/A'
        data_dicts[i]["HP"] = 'N/A'
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

#write data to new json file
data_json = json.dumps(data_dicts)
fp = open('cards.json', 'a')
fp.write(data_json)
fp.close()
