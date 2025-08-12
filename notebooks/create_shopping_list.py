# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 14:40:30 2025

@author: bianru
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\bianru\\Downloads\\metadata.csv')


credit =(36805-1)/2

idx_list = df['taxonID_index'].value_counts()[::-1]
wanted_taxon_idxs = idx_list.cumsum()<credit

wanted = wanted_taxon_idxs.index[wanted_taxon_idxs]

least_common_taxons = df.loc[df['taxonID_index'].isin(wanted), 'filename_index']

least_common_taxons_df = pd.DataFrame(least_common_taxons)
least_common_taxons_df['want'] = 'Habitat'

least_common_taxons_df[['filename_index', 'want']].to_csv(
    "C:/bianru/courses/multimodal_learning/shopping_list.csv", 
    index=False, header=False)

