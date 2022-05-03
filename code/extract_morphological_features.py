#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:53:58 2022

@author: yuezhao
"""

import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd


PROFILE_PATH = '../data/profiles_mean_well/'
folder = os.listdir(PROFILE_PATH)

broad2features = defaultdict(list)
n = 0
features = []

for k, file in enumerate(folder):
    if not file.endswith('.csv'):
        continue
    
    if k%10 == 0:
        print (f'{k}th file')
    df = pd.read_csv(PROFILE_PATH+file)
    broad_id = df['Metadata_broad_sample'].to_numpy()
    df = df.iloc[:, 17:]
    features.append(df.to_numpy())
    for broad in broad_id:
        
        # update broad2features mapping
        broad2features[broad].append(n)
        n += 1
        
features = np.vstack(features)        
    
#%%
'''
save_file = '../data/featues.npy'
with open(save_file, 'wb') as outfile1:
    np.save(outfile1, features)
    
#%%
broad2features_file = '../data/broad2features.json'
with open(broad2features_file, 'w') as outfile2:
    json.dump(broad2features, outfile2)
'''