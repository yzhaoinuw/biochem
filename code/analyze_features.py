#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 13:46:02 2022

@author: yuezhao
"""

import json
import numpy as np

DATA_PATH = '../data/'

broad2features_file = 'broad2features.json'
broad2smiles_file = 'broad2smiles.json'
features_file = 'features.npy'

with open(DATA_PATH+broad2features_file, 'r') as infile1:
    broad2features = json.load(infile1)
    
with open(DATA_PATH+broad2smiles_file, 'r') as infile1:
    broad2smiles = json.load(infile1)    
    
features = np.load(DATA_PATH+features_file)

#%%
rows = broad2features['BRD-A00100033-001-04-8']
features[rows]

sample_count = {broad: len(row_inds) for broad, row_inds in broad2features.items()}
sorted_count = sorted(sample_count.items(), key=lambda kv:- kv[1])