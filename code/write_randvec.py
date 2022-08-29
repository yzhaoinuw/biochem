# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 19:39:38 2022

@author: Yue
"""

import json
import numpy as np


DATA_PATH = "../data/"

broad2smiles_file = "broad2smiles.json"

with open(DATA_PATH + broad2smiles_file, "r") as infile1:
    broad2smiles = json.load(infile1)

#%%
broad2randvec = {}
n = 0
for broad_id, smiles in broad2smiles.items():
    if n % 1000 == 0:
        print(n)
    n += 1
    if not isinstance(smiles, str):
        print(f"{broad_id} - {smiles}")
        broad2randvec[broad_id] = None
        continue
    if broad_id in broad2randvec:
        continue

    randvec = np.random.uniform(low=-1, high=1, size=(2048,)).tolist()
    broad2randvec[broad_id] = randvec

#%%
"""
save_file = DATA_PATH + 'broad2randvec.json'
with open(save_file, 'w') as outfile1:
    json.dump(broad2randvec, outfile1)
"""
