# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 12:55:09 2022

@author: Yue
"""

import json
from collections import defaultdict

import numpy as np


DATA_PATH = "../data/"
WRITE_LOC = "../data/"
broad2smiles_file = "broad2smiles.json"

with open(DATA_PATH + broad2smiles_file, "r") as infile1:
    broad2smiles = json.load(infile1)

broad2ind = {}
biochem_smiles = []
smiles_length = defaultdict(int)
for i, (broad_id, smiles) in enumerate(broad2smiles.items()):
    if not isinstance(smiles, str):
        continue
    smiles_length[len(smiles)] += 1
    broad2ind[broad_id] = i
    biochem_smiles.append(smiles)

biochem_smiles_file = "biochem_smiles.txt"
broad2ind_file = "broad2ind.json"


with open(WRITE_LOC + biochem_smiles_file, "w") as outfile1:
    outfile1.write("\n".join(biochem_smiles))

with open(WRITE_LOC + broad2ind_file, "w") as outfile1:
    json.dump(broad2ind, outfile1)


#%%
smiles_length_sorted = sorted(smiles_length.items(), key=lambda kv: -kv[1])
