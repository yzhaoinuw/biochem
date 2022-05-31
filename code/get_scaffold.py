# -*- coding: utf-8 -*-
"""
Created on Tue May 31 01:59:30 2022

@author: Yue
"""

import json
from collections import defaultdict

import numpy as np

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

DATA_PATH = '../data/'
WRITE_LOC = '../data/'

broad2smiles_file = 'broad2smiles.json'

with open(DATA_PATH+broad2smiles_file, 'r') as infile1:
    broad2smiles = json.load(infile1)

#%%
scaffold_groups = defaultdict(list)
broad2scaffold = {}

n = 0
for broad_id, smiles in broad2smiles.items():
    if n%1000 == 0:
        print (n)
    n += 1
    if not isinstance(smiles, str):
        print (f"{broad_id} - {smiles}")
        broad2scaffold[broad_id] = None
        continue
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    broad2scaffold[broad_id] = scaffold_smiles
    scaffold_groups[scaffold_smiles].append(broad_id)
    
broad2scaffold_file = 'broad2scaffold.json'
#with open(WRITE_LOC+broad2scaffold_file, 'w') as outfile1:
#    json.dump(broad2scaffold, outfile1)