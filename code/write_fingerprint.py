# -*- coding: utf-8 -*-
"""
Created on Wed May 11 00:15:29 2022

@author: Yue
"""

import json

import numpy as np

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

DATA_PATH = "../data/"

broad2smiles_file = "broad2smiles.json"

with open(DATA_PATH + broad2smiles_file, "r") as infile1:
    broad2smiles = json.load(infile1)

#%%
broad2fingerprint = {}
n = 0
for broad_id, smiles in broad2smiles.items():
    if n % 1000 == 0:
        print(n)
    n += 1
    if not isinstance(smiles, str):
        print(f"{broad_id} - {smiles}")
        broad2fingerprint[broad_id] = None
        continue
    mol = Chem.MolFromSmiles(smiles)
    fingerprint = Chem.RDKFingerprint(mol).ToBitString()
    broad2fingerprint[broad_id] = fingerprint

# smiles = "NS(=O)(=O)c1cc2c(NC(CSCC=C)NS2(=O)=O)cc1Cl"
# scaffold = MurckoScaffold.GetScaffoldForMol(mol)
#%%
"""
save_file = DATA_PATH + 'broad2fingerprints.json'
with open(save_file, 'w') as outfile1:
    json.dump(broad2fingerprint, outfile1)
"""
