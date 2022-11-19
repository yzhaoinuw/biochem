# -*- coding: utf-8 -*-
"""
Created on Wed May 11 00:15:29 2022

@author: Yue
"""

import json

from rdkit import Chem
from mordred import Calculator, descriptors

DATA_PATH = "../data/"

broad2smiles_file = "broad2smiles.json"
save_file = DATA_PATH + "broad2mordred.json"

with open(DATA_PATH + broad2smiles_file, "r") as infile1:
    broad2smiles = json.load(infile1)

#%%
calc = Calculator(descriptors, ignore_3D=True)
try:
    with open(save_file, "r") as infile1:
        broad2fingerprint = json.load(infile1)
except FileNotFoundError:
    broad2fingerprint = {}

n = 0
for broad_id, smiles in broad2smiles.items():
    if broad_id in broad2fingerprint:
        continue

    if not isinstance(smiles, str):
        print(f"{broad_id} - {smiles}")
        broad2fingerprint[broad_id] = None
        continue

    n += 1
    if n % 100 == 0:
        break

    mol = Chem.MolFromSmiles(smiles)
    # print (smiles)
    fingerprint = calc(mol)
    # fingerprint = Chem.RDKFingerprint(mol).ToBitString()
    broad2fingerprint[broad_id] = fingerprint[:]


# smiles = "NS(=O)(=O)c1cc2c(NC(CSCC=C)NS2(=O)=O)cc1Cl"
# scaffold = MurckoScaffold.GetScaffoldForMol(mol)
#%%
outfile1 = save_file
# with open(save_file, 'w') as outfile1:
#    json.dump(broad2fingerprint, outfile1)
