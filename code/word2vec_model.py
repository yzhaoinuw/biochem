# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 09:38:21 2022

@author: Yue
"""

import json
import numpy as np
from functools import partial
from multiprocessing import Pool, Manager

from rdkit import Chem
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence


DATA_PATH = "../data/"
WRITE_LOC = "../data/"
MODEL_PATH = "../model/"
broad2smiles_file = "broad2smiles.json"

with open(DATA_PATH + broad2smiles_file, "r") as infile1:
    broad2smiles = json.load(infile1)

#%%
class Mol2Vec:
    def __init__(self, model_path):
        self.model = word2vec.Word2Vec.load(model_path)

    def create_embedding(self, broad_id, d):
        smiles = broad2smiles[broad_id]
        mol = Chem.MolFromSmiles(smiles)
        sentence = mol2alt_sentence(mol, 1)
        mol_vec = np.array(
            [
                self.model.wv[substructure]
                for substructure in sentence
                if substructure in self.model.wv
            ]
        )
        mol_embedding = np.mean(mol_vec, axis=0)
        d[broad_id] = mol_embedding.tolist()


#%%
"""
if __name__ == '__main__':
    model_name = 'model_300dim.pkl'
    mol2vec = Mol2Vec(MODEL_PATH+model_name)
    manager = Manager()
    broad2vec = manager.dict()
    with Pool() as pool:
        pool.map(partial(mol2vec.create_embedding, d=broad2vec), [broad_id for broad_id, smiles in broad2smiles.items() if isinstance(smiles, str)])
        
    save_file = DATA_PATH + 'broad2vec.json'
    with open(save_file, 'w') as outfile1:
        json.dump(broad2vec.copy(), outfile1)
"""
