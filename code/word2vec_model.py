# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 09:38:21 2022

@author: Yue
"""

import json
import numpy as np

from rdkit import Chem
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from mol2vec.helpers import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg


model = word2vec.Word2Vec.load('../model/model_300dim.pkl')

DATA_PATH = '../data/'
WRITE_LOC = '../data/'

broad2smiles_file = 'broad2smiles.json'

with open(DATA_PATH+broad2smiles_file, 'r') as infile1:
    broad2smiles = json.load(infile1)
    
#%%
smiles = broad2smiles['BRD-K18438502-001-02-6']
mol = Chem.MolFromSmiles(smiles)
sentence = mol2alt_sentence(mol, 1)
substructure = sentence[0]
mol_vec = np.array([model.wv[substructure] for substructure in sentence])
