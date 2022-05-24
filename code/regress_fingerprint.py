# -*- coding: utf-8 -*-
"""
Created on Tue May 17 08:47:26 2022

@author: Yue
"""

import json
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

DATA_PATH = '../data/'

broad2features_file = 'broad2features.json'
broad2smiles_file = 'broad2fingerprints.json'
features_file = 'features.npy'


with open(DATA_PATH+broad2features_file, 'r') as infile1:
    broad2features = json.load(infile1)
    
with open(DATA_PATH+broad2smiles_file, 'r') as infile1:
    broad2fingerprint = json.load(infile1)

features = np.load(DATA_PATH+features_file)

#%%
X = []
Y = []

for broad_id, fingerprint in broad2fingerprint.items():
    # turn fingerprint (bit string) into a numpy array
    if fingerprint is None:
        continue
    row_indices = broad2features[broad_id]
    for row_ind in row_indices:
        cell_area = features[row_ind, 0]
        cytoplasm_area = features[row_ind, 596]
        nuclei_area = features[row_ind, 1178]
        Y.append(np.array([cell_area, cytoplasm_area, nuclei_area]))
        X.append(np.array(list(fingerprint)).astype(float))
        
#%%
X = np.array(X)
Y = np.array(Y)

feature_ind = 2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
regr = MLPRegressor(random_state=1,
                    hidden_layer_sizes=(300, 100),
                    batch_size=32,
                    early_stopping=True,
                    verbose=True,
                    ).fit(X_train, Y_train[:, feature_ind]) # Cells_AreaShape_Area

print (f"R squared: {regr.score(X_test, Y_test[:,feature_ind])}")
#%%
