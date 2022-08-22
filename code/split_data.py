# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 23:49:53 2022

@author: Yue
"""

import json
import numpy as np

DATA_PATH = "../data/"

broad2features_file = "broad2features.json"
broad2smiles_file = "broad2smiles.json"
features_file = "features.npy"
feature_name_file = "feature_names.json"

with open(DATA_PATH + broad2features_file, "r") as infile1:
    broad2features = json.load(infile1)

with open(DATA_PATH + broad2smiles_file, "r") as infile1:
    broad2smiles = json.load(infile1)

with open(DATA_PATH + feature_name_file, "r") as infile1:
    feature_names = json.load(infile1)

feature_names = np.array(feature_names)
features = np.load(DATA_PATH + features_file)

#%%

MIN_SAMPLE_COUNT = 4
MAX_SAMPLE_COUNT = 8
VALID_SIZE = 0.2
TEST_SIZE = 0.2
TRAIN_SIZE = 1-(TEST_SIZE+VALID_SIZE)

train_inds = []
valid_is_inds = []
valid_os_inds = []
test_is_inds = []
test_os_inds = []

for broad_id, smiles in broad2smiles.items():

    row_indices = broad2features[broad_id]

    l = min(len(row_indices), MAX_SAMPLE_COUNT)
    if l < MIN_SAMPLE_COUNT:
        continue
    np.random.shuffle(row_indices)
    row_indices = row_indices[:l]
        
    test_prob = np.random.uniform()
    if test_prob < TRAIN_SIZE: 
        for row_index in row_indices:
            test_prob = np.random.uniform()
            if test_prob < TRAIN_SIZE:
                
                train_inds.append((broad_id, row_index))
            elif test_prob > (1-TEST_SIZE):
                test_is_inds.append((broad_id, row_index))
            else:
                valid_is_inds.append((broad_id, row_index))

    else:
        for row_index in row_indices:
            if test_prob > (1-TEST_SIZE):
                test_os_inds.append((broad_id, row_index))
            else:
                label = "validation_os"
                valid_os_inds.append((broad_id, row_index))


#%%
train_test_split = {"train": train_inds,
                    "validation_is": valid_is_inds,
                    "validation_os": valid_os_inds,
                    "test_is": test_is_inds,
                    "test_os": test_os_inds}
'''
WRITE_LOC = "../data/"
with open(WRITE_LOC+f"biochem_dataset_train_size{TRAIN_SIZE}_min_sample_count{MIN_SAMPLE_COUNT}.json", "w") as outfile:
    json.dump(train_test_split, outfile, indent=4)
'''