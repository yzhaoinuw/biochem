#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 13:46:02 2022

@author: yuezhao
"""

import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

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
# rows = broad2features['BRD-A00100033-001-04-8']
# features[rows]

sample_count = {broad: len(row_inds) for broad, row_inds in broad2features.items()}
sorted_count = sorted(sample_count.items(), key=lambda kv: -kv[1])

features_standardized = stats.zscore(features, axis=1, ddof=1)
#%%

N = 50
pca = PCA(n_components=N)
model = pca.fit(features_standardized)

main_components = (model.explained_variance_ratio_ > 0.1).nonzero()[0]
main_components = np.absolute(model.components_[main_components])
main_feature_inds = np.unique(np.transpose((main_components > 0.1).nonzero())[:, -1])
main_features = feature_names[main_feature_inds]
print(main_features)
