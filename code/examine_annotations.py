#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:50:21 2022

@author: yuezhao
"""

import os
import json
from collections import defaultdict

import pandas as pd

DATA_PATH = "../data/"
annotation_file = "chemical_annotations.csv"
profile_file = "profile_merged.csv"

save_file = DATA_PATH + "broad2smiles.json"

annotation_cols = ["BROAD_ID", "CPD_NAME", "CPD_SMILES"]
profile_cols = ["Metadata_broad_sample"]

df_annotation = pd.read_csv(DATA_PATH + annotation_file, usecols=annotation_cols)
# df_annotation = df_annotation.dropna()

df_profile = pd.read_csv(DATA_PATH + profile_file, usecols=profile_cols)
# df_profile = df_profile.dropna()
#%%
# df_profile['prod_type'] = df_profile['prod_type'].replace({'respon':'responsive', 'r':'responsive'})
# df2 = df_profile[df_profile['Metadata_broad_sample'] != df_profile['Metadata_pert_mfc_id']]
broad_id = set(df_annotation["BROAD_ID"].unique())
broad_sample = set(df_profile["Metadata_broad_sample"].unique())
intersect = broad_id.intersection(broad_sample)

broad2smiles = dict(zip(df_annotation["BROAD_ID"], df_annotation["CPD_SMILES"]))

"""
with open(save_file, 'w') as outfile1:
    json.dump(broad2smiles, outfile1)

name2smiles = defaultdict(set)
for cpd_name, smiles in zip(df_annotation['CPD_NAME'], df_annotation['CPD_SMILES']):
    name2smiles[cpd_name].add(smiles)
    
#%%
n = 1
for cpd_name, smiles in name2smiles.items():
    if len(smiles) > 1:
        print (f'{n}: {cpd_name}, {smiles}')
        n += 1
        print ()
"""
